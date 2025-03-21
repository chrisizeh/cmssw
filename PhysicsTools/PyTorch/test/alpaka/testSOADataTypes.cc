#include <torch/torch.h>
#include <torch/script.h>
#include <math.h>
#include <random>
#include <Eigen/Dense>

#include <cppunit/extensions/HelperMacros.h>

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#endif

#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

#include "PhysicsTools/PyTorch/interface/AlpakaConfig.h"
#include "PhysicsTools/PyTorch/interface/Converter.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace torch_alpaka;

class testSOADataTypes : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testSOADataTypes);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

public:
  void test();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testSOADataTypes);

GENERATE_SOA_LAYOUT(SoATemplate,
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, a),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, b),

                    SOA_EIGEN_COLUMN(Eigen::Matrix2f, c),

                    SOA_COLUMN(double, x),
                    SOA_COLUMN(double, y),
                    SOA_COLUMN(double, z),

                    SOA_SCALAR(float, type),
                    SOA_SCALAR(int, someNumber));

using SoA = SoATemplate<>;
using SoAView = SoA::View;

void testSOADataTypes::test() {
  std::cout << "ALPAKA Platform info:" << std::endl;
  int idx = 0;
  try {
    for (;;) {
      alpaka::Platform<alpaka::DevCpu> platformHost;
      alpaka::DevCpu host = alpaka::getDevByIdx(platformHost, idx);
      std::cout << "Host[" << idx++ << "]:   " << alpaka::getName(host) << std::endl;
    }
  } catch (...) {
  }
  Platform platform;
  std::vector<Device> alpakaDevices = alpaka::getDevs(platform);
  idx = 0;
  for (const auto& d : alpakaDevices) {
    std::cout << "Device[" << idx++ << "]:   " << alpaka::getName(d) << std::endl;
  }
  const auto& alpakaHost = alpaka::getDevByIdx(alpaka_common::PlatformHost(), 0u);
  CPPUNIT_ASSERT(alpakaDevices.size());
  const auto& alpakaDevice = alpakaDevices[0];
  Queue queue{alpakaDevice};

  std::cout << "Will create torch device with type=" << torch_alpaka::kDeviceType << std::endl;
  torch::Device torchDevice(torch_alpaka::kDeviceType);

  // Large batch size, so multiple bunches needed
  const std::size_t batch_size = 35;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distrib(0, 2 * M_PI);

  // Create and fill needed portable collections
  PortableHostCollection<SoA> hostCollection(batch_size, alpakaHost);
  PortableCollection<SoA, Device> deviceCollection(batch_size, alpakaDevice);
  SoAView& hostCollectionView = hostCollection.view();

  hostCollectionView.type() = 4;
  hostCollectionView.someNumber() = 5;

  for (size_t i = 0; i < batch_size; i++) {
    hostCollectionView[i].a()(0) = 1 + i;
    hostCollectionView[i].a()(1) = 2 + i;
    hostCollectionView[i].a()(2) = 3 + i;

    hostCollectionView[i].b()(0) = 4 + i;
    hostCollectionView[i].b()(1) = 5 + i;
    hostCollectionView[i].b()(2) = 6 + i;

    hostCollectionView[i].c()(0, 0) = 4 + i;
    hostCollectionView[i].c()(0, 1) = 6 + i;
    hostCollectionView[i].c()(1, 0) = 8 + i;
    hostCollectionView[i].c()(1, 1) = 10 + i;

    hostCollectionView.x()[i] = 12 + i;
    hostCollectionView.y()[i] = 2.5 * i;
    hostCollectionView.z()[i] = 36 * i;
  }

  alpaka::memcpy(queue, deviceCollection.buffer(), hostCollection.buffer());

  // Run Converter for single tensor
  InputMetadata input({Double, Float, Double, Float, Int}, {{{2, 3}}, {{1, 2, 2}}, 3, 0, 0});
  OutputMetadata output(Double, 3);
  ModelMetadata metadata(batch_size, input, output);

  std::vector<torch::IValue> tensor =
      Converter<SoA>::convert_input(metadata, torchDevice, deviceCollection.buffer().data());

  for (size_t i = 0; i < batch_size; i++) {
    // Block: Eigen Vector Columns Double
    CPPUNIT_ASSERT(std::abs(hostCollectionView[i].a()(0) - tensor[0].toTensor()[i][0][0].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(hostCollectionView[i].a()(1) - tensor[0].toTensor()[i][0][1].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(hostCollectionView[i].a()(2) - tensor[0].toTensor()[i][0][2].item<double>()) <= 1.0e-05);

    CPPUNIT_ASSERT(std::abs(hostCollectionView[i].b()(0) - tensor[0].toTensor()[i][1][0].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(hostCollectionView[i].b()(1) - tensor[0].toTensor()[i][1][1].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(hostCollectionView[i].b()(2) - tensor[0].toTensor()[i][1][2].item<double>()) <= 1.0e-05);

    // Block: Eigen Matrix Columns Float
    CPPUNIT_ASSERT(std::abs(hostCollectionView[i].c()(0, 0) - tensor[1].toTensor()[i][0][0][0].item<float>()) <=
                   1.0e-05);
    CPPUNIT_ASSERT(std::abs(hostCollectionView[i].c()(0, 1) - tensor[1].toTensor()[i][0][0][1].item<float>()) <=
                   1.0e-05);
    CPPUNIT_ASSERT(std::abs(hostCollectionView[i].c()(1, 0) - tensor[1].toTensor()[i][0][1][0].item<float>()) <=
                   1.0e-05);
    CPPUNIT_ASSERT(std::abs(hostCollectionView[i].c()(1, 1) - tensor[1].toTensor()[i][0][1][1].item<float>()) <=
                   1.0e-05);

    // Block: Columns Double
    CPPUNIT_ASSERT(std::abs(hostCollectionView.x()[i] - tensor[2].toTensor()[i][0].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(hostCollectionView.y()[i] - tensor[2].toTensor()[i][1].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(hostCollectionView.z()[i] - tensor[2].toTensor()[i][2].item<double>()) <= 1.0e-05);

    // Block: Scalar Float
    CPPUNIT_ASSERT(std::abs(hostCollectionView.type() - tensor[3].toTensor()[i].item<double>()) <= 1.0e-05);

    // Block: Scalar Int
    CPPUNIT_ASSERT(std::abs(hostCollectionView.someNumber() - tensor[4].toTensor()[i].item<int>()) == 0);
  }
};