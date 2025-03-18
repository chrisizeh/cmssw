#include <torch/torch.h>
#include <torch/script.h>
#include <math.h>
#include <random>
#include <any>

#include <cppunit/extensions/HelperMacros.h>

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#endif

#include "PhysicsTools/PyTorch/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

#include "../Converter.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

class testSOADataTypes : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testSOADataTypes);
  CPPUNIT_TEST(test_input_convert);
  CPPUNIT_TEST(test_output_convert);
  CPPUNIT_TEST_SUITE_END();

public:
  void test_input_convert();
  void test_output_convert();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testSOADataTypes);

GENERATE_SOA_LAYOUT(SOAPoseTemplate,
                    SOA_COLUMN(double, x),
                    SOA_COLUMN(double, y),
                    SOA_COLUMN(double, z),
                    SOA_COLUMN(int, t),
                    SOA_COLUMN(float, phi),
                    SOA_COLUMN(float, psi),
                    SOA_COLUMN(float, theta))

using SoAPose = SOAPoseTemplate<>;
using SoAPoseView = SoAPose::View;

void testSOADataTypes::test_input_convert() {
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

  std::cout << "Will create torch device with type=" << torch_common::kDeviceType << std::endl;
  torch::Device torchDevice(torch_common::kDeviceType);

  // Simple SOA with one bunch filled.
  const std::size_t batch_size = 35;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distrib(0, 2 * M_PI);

  // Create and fill needed portable collections
  PortableHostCollection<SoAPose> poseHostCollection(batch_size, cms::alpakatools::host());
  PortableCollection<SoAPose, Device> poseCollection(batch_size, alpakaDevice);
  SoAPoseView& poseHostCollectionView = poseHostCollection.view();

  for (size_t i = 0; i < batch_size; i++) {
    poseHostCollectionView.x()[i] = 12 + i;
    poseHostCollectionView.y()[i] = 2.5 * i;
    poseHostCollectionView.z()[i] = 36 * i;
    poseHostCollectionView.phi()[i] = distrib(gen);
    poseHostCollectionView.psi()[i] = distrib(gen);
    poseHostCollectionView.theta()[i] = distrib(gen);
    poseHostCollectionView.t()[i] = i;
  }
  alpaka::memcpy(queue, poseCollection.buffer(), poseHostCollection.buffer());

  // Run Converter for multiple tensors
  InputMetadata input({{torch::kDouble, torch::kInt, torch::kFloat}}, {{3, 1, 3}}, {{0, -1, 1}});
  ModelMetadata metadata(batch_size, input, {torch::kInt, 1});
  std::vector<torch::IValue> result =
      Converter<SoAPose>::convert_input(metadata, torchDevice, poseCollection.buffer().data());
  std::cout << "Length of result vector: " << result.size() << std::endl;
  CPPUNIT_ASSERT(result.size() == 2);

  for (size_t i = 0; i < result.size(); i++) {
    std::cout << result[i] << std::endl;
  }

  for (size_t i = 0; i < batch_size; i++) {
    CPPUNIT_ASSERT(std::abs(poseHostCollectionView.x()[i] - result[0].toTensor()[i][0].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(poseHostCollectionView.y()[i] - result[0].toTensor()[i][1].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(poseHostCollectionView.z()[i] - result[0].toTensor()[i][2].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(poseHostCollectionView.phi()[i] - result[1].toTensor()[i][0].item<float>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(poseHostCollectionView.psi()[i] - result[1].toTensor()[i][1].item<float>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(poseHostCollectionView.theta()[i] - result[1].toTensor()[i][2].item<float>()) <= 1.0e-05);
  }
};

void testSOADataTypes::test_output_convert() {
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

  std::cout << "Will create torch device with type=" << torch_common::kDeviceType << std::endl;
  torch::Device torchDevice(torch_common::kDeviceType);

  // Simple SOA with one bunch filled.
  const std::size_t batch_size = 35;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distrib(0, 2 * M_PI);

  // Create and fill needed portable collections
  PortableHostCollection<SoAPose> poseHostCollection(batch_size, alpakaHost);
  PortableCollection<SoAPose, Device> poseCollection(batch_size, alpakaDevice);
  SoAPoseView& poseHostCollectionView = poseHostCollection.view();

  for (size_t i = 0; i < batch_size; i++) {
    poseHostCollectionView.x()[i] = 12 + i;
    poseHostCollectionView.y()[i] = 2.5 * i;
    poseHostCollectionView.z()[i] = 36 * i;
    poseHostCollectionView.phi()[i] = distrib(gen);
    poseHostCollectionView.psi()[i] = distrib(gen);
    poseHostCollectionView.theta()[i] = distrib(gen);
    poseHostCollectionView.t()[i] = i;
  }
  alpaka::memcpy(queue, poseCollection.buffer(), poseHostCollection.buffer());

  // Run Converter for single tensor
  InputMetadata input({{torch::kDouble, torch::kInt, torch::kFloat}}, {{3, 1, 3}}, {{true, false, true}});
  OutputMetadata output(torch::kDouble, 3);
  ModelMetadata metadata(batch_size, input, output);

  torch::Tensor tensor = Converter<SoAPose>::convert_output(metadata, torchDevice, poseCollection.buffer().data());
  std::cout << tensor << std::endl;

  for (size_t i = 0; i < batch_size; i++) {
    CPPUNIT_ASSERT(std::abs(poseHostCollectionView.x()[i] - tensor[i][0].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(poseHostCollectionView.y()[i] - tensor[i][1].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(poseHostCollectionView.z()[i] - tensor[i][2].item<double>()) <= 1.0e-05);
  }
};