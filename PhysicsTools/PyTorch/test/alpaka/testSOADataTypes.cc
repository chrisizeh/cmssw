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
  CPPUNIT_TEST(test_multiple_SOA);
  CPPUNIT_TEST(test_single_SOA);
  CPPUNIT_TEST_SUITE_END();

public:
  void test_multiple_SOA();
  void test_single_SOA();
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

void testSOADataTypes::test_multiple_SOA() {
  Platform platform;
  std::vector<Device> alpakaDevices = alpaka::getDevs(platform);
  int idx = 0;
  for (const auto& d : alpakaDevices) {
    std::cout << "Device[" << idx++ << "]:   " << alpaka::getName(d) << std::endl;
  }
  const auto& alpakaHost = alpaka::getDevByIdx(alpaka_common::PlatformHost(), 0u);
  CPPUNIT_ASSERT(alpakaDevices.size());
  const auto& alpakaDevice = alpakaDevices[0];
  Queue queue{alpakaDevice};

  std::cout << "Will create torch device with type=" << torch_common::kDeviceType
       << " and native handle=" << alpakaDevice.getNativeHandle() << std::endl;
  torch::Device torchDevice(torch_common::kDeviceType, alpakaDevice.getNativeHandle());

  // Simple SOA with one bunch filled.
  const std::size_t batch_size = 35;
  std::random_device rd; 
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distrib(0, 2*M_PI);

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
  Mask mask(batch_size, {{torch::kDouble, torch::kInt, torch::kFloat}}, {{3, 1, 3}}, {{true, false, true}});
  std::vector<torch::Tensor> result = Converter<SoAPose>::convert_multiple(mask, torchDevice, poseCollection.buffer().data());
  std::cout << "Length of result vector: " << result.size() << std::endl;
  CPPUNIT_ASSERT(result.size() == 2);

  for (size_t i = 0; i < result.size(); i++) {
    std::cout << result[i] << std::endl;
  }

  for (size_t i = 0; i < batch_size; i++) {
    CPPUNIT_ASSERT(std::abs(poseHostCollectionView.x()[i] - result[0][i][0].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(poseHostCollectionView.y()[i] - result[0][i][1].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(poseHostCollectionView.z()[i] - result[0][i][2].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(poseHostCollectionView.phi()[i] - result[1][i][0].item<float>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(poseHostCollectionView.psi()[i] - result[1][i][1].item<float>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(poseHostCollectionView.theta()[i] - result[1][i][2].item<float>()) <= 1.0e-05);
  }
};


void testSOADataTypes::test_single_SOA() {
  Platform platform;
  std::vector<Device> alpakaDevices = alpaka::getDevs(platform);
  int idx = 0;
  for (const auto& d : alpakaDevices) {
    std::cout << "Device[" << idx++ << "]:   " << alpaka::getName(d) << std::endl;
  }
  const auto& alpakaHost = alpaka::getDevByIdx(alpaka_common::PlatformHost(), 0u);
  CPPUNIT_ASSERT(alpakaDevices.size());
  const auto& alpakaDevice = alpakaDevices[0];
  Queue queue{alpakaDevice};

  std::cout << "Will create torch device with type=" << torch_common::kDeviceType
       << " and native handle=" << alpakaDevice.getNativeHandle() << std::endl;
  torch::Device torchDevice(torch_common::kDeviceType, alpakaDevice.getNativeHandle());

  // Simple SOA with one bunch filled.
  const std::size_t batch_size = 35;
  std::random_device rd; 
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distrib(0, 2*M_PI);

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

  // Run Converter for single tensor
  torch::Tensor tensor = Converter<SoAPose>::convert_single(batch_size, MaskElement(torch::kDouble, 3, true), torchDevice, poseCollection.buffer().data());
  std::cout << tensor << std::endl;

  for (size_t i = 0; i < batch_size; i++) {
    CPPUNIT_ASSERT(std::abs(poseHostCollectionView.x()[i] - tensor[i][0].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(poseHostCollectionView.y()[i] - tensor[i][1].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(poseHostCollectionView.z()[i] - tensor[i][2].item<double>()) <= 1.0e-05);
  }
};