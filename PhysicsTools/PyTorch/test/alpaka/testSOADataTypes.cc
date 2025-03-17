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

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

// Next Steps: Create Class and improve readability and user handling

class testSOADataTypes : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testSOADataTypes);
  CPPUNIT_TEST(test_pose_SOA);
  CPPUNIT_TEST_SUITE_END();

public:
  void test_pose_SOA();
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



struct MaskElement {
  torch::ScalarType type;
  int bytes;
  int columns;
  bool used;

  MaskElement(torch::ScalarType type_, int columns_, bool used_): type(type_),columns(columns_), used(used_){ 
    bytes = torch::elementSize(type);
  }
};

class Mask {
  public: 
    int nBlocks;
    int nElements;

    Mask(int nElements_, std::vector<torch::ScalarType> types, std::vector<int> columns, std::vector<bool> used) {
      nElements = nElements_;
      nBlocks = std::min({types.size(), columns.size(), used.size()});
      blocks.reserve(nBlocks);

      for (int i = 0; i < nBlocks; i++) {
        blocks.push_back(MaskElement(types[i], columns[i], used[i]));
      }
    }

    MaskElement operator [](int i) const {return blocks[i];}

  private: 
    std::vector<MaskElement> blocks;
};

template <typename SOA_Layout>
class Converter {
  public:
    static std::vector<torch::Tensor> convert_multiple(Mask mask, torch::Device device, std::byte* arr);
    static torch::Tensor convert_single(int nElements, MaskElement element, torch::Device device, std::byte* arr);

  private:
    static std::array<long int, 2> soa_get_stride(int nElements, int bytes);
    static std::array<long int, 2> soa_get_size(int nElements, int bytes);

    template <size_t N>
    static torch::Tensor array_to_tensor(torch::Device device, torch::ScalarType type, std::byte* arr, const long int* size, const long int* stride);

};

template <typename SOA_Layout>
std::array<long int, 2> Converter<SOA_Layout>::soa_get_stride(int nElements, int bytes) {
  int per_bunch = SOA_Layout::alignment/bytes;
  int bunches = std::ceil(1.0 * nElements/per_bunch);
  std::array<long int, 2> stride{{1, bunches * per_bunch}};
  return stride;
}

template <typename SOA_Layout>
std::array<long int, 2> Converter<SOA_Layout>::soa_get_size(int nElements, int cols) {
  std::array<long int, 2> size{{nElements, cols}};
  return size;
}

template <typename SOA_Layout>
template <size_t N>
torch::Tensor Converter<SOA_Layout>::array_to_tensor(torch::Device device, torch::ScalarType type, std::byte* arr, const long int* size, const long int* stride) {
  long int arr_size[N];
  long int arr_stride[N];
  std::copy(size, size+N, arr_size);
  std::copy(stride, stride+N, arr_stride);

  auto options = torch::TensorOptions().dtype(type).device(device).pinned_memory(true);
  return torch::from_blob(arr, arr_size, arr_stride, options);
}

template <typename SOA_Layout>
std::vector<torch::Tensor> Converter<SOA_Layout>::convert_multiple(Mask mask, torch::Device device, std::byte* arr) {
  std::vector<torch::Tensor> tensors;
  int skip = 0;
  std::array<long int, 2> stride;
  std::array<long int, 2> size;
  torch::Tensor tensor;

  for (int i = 0; i < mask.nBlocks; i++) {
    std::cout << "Block " << i << std::endl;
    stride = Converter<SOA_Layout>::soa_get_stride(mask.nElements, mask[i].bytes);
    std::cout << "Stride: {" << stride[0] << ", " << stride[1] << "}" << std::endl;

    if(mask[i].used) {
      size = Converter<SOA_Layout>::soa_get_size(mask.nElements, mask[i].columns);
      std::cout << "Size: {" << size[0] << ", " << size[1] << "}" << std::endl;

      tensor = Converter<SOA_Layout>::array_to_tensor<2>(device, mask[i].type, arr+skip, size.data(), stride.data());
      tensors.push_back(tensor);
    }
    skip += mask[i].columns * stride[1] * mask[i].bytes;
  }
  return tensors;
}

template <typename SOA_Layout>
torch::Tensor Converter<SOA_Layout>::convert_single(int nElements, MaskElement mask, torch::Device device, std::byte* arr) {
  std::array<long int, 2> stride = Converter<SOA_Layout>::soa_get_stride(nElements, mask.bytes);
  std::cout << "Stride: {" << stride[0] << ", " << stride[1] << "}" << std::endl;
  std::array<long int, 2> size = Converter<SOA_Layout>::soa_get_size(nElements, mask.columns);
  std::cout << "Size: {" << size[0] << ", " << size[1] << "}" << std::endl;

  return Converter<SOA_Layout>::array_to_tensor<2>(device, mask.type, arr, size.data(), stride.data());
}


void testSOADataTypes::test_pose_SOA() {
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

  // Run Converter for single tensor
  torch::Tensor tensor = Converter<SoAPose>::convert_single(batch_size, MaskElement(torch::kDouble, 3, true), torchDevice, poseCollection.buffer().data());
  std::cout << tensor << std::endl;

  for (size_t i = 0; i < batch_size; i++) {
    CPPUNIT_ASSERT(std::abs(poseHostCollectionView.x()[i] - tensor[i][0].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(poseHostCollectionView.y()[i] - tensor[i][1].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(poseHostCollectionView.z()[i] - tensor[i][2].item<double>()) <= 1.0e-05);
  }

};