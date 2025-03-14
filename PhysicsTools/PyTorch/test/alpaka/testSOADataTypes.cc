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

struct MaskElement {
  torch::ScalarType type;
  int bytes;
  int columns;
  bool used;

  MaskElement(torch::ScalarType type_, int columns_, bool used_): type(type_),columns(columns_), used(used_){ 
    bytes = torch::elementSize(type);
  }
};

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

class testSOADataTypes : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testSOADataTypes);
  CPPUNIT_TEST(test_pose_SOA);
  CPPUNIT_TEST_SUITE_END();

public:
  void test_pose_SOA();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testSOADataTypes);

std::array<long int, 2> soa_get_stride(int nelements, long int alignment, int bytes) {
  int per_bunch = alignment/bytes;
  int bunches = std::ceil(1.0 * nelements/per_bunch);
  std::array<long int, 2> stride{{1, bunches * per_bunch}};
  return stride;
}

template <typename SOA_Layout>
std::array<long int, 2> soa_get_size(int rows, int bytes, int cols=-1) {
  int columns;

  if (cols <= 0) {
    // Method for column count will be added -> until then, calculated
    int per_bunch = SOA_Layout::alignment/bytes;
    int bunches = std::ceil(1.0 * rows/per_bunch);
    int byteSize = SOA_Layout::computeDataSize(rows);
    columns = byteSize/(bunches * SOA_Layout::alignment);  
  } else {
    columns = cols;
  }

  std::array<long int, 2> size{{rows, columns}};
  return size;
}

// Create tensor from SOA based on size = {row, column} and alignment
template <std::size_t N>
torch::Tensor array_to_tensor(torch::Device device, torch::ScalarType type, std::byte* arr, const long int* size, const long int* stride) {
  long int arr_size[N];
  long int arr_stride[N];
  std::copy(size, size+N, arr_size);
  std::copy(stride, stride+N, arr_stride);

  auto options = torch::TensorOptions().dtype(type).device(device).pinned_memory(true);
  return torch::from_blob(arr, arr_size, arr_stride, options);
}


void testSOADataTypes::test_pose_SOA() {
  std::cout << "SOA with multiple bunches filled and 7 rows." << std::endl;
  torch::Device device(torch::kCPU);

  // Simple SOA with one bunch filled.
  const std::size_t batch_size = 35;
  std::random_device rd; 
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distrib(0, 2*M_PI);

  // Create and fill needed portable collections
  PortableCollection<SoAPose, DevHost> poseCollection(batch_size, cms::alpakatools::host());
  SoAPoseView& poseCollectionView = poseCollection.view();

  for (size_t i = 0; i < batch_size; i++) {
    poseCollectionView.x()[i] = 12 + i;
    poseCollectionView.y()[i] = 2.5 * i;
    poseCollectionView.z()[i] = 36 * i;  
    poseCollectionView.phi()[i] = distrib(gen);  
    poseCollectionView.psi()[i] = distrib(gen);  
    poseCollectionView.theta()[i] = distrib(gen);  
    poseCollectionView.t()[i] = i;  
  }

  const int numBlocks = 3;
  std::array<MaskElement, numBlocks> mask{{MaskElement(torch::kDouble, 3, true), MaskElement(torch::kInt, 1, false), MaskElement(torch::kFloat, 3, true)}};

  int skip = 0;
  std::array<long int, 2> stride;
  std::array<long int, 2> size;
  torch::Tensor tensor;

  for (int i = 0; i < numBlocks; i++) {
    std::cout << "Block " << i << mask[i].bytes << std::endl;
    stride = soa_get_stride(batch_size, SoAPose::alignment, mask[i].bytes);
    std::cout << "Stride: {" << stride[0] << ", " << stride[1] << "}" << std::endl;

    if(mask[i].used) {
      size = soa_get_size<SoAPose>(poseCollectionView.metadata().size(), mask[i].bytes, mask[i].columns);
      std::cout << "Size: {" << size[0] << ", " << size[1] << "}" << std::endl;

      tensor = array_to_tensor<2>(device, mask[i].type, poseCollection.buffer().data()+skip, size.data(), stride.data());
      std::cout << tensor << std::endl;
    } 
      
    skip += mask[i].columns * stride[1] * mask[i].bytes;
  }

};