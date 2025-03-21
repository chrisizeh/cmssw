#include <alpaka/alpaka.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <math.h>
#include <random>

#include <cppunit/extensions/HelperMacros.h>

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#endif

#include "PhysicsTools/PyTorch/interface/AlpakaConfig.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

using std::cout;
using std::endl;

// Simple SOA
GENERATE_SOA_LAYOUT(SoAPositionTemplate, SOA_COLUMN(int, x), SOA_COLUMN(int, y), SOA_COLUMN(int, z))

using SoAPosition = SoAPositionTemplate<>;
using SoAPositionView = SoAPosition::View;
using SoAPositionConstView = SoAPosition::ConstView;

// Large SOA
GENERATE_SOA_LAYOUT(SOAPoseTemplate,
                    SOA_COLUMN(double, x),
                    SOA_COLUMN(double, y),
                    SOA_COLUMN(double, z),
                    SOA_COLUMN(double, phi),
                    SOA_COLUMN(double, psi),
                    SOA_COLUMN(double, theta),
                    SOA_COLUMN(double, t))

using SoAPose = SOAPoseTemplate<>;
using SoAPoseView = SoAPose::View;

// Unit tests for Stride function on CPU.
class testSOAStride : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testSOAStride);
  CPPUNIT_TEST(test_position_SOA);
  CPPUNIT_TEST(test_multi_bunch);
  CPPUNIT_TEST(test_pose_SOA);
  CPPUNIT_TEST_SUITE_END();

public:
  void test_position_SOA();
  void test_multi_bunch();
  void test_pose_SOA();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testSOAStride);

template <typename T>
std::array<long int, 2> soa_get_stride(int nelements, long int alignment) {
  int per_bunch = alignment / sizeof(T);
  int bunches = std::ceil(1.0 * nelements / per_bunch);
  std::array<long int, 2> stride{{1, bunches * per_bunch}};
  return stride;
}

template <typename T, typename SOA_Layout>
std::array<long int, 2> soa_get_size(int rows) {
  // Method for column count will be added -> until then, calculated
  int per_bunch = SOA_Layout::alignment / sizeof(T);
  int bunches = std::ceil(1.0 * rows / per_bunch);
  int byteSize = SOA_Layout::computeDataSize(rows);
  int columns = byteSize / (bunches * SOA_Layout::alignment);

  std::array<long int, 2> size{{rows, columns}};
  return size;
}

// Create tensor from SOA based on size = {row, column} and alignment
template <typename T, std::size_t N>
torch::Tensor array_to_tensor(torch::Device device, std::byte* arr, const long int* size, const long int* stride) {
  long int arr_size[N];
  long int arr_stride[N];
  std::copy(size, size + N, arr_size);
  std::copy(stride, stride + N, arr_stride);

  auto options = torch::TensorOptions().dtype(torch::CppTypeToScalarType<T>()).device(device).pinned_memory(true);
  return torch::from_blob(arr, arr_size, arr_stride, options);
}

void testSOAStride::test_position_SOA() {
  std::cout << "SOA with int, one bunch filled and 3 rows." << std::endl;
  torch::Device device(torch::kCPU);

  // Simple SOA with one bunch filled.
  const std::size_t batch_size = 4;

  // Create and fill needed portable collections
  PortableCollection<SoAPosition, DevHost> positionCollection(batch_size, cms::alpakatools::host());
  SoAPositionView& positionCollectionView = positionCollection.view();

  for (size_t i = 0; i < batch_size; i++) {
    positionCollectionView.x()[i] = 12 + i;
    positionCollectionView.y()[i] = 3 * i;
    positionCollectionView.z()[i] = 36 * i;
  }

  std::array<long int, 2> size = soa_get_size<int, SoAPosition>(positionCollectionView.metadata().size());
  std::cout << "Size: {" << size[0] << ", " << size[1] << "}" << std::endl;

  auto stride = soa_get_stride<int>(batch_size, SoAPosition::alignment);
  std::cout << "Stride: {" << stride[0] << ", " << stride[1] << "}" << std::endl;

  CPPUNIT_ASSERT(stride[0] == 1);
  CPPUNIT_ASSERT(stride[1] == 32);

  // Check correct tensor creation with stride
  torch::Tensor tensor =
      array_to_tensor<int, 2>(device, positionCollection.buffer().data(), size.data(), stride.data());

  for (size_t i = 0; i < batch_size; i++) {
    CPPUNIT_ASSERT(positionCollectionView.x()[i] - tensor[i][0].item<int>() == 0);
    CPPUNIT_ASSERT(positionCollectionView.y()[i] - tensor[i][1].item<int>() == 0);
    CPPUNIT_ASSERT(positionCollectionView.z()[i] - tensor[i][2].item<int>() == 0);
  }
}

void testSOAStride::test_multi_bunch() {
  std::cout << "SOA with int, multiple bunches filled and 3 rows." << std::endl;
  torch::Device device(torch::kCPU);

  // Simple SOA with one bunch filled.
  const std::size_t batch_size = 54;

  // Create and fill needed portable collections
  PortableCollection<SoAPosition, DevHost> positionCollection(batch_size, cms::alpakatools::host());
  SoAPositionView& positionCollectionView = positionCollection.view();

  for (size_t i = 0; i < batch_size; i++) {
    positionCollectionView.x()[i] = 12 + i;
    positionCollectionView.y()[i] = 2 * i;
    positionCollectionView.z()[i] = 36 * i;
  }

  std::array<long int, 2> size = soa_get_size<int, SoAPosition>(positionCollectionView.metadata().size());
  std::cout << "Size: {" << size[0] << ", " << size[1] << "}" << std::endl;

  auto stride = soa_get_stride<int>(batch_size, SoAPosition::alignment);
  std::cout << "Stride: {" << stride[0] << ", " << stride[1] << "}" << std::endl;

  CPPUNIT_ASSERT(stride[0] == 1);
  CPPUNIT_ASSERT(stride[1] == 64);

  // Check correct tensor creation with stride
  torch::Tensor tensor =
      array_to_tensor<int, 2>(device, positionCollection.buffer().data(), size.data(), stride.data());

  for (size_t i = 0; i < batch_size; i++) {
    CPPUNIT_ASSERT(positionCollectionView.x()[i] - tensor[i][0].item<int>() == 0);
    CPPUNIT_ASSERT(positionCollectionView.y()[i] - tensor[i][1].item<int>() == 0);
    CPPUNIT_ASSERT(positionCollectionView.z()[i] - tensor[i][2].item<int>() == 0);
  }
}

void testSOAStride::test_pose_SOA() {
  std::cout << "SOA with multiple bunches filled and 7 rows." << std::endl;
  torch::Device device(torch::kCPU);

  // Simple SOA with one bunch filled.
  const std::size_t batch_size = 35;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distrib(0, 2 * M_PI);

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

  std::array<long int, 2> size = soa_get_size<double, SoAPose>(poseCollectionView.metadata().size());
  std::cout << "Size: {" << size[0] << ", " << size[1] << "}" << std::endl;

  auto stride = soa_get_stride<double>(batch_size, SoAPose::alignment);
  std::cout << "Stride: {" << stride[0] << ", " << stride[1] << "}" << std::endl;

  CPPUNIT_ASSERT(stride[0] == 1);
  CPPUNIT_ASSERT(stride[1] == 48);

  // Check correct tensor creation with stride
  torch::Tensor tensor = array_to_tensor<double, 2>(device, poseCollection.buffer().data(), size.data(), stride.data());

  for (size_t i = 0; i < batch_size; i++) {
    CPPUNIT_ASSERT(std::abs(poseCollectionView.x()[i] - tensor[i][0].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(poseCollectionView.y()[i] - tensor[i][1].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(poseCollectionView.z()[i] - tensor[i][2].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(poseCollectionView.phi()[i] - tensor[i][3].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(poseCollectionView.psi()[i] - tensor[i][4].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(poseCollectionView.theta()[i] - tensor[i][5].item<double>()) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(poseCollectionView.t()[i] - tensor[i][6].item<double>()) <= 1.0e-05);
  }
}