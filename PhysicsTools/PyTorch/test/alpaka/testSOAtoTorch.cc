#include <alpaka/alpaka.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <c10/cuda/CUDAStream.h>
#endif
#include <iostream>
#include <exception>
#include <memory>
#include <math.h>
#include <sys/prctl.h>
#include "../testBase.h"
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

#include "PhysicsTools/PyTorch/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

using std::cout;
using std::endl;

// Input SOA
GENERATE_SOA_LAYOUT(SoAPositionTemplate,
  SOA_COLUMN(float, x),
  SOA_COLUMN(float, y),
  SOA_COLUMN(float, z))

using SoAPosition = SoAPositionTemplate<>;
using SoAPositionView = SoAPosition::View;
using SoAPositionConstView = SoAPosition::ConstView;


// Output SOA
GENERATE_SOA_LAYOUT(SoAResultTemplate,
  SOA_COLUMN(float, x),
  SOA_COLUMN(float, y))

using SoAResult = SoAResultTemplate<>;
using SoAResultView = SoAResult::View;

class testSOAToTorch : public testBasePyTorch {
  CPPUNIT_TEST_SUITE(testSOAToTorch);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

public:
  std::string pyScript() const override;
  void test() override;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testSOAToTorch);

std::string testSOAToTorch::pyScript() const { return "create_linear_dnn.py"; }

// Create tensor from SOA based on size = {row, column} and alignment
template <typename T, std::size_t N>
torch::Tensor array_to_tensor(torch::Device device, std::byte* arr, const long int* size, size_t alignment) {
  long int arr_size[N];
  long int arr_stride[N];
  std::copy(size, size+N, arr_size);
  std::copy(size, size+N, arr_stride);
  int per_column = alignment/sizeof(T);

  arr_stride[0] = 1;
  for (size_t i = 1; i < N; i++) {
    arr_stride[i] = arr_stride[i-1]*per_column;
  }

  auto options = torch::TensorOptions().dtype(torch::CppTypeToScalarType<T>()).device(device).pinned_memory(true);
  return torch::from_blob(arr, arr_size, arr_stride, options);
}


// Build Tensor, run model end return pointer to buffer with correct alignment
template <typename T, std::size_t N, std::size_t M>
void run(torch::Device device, torch::jit::script::Module model, std::byte* input, const long int* input_shape, size_t input_alignment, std::byte* output, const long int* output_shape, size_t output_alignment) {
  torch::Tensor input_tensor = array_to_tensor<T, N>(device, input, input_shape, input_alignment);

  std::vector<torch::jit::IValue> inputs{input_tensor};
  array_to_tensor<T, M>(device, output, output_shape, output_alignment) = model.forward(inputs).toTensor();
}


void testSOAToTorch::test() {
  cout << "ALPAKA Platform info:" << endl;
  int idx = 0;
  try {
    for (;;) {
      alpaka::Platform<alpaka::DevCpu> platformHost;
      alpaka::DevCpu host = alpaka::getDevByIdx(platformHost, idx);
      cout << "Host[" << idx++ << "]:   " << alpaka::getName(host) << endl;
    }
  } catch (...) {
  }
  Platform platform;
  std::vector<Device> alpakaDevices = alpaka::getDevs(platform);
  idx = 0;
  for (const auto& d : alpakaDevices) {
    cout << "Device[" << idx++ << "]:   " << alpaka::getName(d) << endl;
  }
  const auto& alpakaHost = alpaka::getDevByIdx(alpaka_common::PlatformHost(), 0u);
  CPPUNIT_ASSERT(alpakaDevices.size());
  const auto& alpakaDevice = alpakaDevices[0];
  Queue queue{alpakaDevice};

  cout << "Will create torch device with type=" << torch_common::kDeviceType
       << " and native handle=" << alpakaDevice.getNativeHandle() << endl;
  torch::Device torchDevice(torch_common::kDeviceType, alpakaDevice.getNativeHandle());

  // Number of elements
  const std::size_t batch_size = 4;

  std::vector<float> input{{1, 2, 3, 2, 2, 4, 4, 3, 1, 3, 1, 2}};
  const long int shape[] = {4, 3};

  float result_check[4][2] = {{2.3, -0.5}, {6.6, 3.0}, {2.5, -4.9}, {4.4, 1.3}};
  const long int result_shape[] = {4, 2};

  // Create and fill needed portable collections
  PortableHostCollection<SoAPosition> positionHostCollection(batch_size, cms::alpakatools::host());
  PortableCollection<SoAPosition, Device> positionCollection(batch_size, alpakaDevice);
  SoAPositionView& positionCollectionView = positionHostCollection.view();
  
  for (size_t i = 0; i < batch_size; i++) {
    positionCollectionView.x()[i] = input[i];
    positionCollectionView.y()[i] = input[i + 1*batch_size];
    positionCollectionView.z()[i] = input[i + 2*batch_size];  
  }
  alpaka::memcpy(queue, positionCollection.buffer(), positionHostCollection.buffer());

  PortableHostCollection<SoAResult> resultHostCollection(batch_size, cms::alpakatools::host());
  PortableCollection<SoAResult, Device> resultCollection(batch_size, alpakaDevice);

  torch::jit::script::Module model;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    std::string model_path = dataPath_ + "/linear_dnn.pt";
    model = torch::jit::load(model_path);
    model.to(torchDevice);

  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n" << e.what() << std::endl;
  }
  
  // Call function to build tensor and run model
  run<float, 2, 2>(torchDevice, model, positionCollection.buffer().data(), shape, SoAPosition::alignment, resultCollection.buffer().data(), result_shape, SoAResult::alignment);

  // Compare if values are the same as for python script
  alpaka::memcpy(queue, resultHostCollection.buffer(), resultCollection.buffer());
  alpaka::wait(queue);

  SoAResultView& resultView = resultHostCollection.view();

  std::cout << "Output Matrix:" << std::endl;
  for (size_t i = 0; i < batch_size; i++) {
    std::cout << resultView.x()[i] << " " << resultView.y()[i] << std::endl;

    CPPUNIT_ASSERT(std::abs(resultView.x()[i] - result_check[i][0]) <= 1.0e-05);
    CPPUNIT_ASSERT(std::abs(resultView.y()[i] - result_check[i][1]) <= 1.0e-05);
  }
  

}
