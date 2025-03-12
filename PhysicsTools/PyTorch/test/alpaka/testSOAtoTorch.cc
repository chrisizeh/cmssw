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

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

using std::cout;
using std::endl;

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

template <typename T, std::size_t N>
torch::Tensor array_to_tensor(torch::Device device, T* arr, const long int* size) {
  long int arr_size[N];
  long int arr_stride[N];
  std::copy(size, size+N, arr_size);
  std::copy(size, size+N, arr_stride);

  std::shift_right(std::begin(arr_stride), std::end(arr_stride), 1);
  arr_stride[0] = 1;
  arr_stride[N-1] *= arr_stride[N-2];

  auto options = torch::TensorOptions().dtype(torch::CppTypeToScalarType<T>()).device(device).pinned_memory(true);
  return torch::from_blob(arr, arr_size, arr_stride, options);
}

template <typename T, std::size_t N>
void print_column_major(T* arr, const long int* size) {
  if (N == 2) {
    for (int i = 0; i < size[0]; i++) {
      for (int j = 0; j < size[1]; j++) {
          std::cout << arr[i + j*size[0]] << " ";
      }
      std::cout << std::endl;
    } 
  } else if (N == 3) {
    for (int i = 0; i < size[0]; i++) {
      std::cout << "(" << i << ", .., ..)" << std::endl;
      for (int j = 0; j < size[1]; j++) {
        for (int k = 0; k < size[2]; k++) {
          std::cout << arr[i + j*size[0] + k*size[0]*size[1]] << " ";
        }
        std::cout << std::endl;
      } 
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;
}


template <typename T, std::size_t N, std::size_t M>
void run(torch::Device device, torch::jit::script::Module model, T* input, const long int* input_shape, T* output, const long int* output_shape) {
  torch::Tensor input_tensor = array_to_tensor<T, N>(device, input, input_shape);

  std::vector<torch::jit::IValue> inputs{input_tensor};
  array_to_tensor<T, M>(device, output, output_shape) = model.forward(inputs).toTensor();
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
  auto alpakaDevices = alpaka::getDevs(platform);
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


  std::vector<float> input{{1, 2, 3, 2, 2, 4, 4, 3, 1, 3, 1, 2}};
  const long int shape[] = {4, 3};
  const int size_input = 12;

  auto input_cpu = cms::alpakatools::make_host_buffer<float[]>(queue, size_input);
  for (int i = 0; i < size_input; ++i) {
    input_cpu[i] = input[i];
  }

  // Prints array in correct form.
  std::cout << "Input Matrix:" << std::endl;
  print_column_major<float, 2>(input_cpu.data(), shape);

  const int size_result = 8;
  std::array<float, size_result> result_cpu{};
  // float result_cpu[size_result];
  float result_check[4][2] = {{2.3, -0.5}, {6.6, 3.0}, {2.5, -4.9}, {4.4, 1.3}};
  const long int result_shape[] = {4, 2};

  auto result_gpu = cms::alpakatools::make_device_buffer<float[]>(queue, size_result);
  auto input_gpu = cms::alpakatools::make_device_buffer<float[]>(queue, size_input);
  alpaka::memcpy(queue, input_gpu, input_cpu);

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
  run<float, 2, 2>(torchDevice, model, input_gpu.data(), shape, result_gpu.data(), result_shape);

  // // Compare if values are the same as for python script
  alpaka::memcpy(queue, result_cpu, result_gpu);
  alpaka::wait(queue);

  std::cout << "Output Matrix:" << std::endl;
  print_column_major<float, 2>(result_cpu.data(), result_shape);

  for (int i = 0; i < result_shape[0]; i++) {
    for (int j = 0; j < result_shape[1]; j++) {
      CPPUNIT_ASSERT(std::abs(result_cpu[i + j*result_shape[0]] - result_check[i][j]) <= 1.0e-05);
    }
  }
  

}
