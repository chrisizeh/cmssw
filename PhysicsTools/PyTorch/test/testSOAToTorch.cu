#include <cstdlib>
#include <cuda/std/array>
#include <cuda/std/ranges>
#include <iostream>
#include <chrono>
#include <array>

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/script.h>

#include "testBase.h"

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

template <std::size_t N>
torch::Tensor array_to_tensor(torch::Device device, float* arr, const long int* size) {
  long int arr_size[N];
  long int arr_stride[N];
  std::copy(size, size+N, arr_size);
  std::copy(size, size+N, arr_stride);

  std::shift_right(std::begin(arr_stride), std::end(arr_stride), 1);
  arr_stride[0] = 1;
  arr_stride[N-1] *= arr_stride[N-2];

  auto options = torch::TensorOptions().dtype(torch::kFloat).device(device).pinned_memory(true);
  torch::Tensor tensor = torch::from_blob(arr, arr_size, arr_stride, options);

  return tensor;
}

template <std::size_t N>
void print_column_major(float* arr, const long int* size) {
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
}


void testSOAToTorch::test() {
  torch::Device device(torch::kCUDA);

  float input_cpu[] = {1, 4, 1, 2, 3, 2, 1, 3, 3, 2, 4, 2};
  long int shape[] = {3, 4};
  const size_t dims = 2;

  std::array<float, 3> result_cpu{};
  std::array<float, 3> result_check{{3.1f, 7.8f, 7.1f}};

  // Prints array in correct form.
  print_column_major<dims>(input_cpu, shape);

  float *input_gpu, *result_gpu;
  cudaMalloc(&input_gpu, sizeof(input_cpu));
  cudaMalloc(&result_gpu, sizeof(result_cpu));
  cudaMemcpy(input_gpu, input_cpu, sizeof(input_cpu), cudaMemcpyHostToDevice);

  // Use stride to read correctly.
  std::cout << "Converting vector to Torch tensors on CPU with stride" << std::endl;
  torch::Tensor input_tensor = array_to_tensor<dims>(device, input_gpu, shape);

  // Load the TorchScript model
  std::string model_path = dataPath_ + "/linear_dnn.pt";

  torch::jit::script::Module model;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load(model_path);
    model.to(device);
    std::vector<torch::jit::IValue> inputs{input_tensor};
    auto options = torch::TensorOptions().dtype(torch::kFloat).device(device).pinned_memory(true);
    torch::from_blob(result_gpu, {3, 1}, options) = model.forward(inputs).toTensor();

  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n" << e.what() << std::endl;
  }

  // Compare if values are the same as for python script
  cudaMemcpy(result_cpu.data(), result_gpu, sizeof(result_cpu), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 3; i++) {
    CPPUNIT_ASSERT(std::abs(result_cpu[i] - result_check[i]) <= 1.0e-05);
  }

}
