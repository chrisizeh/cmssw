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
  torch::Tensor tensor = torch::from_blob(arr, arr_size, arr_stride, options);

  return tensor;
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

  // from_blod doesn't work if use array from parameter list
  long int res_shape[M];
  std::copy(output_shape, output_shape+M, res_shape);

  std::vector<torch::jit::IValue> inputs{input_tensor};
  auto options = torch::TensorOptions().dtype(torch::CppTypeToScalarType<T>()).device(device).pinned_memory(true);
  // RESULT IS IN ROW MAJOR
  torch::from_blob(output, res_shape, options) = model.forward(inputs).toTensor();
}


void testSOAToTorch::test() {
  torch::Device device(torch::kCUDA);
  
  float input_cpu[] = {1, 2, 3, 2, 2, 4, 4, 3, 1, 3, 1, 2};
  const long int shape[] = {4, 3};

  float result_cpu[4][2];
  float result_check[4][2] = {{2.3, -0.5}, {6.6, 3.0}, {2.5, -4.9}, {4.4, 1.3}};
  const long int result_shape[] = {4, 2};

  // Prints array in correct form.
  print_column_major<float, 2>(input_cpu, shape);

  float *input_gpu, *result_gpu;
  cudaMalloc(&input_gpu, sizeof(input_cpu));
  cudaMalloc(&result_gpu, sizeof(result_cpu));
  cudaMemcpy(input_gpu, input_cpu, sizeof(input_cpu), cudaMemcpyHostToDevice); 

  torch::jit::script::Module model;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    std::string model_path = dataPath_ + "/linear_dnn.pt";
    model = torch::jit::load(model_path);
    model.to(device);


  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n" << e.what() << std::endl;
  }
  
  // Call function to build tensor and run model
  run<float, 2, 2>(device, model, input_gpu, shape, result_gpu, result_shape);

  // Compare if values are the same as for python script
  cudaMemcpy(result_cpu, result_gpu, sizeof(result_cpu), cudaMemcpyDeviceToHost);
  for (int i = 0; i < result_shape[0]; i++) {
    for (int j = 0; j < result_shape[1]; j++) {
      CPPUNIT_ASSERT(std::abs(result_cpu[i][j] - result_check[i][j]) <= 1.0e-05);
    }
  }

}
