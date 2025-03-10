#include <cstdlib>
#include <cuda/std/array>
#include <cuda/std/ranges>
#include <iostream>
#include <chrono>
#include <array>

#include <cuda_runtime.h>

#include <torch/torch.h>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

template <std::size_t N>
torch::Tensor array_to_tensor(torch::Device device, int* arr, const long int* size) {
  long int arr_size[N];
  long int arr_stride[N];
  std::copy(size, size+N, arr_size);
  std::copy(size, size+N, arr_stride);

  std::reverse(std::begin(arr_size), std::end(arr_size));
  arr_stride[N-1] = 1;
  arr_stride[0] *= arr_stride[1];

  auto options = torch::TensorOptions().dtype(torch::kInt).device(device).pinned_memory(true);
  torch::Tensor tensor = torch::from_blob(arr, arr_size, arr_stride, options);

  return tensor;
}

template <std::size_t N>
void print_column_major(int* arr, const long int* size) {
  if (N == 2) {
    for (int j = 0; j < size[0]; j++) {
      for (int i = 0; i < size[1]; i++) {
          std::cout << arr[i + j*size[1]] << " ";
      }
      std::cout << std::endl;
    } 
  } else if (N == 3) {
    for (int k = 0; k < size[N-1]; k++) {
      std::cout << "(" << k << ")..." << std::endl;
      for (int j = 0; j < size[N-2]; j++) {
          for (int i = 0; i < size[0]; i++) {
            std::cout << arr[i + j*size[0] + k*size[0]*size[1]] << " ";
          }
          std::cout << std::endl;
      }
      std::cout << std::endl;
    }
}
}


int main(int argc, char* argv[]) {  
  torch::Device device(torch::kCUDA);

  int a_cpu[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
  // 2 Dimensional Example
  // long int a_dim_cpu[] = {3, 6};

  // 3 Dimensional Example
  const long int a_dim_cpu[] = {3, 3, 2};

  const size_t dims = sizeof(a_dim_cpu) / sizeof(long int);
  print_column_major<dims>(a_cpu, a_dim_cpu);

  int *a_gpu;
  cudaMalloc(&a_gpu, sizeof(a_cpu));
  cudaMemcpy(a_gpu, a_cpu, sizeof(a_cpu), cudaMemcpyHostToDevice);

  // bad behaviour
  auto options = torch::TensorOptions().dtype(torch::kInt).device(device).pinned_memory(true);
  std::cout << "Converting vector to Torch tensors on CPU without stride" << std::endl;
  torch::Tensor a_tensor = torch::from_blob(a_gpu, a_dim_cpu, options);
  std::cout << a_tensor << std::endl;  

  // Correct Transposition to get to column major. Still wrong.
  std::cout << "Transpose Tensor to get same dimensions" << std::endl;
  a_tensor = torch::transpose(a_tensor, 0, dims-1);
  a_tensor = torch::transpose(a_tensor, 1, dims-1);
  std::cout << a_tensor << std::endl;


  // Use stride to read correctly.
  std::cout << "Converting vector to Torch tensors on CPU with stride" << std::endl;
  std::cout << array_to_tensor<dims>(device, a_gpu, a_dim_cpu) << std::endl;

  long int size_b[] = {500, 1000};
  int b[size_b[0]][size_b[1]];

  for (int i = 0; i < size_b[0]; i++) {
    for (int j = 0; j < size_b[1]; j++) {
      b[i][j] = rand();
    }
  } 

  int *b_gpu;
  cudaMalloc(&b_gpu, size_b[0] * size_b[1] * sizeof(int));
  cudaMemcpy(b_gpu, b, size_b[0] * size_b[1] * sizeof(int), cudaMemcpyHostToDevice);

  std::cout << "Benchmark stride and transpose:" << std::endl;

  auto t1 = high_resolution_clock::now();
  const size_t dim_b = sizeof(size_b) / sizeof(long int);
  array_to_tensor<dim_b>(device, b_gpu, a_dim_cpu);
  auto t2 = high_resolution_clock::now();
  duration<double, std::milli> ms_double = t2 - t1;
  std::cout << "Stride:" << ms_double.count() << "ms\n";

  t1 = high_resolution_clock::now();
  const size_t dim_b2 = sizeof(size_b) / sizeof(long int);
  torch::Tensor e_tensor = torch::from_blob(b_gpu, dim_b2, options);
  a_tensor = torch::transpose(a_tensor, 0, dim_b2-1);
  a_tensor = torch::transpose(a_tensor, 1, dim_b2-1);
  t2 = high_resolution_clock::now();
  ms_double = t2 - t1;
  std::cout << "Transpose:" << ms_double.count() << "ms\n";

  return 0;
}
