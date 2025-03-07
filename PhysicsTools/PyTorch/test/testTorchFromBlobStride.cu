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

int main(int argc, char* argv[]) {  
  torch::Device device(torch::kCUDA);

  int a_cpu[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
  long int a_dim_cpu[] = {3, 6};

  const size_t dims = sizeof(a_dim_cpu) / sizeof(long int);
  long int a_size_cpu[dims];
  long int a_stride_cpu[dims];
  std::copy(std::begin(a_dim_cpu), std::end(a_dim_cpu), a_size_cpu);
  std::copy(std::begin(a_dim_cpu), std::end(a_dim_cpu), a_stride_cpu);

  std::reverse(std::begin(a_size_cpu), std::end(a_size_cpu));
  a_stride_cpu[0] = 1;

  for (int i = 0; i < a_dim_cpu[0]; i++) {
    for (int j = 0; j < a_dim_cpu[1]; j++) {
        std::cout << a_cpu[i*a_dim_cpu[1] + j] << " ";
    }
    std::cout << std::endl;
  } 

  int *a_gpu;
  cudaMalloc(&a_gpu, sizeof(a_cpu));
  cudaMemcpy(a_gpu, a_cpu, sizeof(a_cpu), cudaMemcpyHostToDevice);

  auto options = torch::TensorOptions().dtype(torch::kInt).device(device).pinned_memory(true);
  std::cout << "Converting vector to Torch tensors on CPU without stride" << std::endl;
  torch::Tensor a_tensor = torch::from_blob(a_gpu, a_dim_cpu, options);
  std::cout << a_tensor << std::endl;

  std::cout << "Transpose Tensor" << std::endl;
  std::cout << torch::transpose(a_tensor, 0, 1) << std::endl;

  std::cout << "Converting vector to Torch tensors on CPU with stride" << std::endl;
  torch::Tensor b_tensor = torch::from_blob(a_gpu, a_size_cpu, a_stride_cpu, options);
  std::cout << b_tensor << std::endl;

  long int b_dim[] = {500, 1000};
  int b[b_dim[0]][b_dim[1]];


  for (int i = 0; i < b_dim[0]; i++) {
    for (int j = 0; j < b_dim[1]; j++) {
      b[i][j] = rand();
    }
  } 

  int *b_gpu;
  cudaMalloc(&b_gpu, b_dim[0] * b_dim[1] * sizeof(int));
  cudaMemcpy(b_gpu, b, b_dim[0] * b_dim[1] * sizeof(int), cudaMemcpyHostToDevice);

  std::cout << "Benchmark stride and transpose:" << std::endl;

  auto t1 = high_resolution_clock::now();
  long int b_size[2];
  long int b_stride[2];
  std::copy(std::begin(b_dim), std::end(b_dim), b_size);
  std::copy(std::begin(b_dim), std::end(b_dim), b_stride);

  std::reverse(std::begin(b_size), std::end(b_size));
  b_stride[0] = 1;

  torch::Tensor d_tensor = torch::from_blob(b_gpu, b_size, b_stride, options);
  auto t2 = high_resolution_clock::now();
  duration<double, std::milli> ms_double = t2 - t1;
  std::cout << "Stride:" << ms_double.count() << "ms\n";

  t1 = high_resolution_clock::now();
  torch::Tensor e_tensor = torch::from_blob(b_gpu, b_dim, options);
  e_tensor = torch::transpose(e_tensor, 0, 1);
  t2 = high_resolution_clock::now();
  ms_double = t2 - t1;
  std::cout << "Transpose:" << ms_double.count() << "ms\n";

  return 0;
}
