#include <cstdlib>
#include <iostream>
#include <chrono>

#include <torch/torch.h>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

int main(int argc, char* argv[]) {  
  torch::Device device(torch::kCUDA);
  int row = 3;
  int column = 6;
  int a[row][column] = {{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}, {13, 14, 15, 16, 17, 18}};

  for (int i = 0; i < row; i++) {
    for (int j = 0; j < column; j++) {
      std::cout << a[i][j] << " ";
    }
    std::cout << std::endl;
  } 
  
  auto options = torch::TensorOptions().dtype(torch::kInt).pinned_memory(true);
  std::cout << "Converting vector to Torch tensors on CPU without stride" << std::endl;
  torch::Tensor a_tensor = torch::from_blob(a, {row, column}, options);
  std::cout << a_tensor << std::endl;

  std::cout << "Converting vector to Torch tensors on CPU with stride" << std::endl;
  torch::Tensor b_tensor = torch::from_blob(a, {column, row}, {1, column}, options);
  std::cout << b_tensor << std::endl;

  std::cout << "Transpose Tensor" << std::endl;
  std::cout << torch::transpose(b_tensor, 0, 1) << std::endl;

  std::cout << "Converting vector to Torch tensors on CPU with other stride" << std::endl;
  torch::Tensor c_tensor = torch::from_blob(a, {row, column}, {column, 1}, options);
  std::cout << c_tensor << std::endl;

  std::cout << "Benchmark stride and transpose:" << std::endl;

  row = 500;
  column = 1000;
  int b[row][column];

  for (int i = 0; i < row; i++) {
    for (int j = 0; j < column; j++) {
      b[i][j] = rand();
    }
  } 

  auto t1 = high_resolution_clock::now();
  torch::Tensor d_tensor = torch::from_blob(b, {row, column}, {column, 1}, options);
  auto t2 = high_resolution_clock::now();
  duration<double, std::milli> ms_double = t2 - t1;
  std::cout << "Stride:" << ms_double.count() << "ms\n";

  t1 = high_resolution_clock::now();
  torch::Tensor e_tensor = torch::from_blob(b, {row, column}, options);
  e_tensor = torch::transpose(e_tensor, 0, 1);
  t2 = high_resolution_clock::now();
  ms_double = t2 - t1;
  std::cout << "Transpose:" << ms_double.count() << "ms\n";

  return 0;
}
