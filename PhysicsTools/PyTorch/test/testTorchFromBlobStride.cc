#include <cstdlib>
#include <iostream>

#include <torch/torch.h>


int main(int argc, char* argv[]) {  
  // int a[3] = {1, 2, 3};
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

  std::cout << "Converting vector to Torch tensors on CPU with other stride" << std::endl;
  torch::Tensor c_tensor = torch::from_blob(a, {row, column}, {column, 1}, options);
  std::cout << c_tensor << std::endl;

  return 0;
}
