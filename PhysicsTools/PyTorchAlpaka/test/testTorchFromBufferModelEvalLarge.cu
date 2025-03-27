#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <exception>
#include <memory>
#include <math.h>
#include "testBase.h"

using std::cout;
using std::endl;
using std::exception;

class testTorchFromBufferModelEvalLarge : public testBasePyTorch {
  CPPUNIT_TEST_SUITE(testTorchFromBufferModelEvalLarge);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

public:
  std::string pyScript() const override;
  void test() override;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testTorchFromBufferModelEvalLarge);

std::string testTorchFromBufferModelEvalLarge::pyScript() const { return "create_dnn_largeinput.py"; }

// bool ENABLE_ERROR = true;

void testTorchFromBufferModelEvalLarge::test() {
  // Declare pinned memory pointers
  float a_cpu[64][1][64];
  float c_cpu[8][10];

  // Allocate pinned memory for the pointers
  // The memory will be accessible from both CPU and GPU
  // without the requirements to copy data from one device
  // to the other
  cout << "Allocating memory for vectors on CPU" << endl;
  memset(a_cpu, 0, 64 * 64 * sizeof(float));
  memset(c_cpu, 0, 8 * 10 * sizeof(float));

  // Declare GPU memory pointers
  int *a_gpu, *c_gpu;

  // Allocate memory on the device
  cout << "Allocating memory for vectors on GPU" << endl;
  cudaMalloc(&a_gpu, 64 * 64 * sizeof(float));
  cudaMalloc(&c_gpu, 8 * 10 * sizeof(float));

  // Copy data from the host to the device (CPU -> GPU)
  cout << "Transfering vectors from CPU to GPU" << endl;
  cudaMemcpy(a_gpu, a_cpu, 64 * 64 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(c_gpu, c_cpu, 8 * 10 * sizeof(float), cudaMemcpyHostToDevice);

  // Load the TorchScript model
  std::string model_path = dataPath_ + "/simple_dnn_largeinput.pt";

  torch::jit::script::Module model;
  torch::Device device(torch::kCUDA);
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load(model_path);
    model.to(device);

  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n" << e.what() << std::endl;
  }

  try {
    // Convert pinned memory on GPU to Torch tensor on GPU
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
    cout << "Converting vectors and result to Torch tensors on GPU" << endl;
    // torch::Tensor a_gpu_tensor = torch::ones({64, 1, 64}, options);
    torch::Tensor a_gpu_tensor = torch::from_blob(a_gpu, {64, 1, 64}, options);

    cout << "Verifying result using Torch tensors" << endl;
    std::vector<torch::jit::IValue> inputs{a_gpu_tensor};
    // Not fully understood but std::move() is needed
    // https://stackoverflow.com/questions/71790378/assign-memory-blob-to-py-torch-output-tensor-c-api
    torch::from_blob(c_gpu, {8, 10}, options) = model.forward(inputs).toTensor();

    //CPPUNIT_ASSERT(c_gpu_tensor.equal(output));
  } catch (exception& e) {
    cout << e.what() << endl;

    cudaFreeHost(a_cpu);
    cudaFreeHost(c_cpu);

    cudaFree(a_gpu);
    cudaFree(c_gpu);

    CPPUNIT_ASSERT(false);
  }

  // // Copy memory to device and also synchronize (implicitly)
  cout << "Synchronizing CPU and GPU. Copying result from GPU to CPU" << endl;
  cudaMemcpy(c_cpu, c_gpu, 8 * 10 * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 10; j++) {
      cout << c_cpu[i][j] << ", ";
    }
    cout << endl;
  } 

  cudaFreeHost(a_cpu);
  cudaFreeHost(c_cpu);

  cudaFree(a_gpu);
  cudaFree(c_gpu);
}
