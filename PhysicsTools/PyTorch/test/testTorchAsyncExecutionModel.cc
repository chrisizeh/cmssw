#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "PhysicsTools/PyTorch/interface/JitLoad.h"
#include "PhysicsTools/PyTorch/test/NvtxScopedRange.h"
#include "PhysicsTools/PyTorch/test/testTorchBase.h"


namespace torchtest {

  using namespace cms::torch;

  class TestTorchAsyncExecutionModel : public testTorchBase {
  public:
    std::string script() const override;
    void test();

  private:
    CPPUNIT_TEST_SUITE(TestTorchAsyncExecutionModel);
    CPPUNIT_TEST(test);
    CPPUNIT_TEST_SUITE_END();

    const int64_t batch_size_ = 2 << 10;
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestTorchAsyncExecutionModel);

  std::string TestTorchAsyncExecutionModel::script() const { return "testExportLinearDnn.py"; }

  void TestTorchAsyncExecutionModel::test() {
    // disable test on non-CUDA devices
    if (!cms::cudatest::testDevices())
      return;

    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) 
      CPPUNIT_FAIL("cudaStreamCreate failed");
    
    auto m_path = modelPath() + "/linear_dnn.pt";
    auto dev = ::torch::Device(::torch::kCUDA, 0);

    // set torch stream from external
    auto default_stream = c10::cuda::getCurrentCUDAStream();
    auto torch_stream = c10::cuda::getStreamFromExternal(stream, dev.index());  
    c10::cuda::setCurrentCUDAStream(torch_stream);

    // async model load and inference check
    NvtxScopedRange range("testAsyncExecutionModel");
    auto m = cms::torch::load(m_path);
    m.to(dev, true);
    auto inputs = std::vector<torch::IValue>();
    inputs.push_back(torch::randn({batch_size_, 3}, dev));
    auto out = m.forward(inputs);
    range.end();

    // restore the default stream
    c10::cuda::setCurrentCUDAStream(default_stream);
    cudaStreamDestroy(stream);
  }

}  // namespace torchtest