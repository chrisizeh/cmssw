#include <cppunit/extensions/HelperMacros.h>

#include "PhysicsTools/PyTorch/interface/Model.h"
#include "PhysicsTools/PyTorch/test/NvtxScopedRange.h"
#include "PhysicsTools/PyTorch/test/testUtilities.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"


namespace torchtest {

  using namespace cms::torch;

  using ModelAot = Model<CompilationType::kAot>;

  class TestModelWrapperAot : public CppUnit::TestFixture {
  public:
    void testCpu();
    void testCuda();

  private:
    CPPUNIT_TEST_SUITE(TestModelWrapperAot);

    CPPUNIT_TEST(testCpu);
    CPPUNIT_TEST(testCuda);

    CPPUNIT_TEST_SUITE_END();

    const int64_t batch_size_ = 2 << 10;
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestModelWrapperAot);

  void TestModelWrapperAot::testCpu() {
    auto m_path = cmsswPath("/src/PhysicsTools/PyTorch/models/regression_cpu_el9_amd64_gcc12.pt2");
    auto m = ModelAot(m_path);
    
    std::vector<::torch::IValue> inputs;
    inputs.push_back(torch::ones({batch_size_, 3}, m.device()));

    std::vector<at::Tensor> inputs_tensor;
    for (const auto& val : inputs)
      inputs_tensor.push_back(val.toTensor());

    auto outputs = m.forward(inputs_tensor);
    for (const auto& val : outputs) {
      CPPUNIT_ASSERT(::torch::allclose(val, ::torch::full_like(val, 0.5f)));
    }
  }

  void TestModelWrapperAot::testCuda() {
    // disable test on non-CUDA devices
    if (!cms::cudatest::testDevices())
      return;

    auto m_path = cmsswPath("/src/PhysicsTools/PyTorch/models/regression_cuda_el9_amd64_gcc12.pt2");
    auto m = ModelAot(m_path);
    
    std::vector<::torch::IValue> inputs;
    inputs.push_back(torch::ones({batch_size_, 3}, m.device()));

    std::vector<at::Tensor> inputs_tensor;
    for (const auto& val : inputs)
      inputs_tensor.push_back(val.toTensor());

    auto outputs = m.forward(inputs_tensor);
    for (const auto& val : outputs) {
      CPPUNIT_ASSERT(::torch::allclose(val, ::torch::full_like(val, 0.5f)));
    }
  }

}  // namespace torchtest