#include "PhysicsTools/PyTorch/interface/Model.h"
#include "PhysicsTools/PyTorch/test/NvtxScopedRange.h"
#include "PhysicsTools/PyTorch/test/testTorchBase.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"


namespace torchtest {

  using namespace cms::torch;

  using ModelJit = Model<CompilationType::kJit>;

  class TestModelWrapperJit : public testTorchBase {
  public:
    std::string script() const override;

    void testCtor_DefaultDeviceIsCpu();
    void testCtor_ExplicitDeviceIsHonored();
    void testCtor_BadModelPathThrows();

    void testToDevice_UpdatesUnderlyingState();
    void testToDevice_NonBlocking();

    void testForward_IdempotentOutput();
    void testForward_OutputOnCorrectDevice();

  private:
    CPPUNIT_TEST_SUITE(TestModelWrapperJit);

    CPPUNIT_TEST(testCtor_DefaultDeviceIsCpu);
    CPPUNIT_TEST(testCtor_ExplicitDeviceIsHonored);
    CPPUNIT_TEST(testCtor_BadModelPathThrows);

    CPPUNIT_TEST(testToDevice_UpdatesUnderlyingState);
    CPPUNIT_TEST(testToDevice_NonBlocking);

    CPPUNIT_TEST(testForward_IdempotentOutput);
    CPPUNIT_TEST(testForward_OutputOnCorrectDevice);

    CPPUNIT_TEST_SUITE_END();

    const int64_t batch_size_ = 2 << 10;
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestModelWrapperJit);

  std::string TestModelWrapperJit::script() const { return "testExportLinearDnn.py"; }

  void TestModelWrapperJit::testCtor_DefaultDeviceIsCpu() {
    auto m_path = modelPath() + "/linear_dnn.pt";
    auto m = ModelJit(m_path);

    CPPUNIT_ASSERT_EQUAL(::torch::kCPU, m.device().type());
  }

  void TestModelWrapperJit::testCtor_ExplicitDeviceIsHonored() {
    // disable test on non-CUDA devices
    if (!cms::cudatest::testDevices())
      return;

    auto m_path = modelPath() + "/linear_dnn.pt";
    auto dev = ::torch::Device(::torch::kCUDA, 0);
    auto m = ModelJit(m_path, dev);

    CPPUNIT_ASSERT_EQUAL(dev, m.device());
  }

  void TestModelWrapperJit::testCtor_BadModelPathThrows() {
    auto m_path = modelPath() + "/not_existing_model.pt";
    CPPUNIT_ASSERT_THROW(ModelJit m(m_path), cms::Exception);
  }

  void TestModelWrapperJit::testToDevice_UpdatesUnderlyingState() {
    // disable test on non-CUDA devices
    if (!cms::cudatest::testDevices())
      return;

    auto m_path = modelPath() + "/linear_dnn.pt";
    auto m = ModelJit(m_path);
    auto dev = ::torch::Device(::torch::kCUDA, 0);
    m.to(dev);

    CPPUNIT_ASSERT_EQUAL(dev, m.device());

    m.to(::torch::kCPU);
    CPPUNIT_ASSERT_EQUAL(::torch::kCPU, m.device().type());
  }

  void TestModelWrapperJit::testToDevice_NonBlocking() {
    // disable test on non-CUDA devices
    if (!cms::cudatest::testDevices())
      return;

    auto m_path = modelPath() + "/linear_dnn.pt";
    auto m = ModelJit(m_path);
    auto dev = ::torch::Device(::torch::kCUDA, 0);
    NvtxScopedRange range("testToDevice_NonBlocking");
    m.to(dev, true);
    range.end();

    CPPUNIT_ASSERT_EQUAL(dev, m.device());
  }
  
  void TestModelWrapperJit::testForward_IdempotentOutput() {
    auto m_path = modelPath() + "/linear_dnn.pt";
    auto m = ModelJit(m_path);
    auto inputs = std::vector<torch::IValue>();
    inputs.push_back(torch::randn({batch_size_, 3}));
    auto out1 = m.forward(inputs);
    auto out2 = m.forward(inputs);
    CPPUNIT_ASSERT(out1.equal(out2));
  }

  void TestModelWrapperJit::testForward_OutputOnCorrectDevice() {
    // disable test on non-CUDA devices
    if (!cms::cudatest::testDevices())
     return;

    auto m_path = modelPath() + "/linear_dnn.pt";
    auto dev = ::torch::Device(::torch::kCUDA, 0);
    auto m = ModelJit(m_path, dev);
    auto inputs = std::vector<torch::IValue>();
    inputs.push_back(torch::randn({batch_size_, 3}, dev));
    auto out = m.forward(inputs);
    CPPUNIT_ASSERT_EQUAL(dev, out.device());
  }

}  // namespace torchtest