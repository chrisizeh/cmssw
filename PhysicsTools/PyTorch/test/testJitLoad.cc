#include "PhysicsTools/PyTorch/interface/JitLoad.h"
#include "PhysicsTools/PyTorch/test/testTorchBase.h"

namespace torchtest {

  class TestJitLoad : public testTorchBase {
  public:
    std::string script() const override;
    void testJitLoadNoException();
    void testJitLoadThrowException();

  private:
    CPPUNIT_TEST_SUITE(TestJitLoad);
    CPPUNIT_TEST(testJitLoadNoException);
    CPPUNIT_TEST(testJitLoadThrowException);
    CPPUNIT_TEST_SUITE_END();
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestJitLoad);

  std::string TestJitLoad::script() const { return "testExportLinearDnn.py"; }

  void TestJitLoad::testJitLoadNoException() {
    auto model_path = modelPath() + "/linear_dnn.pt";
    const auto model = cms::torch::load(model_path);
  }

  void TestJitLoad::testJitLoadThrowException() {
    auto model_path = modelPath() + "/non_existing_model.pt";
    CPPUNIT_ASSERT_THROW(cms::torch::load(model_path), cms::Exception);
  }

}  // namespace torchtest