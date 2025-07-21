#include <vector>

#include <cppunit/extensions/HelperMacros.h>

#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "PhysicsTools/PyTorch/interface/JitLoad.h"
#include "PhysicsTools/PyTorch/test/testTorchBase.h"

namespace torchtest {

  class TestJitLoadToDeviceDirect : public testTorchBase {
  public:
    std::string script() const override;
    void testDirectDeviceLoad();

    const int64_t batch_size_ = 2 << 10;
    const c10::Device device_ = c10::Device(torch::kCUDA, 0);

  private:
    CPPUNIT_TEST_SUITE(TestJitLoadToDeviceDirect);
    CPPUNIT_TEST(testDirectDeviceLoad);
    CPPUNIT_TEST_SUITE_END();
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestJitLoadToDeviceDirect);

    std::string TestJitLoadToDeviceDirect::script() const { return "testExportLinearDnn.py"; }

  void TestJitLoadToDeviceDirect::testDirectDeviceLoad() {
    // disable test on non-CUDA devices
    if (not cms::cudatest::testDevices())
      return;

    auto model_path = modelPath() + "/linear_dnn.pt";
    auto model = cms::torch::load(model_path, device_);

    auto inputs = std::vector<c10::IValue>();
    inputs.push_back(torch::ones({batch_size_, 3}, device_));
    auto outputs = model.forward(inputs).toTensor();

    auto expected = torch::tensor({2.1f, 1.8f}, torch::TensorOptions().device(device_)).repeat({batch_size_, 1});
    CPPUNIT_ASSERT(torch::allclose(outputs, expected));
  }

}  // namespace torchtest