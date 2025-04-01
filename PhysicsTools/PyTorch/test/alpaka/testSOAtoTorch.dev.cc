#include <alpaka/alpaka.hpp>
#include <torch/script.h>
#include <torch/torch.h>

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#endif

#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <sys/prctl.h>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "PhysicsTools/PyTorch/interface/AlpakaConfig.h"
#include "PhysicsTools/PyTorch/interface/Converter.h"
#include "PhysicsTools/PyTorch/test/testBase.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE::torch_alpaka {

using namespace ::torch_alpaka;

// Input SOA
GENERATE_SOA_LAYOUT(SoAPositionTemplate, 
  SOA_COLUMN(float, x), 
  SOA_COLUMN(float, y), 
  SOA_COLUMN(float, z)
)

using SoAPosition = SoAPositionTemplate<>;
using SoAPositionView = SoAPosition::View;
using SoAPositionConstView = SoAPosition::ConstView;

// Output SOA
GENERATE_SOA_LAYOUT(SoAResultTemplate, 
  SOA_COLUMN(float, x), 
  SOA_COLUMN(float, y)
)

using SoAResult = SoAResultTemplate<>;
using SoAResultView = SoAResult::View;

class testSOAToTorch : public testBasePyTorch {
  CPPUNIT_TEST_SUITE(testSOAToTorch);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

 public:
  std::string pyScript() const override;
  void test() override;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testSOAToTorch);

std::string testSOAToTorch::pyScript() const { return "create_linear_dnn.py"; }

// Build Tensor, run model and fill output pointer with result
template <typename SOA_Input, typename SOA_Output>
void run(torch::Device device,
         torch::jit::script::Module model,
         ModelMetadata metadata,
         std::byte* input,
         std::byte* output) {
  std::vector<torch::jit::IValue> input_tensor = Converter<SOA_Input>::convert_input(metadata, device, input);
  Converter<SOA_Output>::convert_output(metadata, device, output) = model.forward(input_tensor).toTensor();
}

class FillKernel {
 public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, PortableCollection<SoAPosition, Device>::View view) const {
    float input[4][3] = {{1, 2, 1}, {2, 4, 3}, {3, 4, 1}, {2, 3, 2}};
    for (int32_t i : cms::alpakatools::uniform_elements(acc, view.metadata().size())) {
      view.x()[i] = input[i][0];
      view.y()[i] = input[i][1];
      view.z()[i] = input[i][2];
    }
  }
};

class TestVerifyKernel {
 public:
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, PortableCollection<SoAResult, Device>::View view) const {
    float result_check[4][2] = {{2.3, -0.5}, {6.6, 3.0}, {2.5, -4.9}, {4.4, 1.3}};
    for (uint32_t i : cms::alpakatools::uniform_elements(acc, view.metadata().size())) {
      ALPAKA_ASSERT_ACC(view.x()[i] - result_check[i][0] < 1.0e-05);
      ALPAKA_ASSERT_ACC(view.x()[i] - result_check[i][0] > - 1.0e-05);
      ALPAKA_ASSERT_ACC(view.y()[i] - result_check[i][1] < 1.0e-05);
      ALPAKA_ASSERT_ACC(view.y()[i] - result_check[i][1] > -1.0e-05);
    }
  }
};

void fill(Queue& queue, PortableCollection<SoAPosition, Device>& collection) {
  uint32_t items = 64;
  uint32_t groups = cms::alpakatools::divide_up_by(collection->metadata().size(), items);
  auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
  alpaka::exec<Acc1D>(queue, workDiv, FillKernel{}, collection.view());
}

void check(Queue& queue, PortableCollection<SoAResult, Device>& collection) {
  uint32_t items = 64;
  uint32_t groups = cms::alpakatools::divide_up_by(collection->metadata().size(), items);
  auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
  alpaka::exec<Acc1D>(queue, workDiv, TestVerifyKernel{}, collection.view());
}

void testSOAToTorch::test() {
  Platform platform;
  std::vector<Device> alpakaDevices = alpaka::getDevs(platform);
  const auto& alpakaHost = alpaka::getDevByIdx(alpaka_common::PlatformHost(), 0u);
  CPPUNIT_ASSERT(alpakaDevices.size());
  const auto& alpakaDevice = alpakaDevices[0];
  Queue queue{alpakaDevice};
  torch::Device torchDevice(kTorchDeviceType);

  // Number of elements
  const std::size_t batch_size = 4;

  // Create and fill needed portable collections
  PortableCollection<SoAPosition, Device> positionCollection(batch_size, alpakaDevice);
  PortableCollection<SoAResult, Device> resultCollection(batch_size, alpakaDevice);
  fill(queue, positionCollection);
  alpaka::wait(queue);

  torch::jit::script::Module model;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    std::string model_path = dataPath_ + "/linear_dnn.pt";
    model = torch::jit::load(model_path);
    model.to(torchDevice);

  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n" << e.what() << std::endl;
  }

  // Call function to build tensor and run model
  InputMetadata inputMask(Float, 3);
  ModelMetadata mask(batch_size, inputMask, OutputMetadata(Float, 2));

  run<SoAPosition, SoAResult>(
      torchDevice, model, mask, positionCollection.buffer().data(), resultCollection.buffer().data());
  check(queue, resultCollection);
}

}