#include <alpaka/alpaka.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <c10/cuda/CUDAStream.h>
#endif
#include <iostream>
#include <exception>
#include <memory>
#include <math.h>
#include <sys/prctl.h>
#include "../testBase.h"
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

#include "PhysicsTools/PyTorch/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

#include "../Converter.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

// Input SOA
GENERATE_SOA_LAYOUT(SoAPositionTemplate, SOA_COLUMN(float, x), SOA_COLUMN(float, y))

using SoAPosition = SoAPositionTemplate<>;
using SoAPositionView = SoAPosition::View;
using SoAPositionConstView = SoAPosition::ConstView;

// Output SOA
GENERATE_SOA_LAYOUT(SoAResultTemplate, SOA_COLUMN(float, x))

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

std::string testSOAToTorch::pyScript() const { return "create_dnn_sum.py"; }

template <typename SOA_Input, typename SOA_Output>
void run_multi(
    torch::Device device, torch::jit::script::Module model, ModelMetadata mask, std::byte* input, std::byte* output) {
  std::vector<torch::IValue> input_tensors = Converter<SOA_Input>::convert_input(mask, device, input);

  // std::vector<torch::jit::IValue> inputs;
  // inputs.reserve(input_tensors.size());

  // for (size_t i = 0; i < input_tensors.size(); i++) {
  //   inputs.push_back(input_tensors[mask.mapping[i]]);
  // }

  Converter<SOA_Output>::convert_output(mask, device, output) = model.forward(input_tensors).toTensor();
}

void testSOAToTorch::test() {
  std::cout << "ALPAKA Platform info:" << std::endl;
  int idx = 0;
  try {
    for (;;) {
      alpaka::Platform<alpaka::DevCpu> platformHost;
      alpaka::DevCpu host = alpaka::getDevByIdx(platformHost, idx);
      std::cout << "Host[" << idx++ << "]:   " << alpaka::getName(host) << std::endl;
    }
  } catch (...) {
  }
  Platform platform;
  std::vector<Device> alpakaDevices = alpaka::getDevs(platform);
  idx = 0;
  for (const auto& d : alpakaDevices) {
    std::cout << "Device[" << idx++ << "]:   " << alpaka::getName(d) << std::endl;
  }
  const auto& alpakaHost = alpaka::getDevByIdx(alpaka_common::PlatformHost(), 0u);
  CPPUNIT_ASSERT(alpakaDevices.size());
  const auto& alpakaDevice = alpakaDevices[0];
  Queue queue{alpakaDevice};

  std::cout << "Will create torch device with type=" << torch_common::kDeviceType
            << " and native handle=" << alpakaDevice.getNativeHandle() << std::endl;
  torch::Device torchDevice(torch_common::kDeviceType);

  // Number of elements
  const std::size_t batch_size = 6;

  std::vector<float> input{{1, 2, 3, 2, 2, 4, 4, 3, 1, 3, 1, 2}};
  std::array<float, 6> result_check{{5, 5, 4, 5, 3, 6}};

  // Create and fill needed portable collections
  PortableHostCollection<SoAPosition> positionHostCollection(batch_size, cms::alpakatools::host());
  PortableCollection<SoAPosition, Device> positionCollection(batch_size, alpakaDevice);
  SoAPositionView& positionCollectionView = positionHostCollection.view();

  for (size_t i = 0; i < batch_size; i++) {
    positionCollectionView.x()[i] = input[i];
    positionCollectionView.y()[i] = input[i + 1 * batch_size];
  }
  alpaka::memcpy(queue, positionCollection.buffer(), positionHostCollection.buffer());

  PortableHostCollection<SoAResult> resultHostCollection(batch_size, cms::alpakatools::host());
  PortableCollection<SoAResult, Device> resultCollection(batch_size, alpakaDevice);

  torch::jit::script::Module model;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    std::string model_path = dataPath_ + "/simple_dnn_sum.pt";
    model = torch::jit::load(model_path);
    model.to(torchDevice);

  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n" << e.what() << std::endl;
  }

  // Prepare SOA Mask
  InputMetadata inputMask({torch::kFloat, torch::kFloat}, {1, 1}, {1, 0});
  ModelMetadata mask(batch_size, inputMask, OutputMetadata(torch::kFloat, 1));

  // Call function to build tensor and run model
  run_multi<SoAPosition, SoAResult>(
      torchDevice, model, mask, positionCollection.buffer().data(), resultCollection.buffer().data());

  // Compare if values are the same as for python script
  alpaka::memcpy(queue, resultHostCollection.buffer(), resultCollection.buffer());
  alpaka::wait(queue);

  SoAResultView& resultView = resultHostCollection.view();

  std::cout << "Output Matrix:" << std::endl;
  for (size_t i = 0; i < batch_size; i++) {
    std::cout << resultView.x()[i] << std::endl;

    CPPUNIT_ASSERT(std::abs(resultView.x()[i] - result_check[i]) <= 1.0e-05);
  }
}