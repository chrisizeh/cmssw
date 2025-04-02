#include <alpaka/alpaka.hpp>
#include <torch/script.h>
#include <torch/torch.h>

// #include <cmath>
// #include <exception>
// #include <iostream>
// #include <memory>
// #include <sys/prctl.h>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "PhysicsTools/PyTorch/interface/AlpakaConfig.h"
#include "PhysicsTools/PyTorch/interface/Model.h"
#include "PhysicsTools/PyTorch/test/testBase.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace torch_alpaka;

// Input SOA
GENERATE_SOA_LAYOUT(SoAInputsTemplate, 
  SOA_COLUMN(float, x), 
  SOA_COLUMN(float, y), 
  SOA_COLUMN(float, z)
)

using SoAInputs = SoAInputsTemplate<>;
using SoAInputsView = SoAInputs::View;
using SoAInputsConstView = SoAInputs::ConstView;

// Output SOA
GENERATE_SOA_LAYOUT(SoAOutputsTemplate, 
  SOA_COLUMN(float, m), 
  SOA_COLUMN(float, n)
)

using SoAOutputs = SoAOutputsTemplate<>;
using SoAOutputsView = SoAOutputs::View;

class testModelInference : public testBasePyTorch {
  CPPUNIT_TEST_SUITE(testModelInference);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

 public:
  std::string pyScript() const override;
  void test() override;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testModelInference);

std::string testModelInference::pyScript() const { return "create_classifier.py"; }

void testModelInference::test() {
  // alpaka setup
  Platform platform;
  std::vector<Device> alpakaDevices = alpaka::getDevs(platform);
  const auto& alpakaHost = alpaka::getDevByIdx(alpaka_common::PlatformHost(), 0u);
  CPPUNIT_ASSERT(alpakaDevices.size());
  const auto& alpakaDevice = alpakaDevices[0];
  Queue queue{alpakaDevice};

  const std::size_t batch_size = 32;

  // host structs
  PortableHostCollection<SoAInputs> inputs_host(batch_size, cms::alpakatools::host());
  PortableHostCollection<SoAOutputs> outputs_host(batch_size, cms::alpakatools::host());
  // device structs
  PortableCollection<SoAInputs, Device> inputs_device(batch_size, alpakaDevice);
  PortableCollection<SoAOutputs, Device> outputs_device(batch_size, alpakaDevice);
  
  // prepare inputs
  for (size_t i = 0; i < batch_size; i++) { 
    inputs_host.view().x()[i] = 0.0f;
    inputs_host.view().y()[i] = 0.0f;
    inputs_host.view().z()[i] = 0.0f;
  }
  alpaka::memcpy(queue, inputs_device.buffer(), inputs_host.buffer());
  alpaka::wait(queue);

  // instantiate model
  std::string model_path = dataPath_ + "/classifier.pt";
  auto model = Model(model_path);
  model.to(queue);
  CPPUNIT_ASSERT(tools::device(queue) == model.device());

  // metadata for automatic tensor conversion
  InputMetadata inputMask(Float, 3);
  OutputMetadata outputMask(Float, 2); 
  ModelMetadata metadata(batch_size, inputMask, outputMask);
  // inference
  model.forward<SoAInputs, SoAOutputs>(metadata, inputs_device.buffer().data(), outputs_device.buffer().data());

  // check outputs
  alpaka::memcpy(queue, outputs_host.buffer(), outputs_device.buffer());
  alpaka::wait(queue);
  for (size_t i = 0; i < batch_size; i++) {
    CPPUNIT_ASSERT(outputs_host.const_view().m()[i] == 0.5f);
    CPPUNIT_ASSERT(outputs_host.const_view().n()[i] == 0.5f);
  }
}
