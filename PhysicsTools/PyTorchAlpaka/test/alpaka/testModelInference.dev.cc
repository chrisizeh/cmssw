#include <alpaka/alpaka.hpp>
#include <torch/script.h>
#include <torch/torch.h>

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#endif

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || \
    defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
#include <nvtx3/nvToolsExt.h>
#endif

#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <sys/prctl.h>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "PhysicsTools/PyTorchAlpaka/interface/AlpakaConfig.h"
#include "PhysicsTools/PyTorchAlpaka/interface/Model.h"
#include "PhysicsTools/PyTorchAlpaka/test/testBase.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace torch_alpaka;

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
  SOA_COLUMN(float, m), 
  SOA_COLUMN(float, n)
)

using SoAResult = SoAResultTemplate<>;
using SoAResultView = SoAResult::View;

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

class FillKernel {
 public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, PortableCollection<SoAPosition, Device>::View view) const {
    for (int32_t i : cms::alpakatools::uniform_elements(acc, view.metadata().size())) {
      view.x()[i] = 0.10 * i;
      view.y()[i] = 0.11 * i;
      view.z()[i] = 0.12 * i;
    }
  }
};

void fill(Queue& queue, PortableCollection<SoAPosition, Device>& collection) {
  uint32_t items = 32;
  uint32_t groups = cms::alpakatools::divide_up_by(collection->metadata().size(), items);
  auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
  alpaka::exec<Acc1D>(queue, workDiv, FillKernel{}, collection.view());
}

class CheckKernel {
 public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, PortableCollection<SoAResult, Device>::View view) const {
    if (cms::alpakatools::once_per_grid(acc)) {
      printf("| m | n |\n");
      for (int i = 0; i < view.metadata().size(); i++) {
        printf("| %1.1f | %1.1f |\n", view.m()[i], view.n()[i]);
      }
    }
  }
};
 
void check(Queue& queue, PortableCollection<SoAResult, Device>& collection) {
  uint32_t items = 32;
  uint32_t groups = cms::alpakatools::divide_up_by(collection->metadata().size(), items);
  auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
  alpaka::exec<Acc1D>(queue, workDiv, CheckKernel{}, collection.view());
}

void testModelInference::test() {
  Platform platform;
  std::vector<Device> alpakaDevices = alpaka::getDevs(platform);
  const auto& alpakaHost = alpaka::getDevByIdx(alpaka_common::PlatformHost(), 0u);
  CPPUNIT_ASSERT(alpakaDevices.size());
  const auto& alpakaDevice = alpakaDevices[0];
  Queue queue{alpakaDevice};

  // Number of elements
  const std::size_t batch_size = 32;

  // Create and fill needed portable collections
  PortableCollection<SoAPosition, Device> positionCollection(batch_size, alpakaDevice);
  PortableCollection<SoAResult, Device> resultCollection(batch_size, alpakaDevice);
  fill(queue, positionCollection);
  alpaka::wait(queue);
  check(queue, resultCollection);

  std::string model_path = dataPath_ + "/classifier.pt";

  auto model = Model(model_path);
  model.to(queue);
  CPPUNIT_ASSERT(tools::device(queue) == model.device());

  InputMetadata inputMask(Float, 3);
  OutputMetadata outputMask(Float, 2); 
  ModelMetadata metadata(batch_size, inputMask, outputMask);

  model.forward<SoAPosition, SoAResult>(metadata, positionCollection.buffer().data(), resultCollection.buffer().data());
  check(queue, resultCollection);
}
