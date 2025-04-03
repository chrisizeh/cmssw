// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "PhysicsTools/PyTorchTest/plugins/alpaka/Kernels.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

using namespace cms::alpakatools;

class FillParticleCollectionKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void operator()(const TAcc &acc, torchportable::ParticleCollection::View data, float value) const {
    for (auto tid : uniform_elements(acc, data.metadata().size())) {
      data.pt()[tid] = value;
      data.phi()[tid] = value;
      data.eta()[tid] = value;
    }
  }
};

void Kernels::FillParticleCollection(Queue &queue, torchportable::ParticleCollection &data, float value) {
  uint32_t threads_per_block = 512;
  uint32_t blocks_per_grid = divide_up_by(data.view().metadata().size(), threads_per_block);      
  auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
  alpaka::exec<Acc1D>(queue, grid, FillParticleCollectionKernel{}, data.view(), value);
  alpaka::wait(queue);
}

class AssertClassificationKernel {
 public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void operator()(const TAcc &acc, torchportable::ClassificationCollection::View data) const {
    for (auto tid : uniform_elements(acc, data.metadata().size())) {
      ALPAKA_ASSERT_ACC(data.c1()[tid] == 0.5f);
      ALPAKA_ASSERT_ACC(data.c2()[tid] == 0.5f);
    }
  }
};

void Kernels::AssertClassification(Queue &queue, torchportable::ClassificationCollection &data) {
  uint32_t threads_per_block = 512;
  uint32_t blocks_per_grid = divide_up_by(data.view().metadata().size(), threads_per_block);      
  auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
  alpaka::exec<Acc1D>(queue, grid, AssertClassificationKernel{}, data.view());
  alpaka::wait(queue);
}

class AssertRegressionKernel {
 public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void operator()(const TAcc &acc, torchportable::RegressionCollection::View data) const {
    for (auto tid : uniform_elements(acc, data.metadata().size())) {
      ALPAKA_ASSERT_ACC(data.reco_pt()[tid] == 0.5f);
    }
  }
};

void Kernels::AssertRegression(Queue &queue, torchportable::RegressionCollection &data) {
  uint32_t threads_per_block = 512;
  uint32_t blocks_per_grid = divide_up_by(data.view().metadata().size(), threads_per_block);      
  auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
  alpaka::exec<Acc1D>(queue, grid, AssertRegressionKernel{}, data.view());
  alpaka::wait(queue);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE