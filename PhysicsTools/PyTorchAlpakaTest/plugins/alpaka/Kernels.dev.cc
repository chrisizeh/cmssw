// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "PhysicsTools/PyTorchAlpakaTest/plugins/alpaka/Kernels.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

using namespace cms::alpakatools;

class FillParticleCollectionKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void operator()(const TAcc &acc, ParticleCollection::View data, float value) const {
    for (auto tid : uniform_elements(acc, data.metadata().size())) {
      data.pt()[tid] = value;
      data.phi()[tid] = value;
      data.eta()[tid] = value;
    }
  }
};

void Kernels::FillParticleCollection(Queue &queue, ParticleCollection &data, float value) {
  uint32_t threads_per_block = 512;
  uint32_t blocks_per_grid = divide_up_by(data.view().metadata().size(), threads_per_block);      
  auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
  alpaka::exec<Acc1D>(queue, grid, FillParticleCollectionKernel{}, data.view(), value);
  alpaka::wait(queue);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE