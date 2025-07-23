#ifndef PhysicsTools_PyTorch_interface_ModelJitAlpaka_h
#define PhysicsTools_PyTorch_interface_ModelJitAlpaka_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/ModelJit.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/Device.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torch {

  /**
   * @brief wrapper of cms::torch::ModelJit with Alpaka support
   * @see: PhysicsTools/PyTorch/interface/ModelJit.h
   */
  class ModelJitAlpaka : public cms::torch::ModelJit {
  public:
    explicit ModelJitAlpaka(std::string &model_path);
    explicit ModelJitAlpaka(std::string &model_path, const Device &dev);
    explicit ModelJitAlpaka(std::string &model_path, const Queue &queue);

    void to(const Device &dev, bool non_blocking = false);
    void to(const Queue &queue, bool non_blocking = false);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torch

#endif  // PhysicsTools_PyTorch_interface_ModelJitAlpaka_h