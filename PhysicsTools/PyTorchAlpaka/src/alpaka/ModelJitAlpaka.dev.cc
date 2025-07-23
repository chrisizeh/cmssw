#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/ModelJitAlpaka.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torch {

  /**
   * Ctor, loads model to Cpu memory
   */
  ModelJitAlpaka::ModelJitAlpaka(std::string &model_path)
      : cms::torch::ModelJit(model_path) {}

  /**
   * Ctor, loads model to specified device from alpaka::Device
   * @note: torchlib interface does not support async loading, 
   *        use `model.to(device, true)` to load asynchronously
   */
  ModelJitAlpaka::ModelJitAlpaka(std::string &model_path, const Device &dev)
      : cms::torch::ModelJit(model_path, alpakatools::device(dev)) {}

  /**
   * Ctor, loads model to specified device from alpaka::Queue
   * @note: torchlib interface does not support async loading, 
   *        use `model.to(device, true)` to load asynchronously
   */
  ModelJitAlpaka::ModelJitAlpaka(std::string &model_path, const Queue &queue)
      : cms::torch::ModelJit(model_path, alpakatools::device(queue)) {}

  void ModelJitAlpaka::to(const Device &dev, bool non_blocking) {
    ModelJit::to(alpakatools::device(dev), non_blocking);
  }
  
  void ModelJitAlpaka::to(const Queue &queue, bool non_blocking) {
    ModelJit::to(alpakatools::device(queue), non_blocking);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torch