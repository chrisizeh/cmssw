#include "PhysicsTools/PyTorch/interface/ModelJit.h"

namespace cms::torch {

  /**
   * Ctor, loads model to Cpu memory
   */
  ModelJit::ModelJit(std::string &model_path)
      : model_(cms::torch::load(model_path)), device_(::torch::kCPU) {}

  /**
   * Ctor, loads model to specified device
   * @note: torchlib interface does not support async loading, 
   *        use `model.to(device, true)` to load asynchronously
   */
  ModelJit::ModelJit(std::string &model_path, ::torch::Device device)
      : model_(cms::torch::load(model_path, device)), device_(device) {}

  /**
   * Move model to specified device memory space
   * @param device Device to move model to
   * @param non_blocking Asynchronous load (in default stream if not set)
   */
  void ModelJit::to(::torch::Device device, bool non_blocking) {
    model_.to(device, non_blocking);
    device_ = device;
  }

  /**
   * Forward pass (inference) of model, returns torch::IValue (multi output support).
   * Match native torchlib interface.
   */
  ::torch::IValue ModelJit::forward(std::vector<::torch::IValue> &inputs) {
    return model_.forward(inputs);
  }

  /**
   * Get model current device information.
   */
  ::torch::Device ModelJit::device() const { return device_; }

}  // namespace cms::torch