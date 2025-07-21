#include "PhysicsTools/PyTorch/interface/Model.h"

namespace cms::torch {

  /**
   * Ctor, loads model to Cpu memory
   */
  Model<CompilationType::kJit>::Model(std::string &model_path)
    : model_(cms::torch::load(model_path)),
      device_(::torch::kCPU) {}

  /**
   * Ctor, loads model to specified device
   * @note torchlib interface does not support async loading, use model.to(device, true) to load asynchronously
   */
  Model<CompilationType::kJit>::Model(std::string &model_path, ::torch::Device device)
      : model_(cms::torch::load(model_path, device)),
        device_(device) {}

  /**
   * Move model to specified device memory space
   * @param device Device to move model to
   * @param non_blocking Asynchronous load
   */ 
  void Model<CompilationType::kJit>::to(::torch::Device device, bool non_blocking) {
    model_.to(device, non_blocking);
    device_ = device;
  }

  /**
   * Forward pass (inference) of model, returns torch::IValue (multi output support).
   * Match native torchlib interface.
   */
  ::torch::IValue Model<CompilationType::kJit>::forward(std::vector<::torch::IValue> &inputs) {
    return model_.forward(inputs);
  }

  /**
   * Get model current device information.
   */
  ::torch::Device Model<CompilationType::kJit>::device() const {
    return device_;
  }

  // __________________________________________________________________________________________________________________
  // Ahead of Time (AOT)

  /**
   * Ctor, loads model to target device memory (information is precompiled in metadata package)
   */
  Model<CompilationType::kAot>::Model(std::string &precompiled_lib_path)
    : pkg_loader_(precompiled_lib_path),
      device_(::torch::Device(pkg_loader_.get_metadata()["AOTI_DEVICE_KEY"])) {}

  /**
   * Forward pass (inference) of model, returns std::vector<at::Tensor> (multi output support).
   * Match native torchlib interface.
   * @note: implementation based on 2.6 version, significant changes in 2.7 release
   *        has to be adjusted accordingly when cmssw switches to 2.7
   */
  std::vector<at::Tensor> Model<CompilationType::kAot>::forward(std::vector<at::Tensor> &inputs, void* stream_handle) {
    return pkg_loader_.run(inputs, stream_handle);
  }

  /**
   * Get model current device information.
   */
  ::torch::Device Model<CompilationType::kAot>::device() {
    for (const auto& [key, value] : pkg_loader_.get_metadata()) {
      std::cout << key << " => " << value << std::endl;
    }
    return device_;
  }

}  // namespace cms::torch