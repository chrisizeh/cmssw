#include "PhysicsTools/PyTorch/interface/Model.h"

namespace cms::torch {

  /**
   * Ctor, loads model to Cpu memory
   */
  Model<CompilationType::kJit>::Model(std::string &model_path)
      : model_(cms::torch::load(model_path)), device_(::torch::kCPU) {}

  /**
   * Ctor, loads model to specified device
   * @note torchlib interface does not support async loading, use model.to(device, true) to load asynchronously
   */
  Model<CompilationType::kJit>::Model(std::string &model_path, ::torch::Device device)
      : model_(cms::torch::load(model_path, device)), device_(device) {}

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
  ::torch::Device Model<CompilationType::kJit>::device() const { return device_; }

  // __________________________________________________________________________________________________________________
  // Ahead of Time (AOT)

  /**
   * Ctor, loads model to target device memory (information is precompiled in metadata package)
   * Does not support async loading, the H2D copy is done on pageable memory in default torch stream.
   * @see: PhysicsTools/PyTorch/test/testModelWrapperAot.cc -> testAsyncExecutionImplicitStream() / testAsyncExecutionExplicitStream()
   */
  Model<CompilationType::kAot>::Model(std::string &precompiled_lib_path)
      : pkg_loader_(precompiled_lib_path), device_(::torch::Device(pkg_loader_.get_metadata()["AOTI_DEVICE_KEY"])) {}

  /**
   * Forward pass (inference) of model, returns std::vector<at::Tensor> (multi output support).
   * Match native torchlib interface. cudaStream_t can be passed to run inference on specific stream.
   * If not passed then the one associated with device is grabed from thread local stream registry.
   * @see: https://github.com/pytorch/pytorch/blob/f41d017aa6ca1bd121cee6e428875b7fd31a7ad0/c10/cuda/CUDAStream.cpp#L169
   * @note: implementation based on 2.6 version, significant changes in 2.7 release
   *        has to be adjusted accordingly when cmssw switches to 2.7
   */
  std::vector<at::Tensor> Model<CompilationType::kAot>::forward(std::vector<at::Tensor> &inputs, void *stream_handle) {
    return pkg_loader_.run(inputs, stream_handle);
  }

  /**
   * Get model current device information.
   */
  ::torch::Device Model<CompilationType::kAot>::device() { return device_; }

}  // namespace cms::torch