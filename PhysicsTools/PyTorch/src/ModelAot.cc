#include "PhysicsTools/PyTorch/interface/ModelAot.h"

namespace cms::torch {

  /**
   * Ctor, loads model to target device memory (information is precompiled in metadata package)
   * Does not support async loading, the H2D copy is done on pageable memory in default torch stream.
   * @see: PhysicsTools/PyTorch/test/testModelWrapperAot.cc -> testAsyncExecutionImplicitStream() / testAsyncExecutionExplicitStream()
   */
  ModelAot::ModelAot(std::string &precompiled_lib_path)
      : pkg_loader_(precompiled_lib_path), device_(::torch::Device(pkg_loader_.get_metadata()["AOTI_DEVICE_KEY"])) {}

  /**
   * Forward pass (inference) of model, returns std::vector<at::Tensor> (multi output support).
   * Match native torchlib interface. cudaStream_t can be passed to run inference on specific stream.
   * If not passed then the one associated with device is grabed from thread local stream registry.
   * @see: https://github.com/pytorch/pytorch/blob/v2.6.0/c10/cuda/CUDAStream.cpp#L169
   * @note: Following torchlib APIs are subject to change due to active development. 
   *        Authors provide NO BC guarantee for these APIs. Implementation based on 2.6 version. 
   *        Backward compatibility may be required to support multiple PyTorch versions within CMSSW
   *        @see: https://github.com/pytorch/pytorch/blob/v2.6.0/torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h#L9
   */
  std::vector<at::Tensor> ModelAot::forward(std::vector<at::Tensor> &inputs, void *stream_handle) {
    return pkg_loader_.run(inputs, stream_handle);
  }

  /**
   * Get model current device information.
   */
  ::torch::Device ModelAot::device() { return device_; }

}  // namespace cms::torch