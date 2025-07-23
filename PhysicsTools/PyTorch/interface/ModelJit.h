#ifndef PhysicsTools_PyTorch_interface_ModelJit_h
#define PhysicsTools_PyTorch_interface_ModelJit_h

#include <string>
#include <torch/torch.h>
#include "PhysicsTools/PyTorch/interface/JitLoad.h"

namespace cms::torch {

  /**
   * @brief wrapper of torch::jit::script::Module
   * @see: https://docs.pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html#class-module
   */
  class ModelJit {
  public:
    explicit ModelJit(std::string &model_path);
    explicit ModelJit(std::string &model_path, ::torch::Device device);

    void to(::torch::Device device, bool non_blocking = false);
    ::torch::IValue forward(std::vector<::torch::IValue> &inputs);
    ::torch::Device device() const;

  protected:
    ::torch::jit::script::Module model_;
    ::torch::Device device_;
  };

}  // namespace cms::torch

#endif  // PhysicsTools_PyTorch_interface_ModelJit_h