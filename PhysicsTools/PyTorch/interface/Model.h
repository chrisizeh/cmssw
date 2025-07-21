#ifndef PhysicsTools_PyTorch_interface_Model_h
#define PhysicsTools_PyTorch_interface_Model_h

#include <string>
#include <torch/torch.h>
#include "PhysicsTools/PyTorch/interface/CompilationType.h"
#include "PhysicsTools/PyTorch/interface/JitLoad.h"

namespace cms::torch {

  template <CompilationType>
  class Model;

  /**
   * @brief wrapper of torch::jit::script::Module
   * @see: https://docs.pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html#class-module
   */
  template <>
  class Model<CompilationType::kJit> {
  public:
    explicit Model(std::string &model_path)
        : model_(cms::torch::load(model_path)), 
          device_(::torch::kCPU) {}

    explicit Model(std::string &model_path, ::torch::Device device)
        : model_(cms::torch::load(model_path, device)), 
          device_(device) {}    

    void to(::torch::Device device, bool non_blocking = false) {
      model_.to(device, false);
      device_ = device;
    }

    auto forward(std::vector<::torch::IValue> &inputs) {
      return model_.forward(inputs).toTensor();
    }

    ::torch::Device device() const { return device_; }

  private:
    ::torch::jit::script::Module model_;
    ::torch::Device device_;
  };

}  // namespace cms::torch

#endif  // PhysicsTools_PyTorch_interface_Model_h