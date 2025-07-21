#ifndef PhysicsTools_PyTorch_interface_Model_h
#define PhysicsTools_PyTorch_interface_Model_h

#include <string>
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include "PhysicsTools/PyTorch/interface/CompilationType.h"
#include "PhysicsTools/PyTorch/interface/JitLoad.h"

namespace cms::torch {

  template <CompilationType>
  class Model;

  // __________________________________________________________________________________________________________________
  // Just In Time (JIT)

  /**
   * @brief wrapper of torch::jit::script::Module
   * @see: https://docs.pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html#class-module
   */
  template <>
  class Model<CompilationType::kJit> {
  public:
    explicit Model(std::string &model_path);
    explicit Model(std::string &model_path, ::torch::Device device);

    void to(::torch::Device device, bool non_blocking = false);
    ::torch::IValue forward(std::vector<::torch::IValue> &inputs);
    ::torch::Device device() const;

  private:
    ::torch::jit::script::Module model_;
    ::torch::Device device_;
  };

  // __________________________________________________________________________________________________________________
  // Ahead of Time (AOT)

  using AOTPkgLoader = ::torch::inductor::AOTIModelPackageLoader;

  /**
   * @brief wrapper of torch::jit::script::Module
   * @see: https://docs.pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html#class-module
   */
  template <>
  class Model<CompilationType::kAot> {
  public:
    explicit Model(std::string &precompiled_lib_path);

    std::vector<at::Tensor> forward(std::vector<at::Tensor> &inputs, void* stream_handle = nullptr);
    ::torch::Device device();

  private:
    AOTPkgLoader pkg_loader_;
    ::torch::Device device_;
  };

}  // namespace cms::torch

#endif  // PhysicsTools_PyTorch_interface_Model_h