#ifndef PhysicsTools_PyTorch_interface_JitLoad_h
#define PhysicsTools_PyTorch_interface_JitLoad_h

#include <c10/core/Device.h>
#include <torch/script.h>


namespace cms::torch {

  ::torch::jit::script::Module load(const std::string &model_path, std::optional<::c10::Device> dev = std::nullopt);

}  // namespace cms::torch

#endif  // PhysicsTools_PyTorch_interface_JitLoad_h