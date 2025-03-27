#ifndef PHYSICS_TOOLS__PYTORCH_ALPAKA__INTERFACE__MODEL_H_
#define PHYSICS_TOOLS__PYTORCH_ALPAKA__INTERFACE__MODEL_H_

#include "PhysicsTools/PyTorchAlpaka/interface/AlpakaConfig.h"
#include "PhysicsTools/PyTorchAlpaka/interface/Converter.h"


namespace torch_alpaka {

class Model {
 public:
  Model() = default;
  Model(const std::string &model_path) 
    : device_(torch::Device(torch::kCPU)), 
      model_(std::move(tools::load(model_path))) {};

  void to(const ALPAKA_ACCELERATOR_NAMESPACE::Device &dev) const {
    device_ = tools::device(dev);
    model_.to(device_);
  };

  void to(const ALPAKA_ACCELERATOR_NAMESPACE::Queue &queue) const {
    device_ = tools::device(queue);
    model_.to(device_);
  };

  void to(const torch::Device &dev) const {
    device_ = dev;
    model_.to(device_);
  };

  const torch::Device &device() const { return device_; };

  template <typename InMemLayout, typename OutMemLayout>
  void forward(
    ModelMetadata &metadata, 
    std::byte *inputs,
    std::byte *outputs
  ) const {
    auto input_tensor = Converter<InMemLayout>::convert_input(metadata, device_, inputs);
    Converter<OutMemLayout>::convert_output(metadata, device_, outputs) = model_.forward(input_tensor).toTensor();
  };

 private:
  mutable torch::Device device_;
  mutable torch::jit::script::Module model_;
};

}  // namespace torch_alpaka

#endif  // PHYSICS_TOOLS__PYTORCH_ALPAKA__INTERFACE__MODEL_H_