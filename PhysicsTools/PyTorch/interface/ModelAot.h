#ifndef PhysicsTools_PyTorch_interface_ModelAot_h
#define PhysicsTools_PyTorch_interface_ModelAot_h

#include <string>
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

namespace cms::torch {

  using AOTPkgLoader = ::torch::inductor::AOTIModelPackageLoader;

  /**
   * @brief wrapper of AOTIModelPackageLoader
   * @note: Following torchlib APIs are subject to change due to active development. 
   *        Authors provide NO BC guarantee for these APIs. Implementation based on 2.6 version. 
   *        Backward compatibility may be required to support multiple PyTorch versions within CMSSW.
   * @see: https://github.com/pytorch/pytorch/blob/v2.6.0/torch/csrc/inductor/aoti_package/model_package_loader.h#L8
   */
  class ModelAot {
  public:
    explicit ModelAot(std::string &precompiled_lib_path);

    std::vector<at::Tensor> forward(std::vector<at::Tensor> &inputs, void *stream_handle = nullptr);
    ::torch::Device device();

  protected:
    AOTPkgLoader pkg_loader_;
    ::torch::Device device_;
  };

}  // namespace cms::torch

#endif  // PhysicsTools_PyTorch_interface_ModelAot_h