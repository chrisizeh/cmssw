#include "PhysicsTools/PyTorch/interface/JitLoad.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace cms::torch {

  /**
   * @brief Loads a JIT exported TorchScript model.
   *
   * This function wraps `torch::jit::load` to load a TorchScript model from a specified path.
   * In case of failure, it throws a CMS-specific `cms::Exception` with detailed context and error information.
   *
   * @param model_path The file path to the TorchScript model (.pt file).
   * @param dev Optional device to load the model on. Async load is not supported. Use model.to(device, true) instead. 
   *        
   * @return A loaded `torch::jit::script::Module` instance. 
   *
   * @throws cms::Exception If loading fails due to file issues, format errors, or internal TorchScript problems.
   *
   * @note This function is intended for model loading in CMSSW environments, providing
   *       integration with the framework's exception handling and logging facilities.
   */
  ::torch::jit::script::Module load(std::string &model_path, std::optional<::torch::Device> dev) {
    ::torch::jit::script::Module model;
    try {
      model = ::torch::jit::load(model_path, dev);
    } catch (const c10::Error &e) {
      cms::Exception ex("ModelLoadingError");
      ex << "Error loading the model from path: " << model_path << "\n"
         << "Details: " << e.what();
      ex.addContext("Calling cms::torch::load(const std::string&)");
      throw ex;
    }
    return model;
  }

}  // namespace cms::torch