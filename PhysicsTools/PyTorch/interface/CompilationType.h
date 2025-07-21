#ifndef PhysicsTools_PyTorch_interface_CompilationType_h
#define PhysicsTools_PyTorch_interface_CompilationType_h

namespace cms::torch {

  enum class CompilationType {
    kJit,  // just-in-time compilation, load and compile at runtime from exported model
    kAot   // ahead-of-time compilation, load precompiled shared library at runtime
  };

}  // namespace cms::torch

#endif  // PhysicsTools_PyTorch_interface_CompilationType_h