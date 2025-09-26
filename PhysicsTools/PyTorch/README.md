# PhysicsTools/PyTorch
The torch interface is split into a general torch wrapper and an [Alpaka supported interface](../PyTorchAlpaka). A full ML CMSSW pipeline is implemented and tested in [PyTorchAlpakaTest](../PyTorchAlpakaTest).

This package enables seamless integration between PyTorch and the CMSSW SoA implementation. It provides:
- Support for automatic conversion of optimized SoA to torch tensors, with memory blobs reusage.
- Support for both just-in-time (JIT) and ahead-of-time (AOT) model execution (Beta version for AOT).
## Interface for Producer
To not interfere with CMSSW streams and threads, `PyTorchService` **MUST** be included in the `cmsRun`,  whenever PyTorch is used. 
An example setup can be found in [PyTorchAlpakaTest](../PyTorchAlpakaTest).
## Inference: JIT and AOT Model Execution
A Wrapper for the torch model stored with Just-in-Time, `Model` class, is provided enabling inference with torch tensor objects. To use SoAs, the Alpaka Interface has to be used.
### Just-in-Time:
- Loads `torch::jit::script::Module` at runtime.
- Compiles model on-the-fly.
- Introduces warm-up overhead without additional optimization.
- When storing model through tracing, compatibility and correctness have to be checked!

```cpp
auto m_path = get_path("example_model.pt");
Model jit_model(m_path);
jit_model.to(queue);
CPPUNIT_ASSERT(cms::torch::alpaka::device(queue) == jit_model.device());
auto outputs = jit_model.forward(inputs);
```

### Ahead-of-Time (beta version):
- Uses PyTorch AOT compiler to generate `.cpp` and `.so` files. (prerequisite done manually by end-user)
- Package provide helper script to automate this process to some extent `PhysicsTools/PyTorch/scripts/pytorch_aot_auto_compilation.sh` (run from within `PhysicsTools/PyTorch` directory)
- Loads compiled model via `AOTIModelContainerRunner`.
- Eliminates JIT overhead, enable optimization, but requires architecture-specific handling.

```cpp
auto lib_path = shared_lib();
auto m_path = get_path("example_precompiled_model.pt2");
ModelAOT aot_model(m_path);
auto outputs = aot_model.forward(inputs_tensor);
```
