# PhysicsTools/PyTorchAlpaka
The torch interface is split into a [general torch wrapper](../PyTorch) and an Alpaka supported interface. A full ML CMSSW pipeline is implemented and tested in [PyTorchAlpakaTest](../PyTorchAlpakaTest)

This package extends the PyTorch implementation and enables seamless integration with the Alpaka-based heterogeneous computing backend, supporting inference workflows with usage of `pytorch` library with `PortableCollection`s objects. It provides:
- Compatibility with Alpaka device/queue abstractions.
- Single-threading and CUDA stream management are handled by Guard objects specialized for each supported backend.

## Alpaka Config
To enable Alpaka-aware compilation, device type mappings (e.g., to `c10::DeviceType`) are provided in the central configuration file:
```cpp
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/Config.h"
```
This header defines the appropriate kTorchDeviceType constant based on the enabled Alpaka backend (e.g., CUDA, CPU). You can include it in your code to avoid duplicating backend selection logic.
## Interface for Alpaka Modules
PyTorch internally uses optimizations such as a custom thread pool initialized with all available threads by default. To manage thread usage and control execution context, a Guard object is provided. It enforces single-threaded execution on CPU backends and sets the appropriate stream (e.g., CUDA stream) on GPU backends based on the associated Queue event.
**To enable proper execution user has to explicitly scope range of execution with `Guard<Queue>` construct inside Module.**

Examples demonstrating the interoperability of PyTorch with Alpaka in the CMSSW environment can be found in the `plugins` directory. The basic test pipeline includes one input data producer and three heterogeneous modules: a dummy Alpaka module with a basic kernel, a second module performing machine learning inference on the target backend, and a third module explicitly configured to run inference on the CPU backend. This setup serves as a test case for evaluating behavior in a multithreaded and multistream CMSSW environment where inference is executed in parallel across different devices.
## Inference: JIT and AOT Model Execution
Check [PyTorch](../PyTorch) for information.
## SoA Wrapper
The interface provides a converter to dynamically wrap SoA data into one or more `torch::tensors` without the need to copy data. This can be used directly with a PyTorch model. The result can also be dynamically placed into a SoA buffer.
### Metadata
The structural information of the input and output SoA are stored in an `SoAMetadata`. These two objects are then combined to a `ModelMetadata`, to be used by the `Converter`.

The `SoAMetadata` can be defined by first initializing the object and then adding blocks to the metadata. Each block is transformed into a tensor whose size and type are derived from the columns provided.

For two example SoAs Templates, which are stored in PortableCollections, columns can be added to `Blocks`, by using the Metarecords implementation of SoAs.

- **Input SoA:**
```cpp
GENERATE_SOA_LAYOUT(SoATemplate,
    SOA_EIGEN_COLUMN(Eigen::Vector3d, a),
    SOA_EIGEN_COLUMN(Eigen::Vector3d, b),
    SOA_EIGEN_COLUMN(Eigen::Matrix2f, c),
    SOA_COLUMN(double, x),
    SOA_COLUMN(double, y),
    SOA_COLUMN(double, z),
    SOA_SCALAR(float, type),
    SOA_SCALAR(int, someNumber));
```
- **Output SoA:**
```cpp
GENERATE_SOA_LAYOUT(SoAOutputTemplate,
                    SOA_COLUMN(int, cluster));
```
- **Get Metarecords from Portable Collections:**
```cpp
PortableCollection<SoA, Device> deviceCollection(batch_size, queue);
PortableCollection<SoA_Result, Device> deviceResultCollection(batch_size, queue);
fill(queue, deviceCollection);
auto records = deviceCollection.view().records();
auto result_records = deviceResultCollection.view().records();
```
- **For each block** (one tensor), **add the columns** which should be merged to a single tensor. The datatypes have to be the same, and the columns have to be contiguous.
```
SoAMetadata<SoA> input(batch_size);
input.append_block("eigen_vector", records.a(), records.b());
input.append_block("eigen_matrix", records.c());
input.append_block("column", records.x(), records.y(), records.z());
input.append_block("scalar", view.type());
input.change_order({"column", "scalar", "eigen_matrix", "eigen_vector"});

SoAMetadata<SoA> output(batch_size);
output.append_block("result", result_view.cluster());
ModelMetadata metadata(input, output);
```

For Eigen columns, if only a single Vector/Matrix is provided for the tensor, is provided, as if each vector dimension is a column. This means size of tensor is (nElements, dimension) instead of (nElements, 1, dimension).

After adding all the blocks to the `SoAMetadata`, the order of the blocks for inference can be adapted by calling `change_order()`. The order should match the expected input configuration of the PyTorch model.

**Example usage with model:**
```cpp
// add namespaces
using namespace ALPAKA_ACCELERATOR_NAMESPACE::torch;
using namespace cms::torch::alpakatools;

// alpaka setup
Platform platform;
std::vector<Device> alpaka_devices = alpaka::getDevs(platform);
const auto& alpaka_host = alpaka::getDevByIdx(alpaka_common::PlatformHost(), 0u);
CPPUNIT_ASSERT(alpaka_devices.size());
const auto& alpaka_device = alpaka_devices[0];
Queue queue{alpaka_device};

// Prepare Data
const std::size_t batch_size = 32;
PortableCollection<SoAInputs, Device> inputs_device(batch_size, alpaka_device);
PortableCollection<SoAOutputs, Device> outputs_device(batch_size, alpaka_device);
auto input_records = inputs_device.view().records();
auto output_records = outputs_device.view().records();

// guard scope
cms::torch::alpaka::Guard<Queue> guard(queue);  

// instantiate model
auto m_path = get_path("example_model.pt");
auto model = AlpakaModel(m_path);
model.to(queue);

// metadata for automatic tensor conversion
SoAMetadata<SoAInputs> inputs_metadata(batch_size);
inputs_metadata.append_block("features", input_records.x(), input_records.y(), input_records.z());

SoAMetadata<SoAOutputs> outputs_metadata(batch_size); 
outputs_metadata.append_block("preds", output_records.m(), output_records.n());

ModelMetadata<SoAInputs, SoAOutputs> metadata(inputs_metadata, outputs_metadata);

// inference
model.forward(metadata);
```
## Limitations
- Current implementation supports CUDA backend only. ROCm backend is not yet supported, see: https://github.com/pytorch/pytorch/blob/main/aten/CMakeLists.txt#L75ROCm.  Connected with:
    - #9786 Pytorch with ROCm (temp): https://github.com/cms-sw/cmsdist/pull/9312
    - #9312 [WIP] Build PyTorch with ROCm https://github.com/cms-sw/cmsdist/pull/9786
    A workaround is provided, which moves the data to the CPU for the inference.
- AOT support is under active development and subject to changes that obey CMSSW releasing rules.
- Currently only a single block can be added as output and only a single SoA layout can be used for input and output respectively, this will be extended in the next PR.
