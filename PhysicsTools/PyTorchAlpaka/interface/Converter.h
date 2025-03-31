#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include "DataFormats/SoATemplate/interface/SoALayout.h"

#include "SoAMetadata.h"

namespace torch_alpaka {

  // Metadata to run model with input SOA and fill output SOA.
  template <typename SOA_Input, typename SOA_Output>
  class ModelMetadata {
  public:
    int nElements;

    SoAMetadata<SOA_Input> input;
    SoAMetadata<SOA_Output> output;

    ModelMetadata(int nElements_, const SoAMetadata<SOA_Input>& input_, const SoAMetadata<SOA_Output>& output_)
        : nElements(nElements_), input(input_), output(output_) {}
  };

  // Static class to wrap raw SOA pointer in tensor object without copying.
  class Converter {
  public:
    // Calculate size and stride of data store based on InputMetadata and return list of IValue, which is parent class of torch::tensor.
    template <typename SOA_Input, typename SOA_Output>
    static std::vector<torch::IValue> convert_input(const ModelMetadata<SOA_Input, SOA_Output>& mask, torch::Device device);
    // Calculate size and stride of data store based on OutputMetadata and return single output tensor
    template <typename SOA_Input, typename SOA_Output>
    static torch::Tensor convert_output(const ModelMetadata<SOA_Input, SOA_Output>& element, torch::Device device);

  private:
    static std::vector<long int> soa_get_size(int nElements, Block block);

    // Wrap raw pointer by torch::Tensor based on type, size and stride.
    static torch::Tensor array_to_tensor(torch::Device device,
                                         Block block,
                                         const std::vector<long int>& size,
                                         const std::vector<long int>& stride);
  };

  std::vector<long int> Converter::soa_get_size(int nElements, Block block) {
    std::vector<long int> size(block.columns.size() + 1);
    size[0] = nElements;
    std::copy(block.columns.columns.begin(), block.columns.columns.end(), size.begin() + 1);

    return size;
  }

  torch::Tensor Converter::array_to_tensor(torch::Device device,
                                                      Block block,
                                                       const std::vector<long int>& size,
                                                       const std::vector<long int>& stride) {                                                 
    auto options = torch::TensorOptions().dtype(block.type).device(device).pinned_memory(true);
    return torch::from_blob(block.ptr, size, stride, options);
  }

  template <typename SOA_Input, typename SOA_Output>
  std::vector<torch::IValue> Converter::convert_input(const ModelMetadata<SOA_Input, SOA_Output>& metadata,
                                                                  torch::Device device) {
    
    std::vector<torch::IValue> tensors(metadata.input.nBlocks);

    // Initialize size and stride vector with default dimension for scalar block
    std::vector<long int> stride(2);
    std::vector<long int> size(2);
    torch::Tensor tensor;

    int N;

    for (int i = 0; i < metadata.input.nBlocks; i++) {
      assert(reinterpret_cast<intptr_t>(metadata.input[metadata.input.order[i]].ptr) % SOA_Input::alignment == 0);
      N = metadata.input[metadata.input.order[i]].columns.size() + 1;

      // Resize if necessary
      // Is used for skip calculation, is therefore calculated also for masked block
      stride.resize(N);
      stride = metadata.input[metadata.input.order[i]].template get_stride<SOA_Input>(metadata.nElements);

      // Only calculate size and build tensor, if not masked
      size.resize(N);
      size = Converter::soa_get_size(metadata.nElements, metadata.input[metadata.input.order[i]]);

      tensors.at(i) = std::move(Converter::array_to_tensor(device, metadata.input[metadata.input.order[i]], size, stride));
    }
    return tensors;
  }

  template <typename SOA_Input, typename SOA_Output>
  torch::Tensor Converter::convert_output(const ModelMetadata<SOA_Input, SOA_Output>& metadata,
                                                      torch::Device device) {
    assert(reinterpret_cast<intptr_t>(metadata.output[metadata.output.order[0]].ptr) % SOA_Output::alignment == 0);
    std::vector<long int> stride = metadata.output[metadata.output.order[0]].template get_stride<SOA_Output>(metadata.nElements);
    std::vector<long int> size = Converter::soa_get_size(metadata.nElements, metadata.output[metadata.output.order[0]]);

    return Converter::array_to_tensor(device, metadata.output[metadata.output.order[0]], size, stride);
  }
}  // namespace torch_alpaka
