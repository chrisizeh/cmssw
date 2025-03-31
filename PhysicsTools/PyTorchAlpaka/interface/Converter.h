#include <torch/torch.h>
#include "DataFormats/SoATemplate/interface/SoALayout.h"

#include "SoAMetadata.h"

namespace torch_alpaka {

  // Metadata to run model with input SOA and fill output SOA.
  template <typename SOA_Input, typename SOA_Output>
  class ModelMetadata {
  public:
    SoAMetadata<SOA_Input> input;
    SoAMetadata<SOA_Output> output;

    ModelMetadata(const SoAMetadata<SOA_Input>& input_, const SoAMetadata<SOA_Output>& output_)
        : input(input_), output(output_) {}
  };

  // Static class to wrap raw SOA pointer in tensor object without copying.
  class Converter {
  public:
    // Calculate size and stride of data store based on InputMetadata and return list of IValue, which is parent class of torch::tensor.
    template <typename SOA_Input, typename SOA_Output>
    static std::vector<torch::IValue> convert_input(const ModelMetadata<SOA_Input, SOA_Output>& metadata,
                                                    torch::Device device) {
      std::vector<torch::IValue> tensors(metadata.input.nBlocks);

      // Initialize size and stride vector with default dimension for scalar block
      torch::Tensor tensor;

      for (int i = 0; i < metadata.input.nBlocks; i++) {
        assert(reinterpret_cast<intptr_t>(metadata.input[metadata.input.order[i]].ptr) % SOA_Input::alignment == 0);
        tensors.at(i) =
            std::move(Converter::array_to_tensor<SOA_Input>(device, metadata.input[metadata.input.order[i]]));
      }
      return tensors;
    }

    // Calculate size and stride of data store based on OutputMetadata and return single output tensor
    template <typename SOA_Input, typename SOA_Output>
    static torch::Tensor convert_output(const ModelMetadata<SOA_Input, SOA_Output>& metadata, torch::Device device) {
      assert(reinterpret_cast<intptr_t>(metadata.output[metadata.output.order[0]].ptr) % SOA_Output::alignment == 0);
      return Converter::array_to_tensor<SOA_Output>(device, metadata.output[metadata.output.order[0]]);
    }

  private:
    // Wrap raw pointer by torch::Tensor based on type, size and stride.
    template <typename SOA_Layout>
    static torch::Tensor array_to_tensor(torch::Device device, Block<SOA_Layout> block) {
      auto options = torch::TensorOptions().dtype(block.type).device(device).pinned_memory(true);
      return torch::from_blob(block.ptr, block.size, block.stride, options);
    }
  };

}  // namespace torch_alpaka
