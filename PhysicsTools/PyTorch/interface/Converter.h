#ifndef PHYSICS_TOOLS__PYTORCH__INTERFACE__CONVERTER_H_
#define PHYSICS_TOOLS__PYTORCH__INTERFACE__CONVERTER_H_

#include <torch/torch.h>
#include <vector>

namespace torch_alpaka {


  struct Block {
    
    int columns;

    Block(void* ptr_, int columns_) : ptr(ptr_), columns(columns_) {}
  }

  // Wrapper struct to merge info about scalar columns and multidimensional eigen columns
  struct Columns {
    std::vector<int> columns;

    // Constructor for scalar columns
    Columns(int columns_) { columns.push_back(columns_); }

    // Constructor for multidimensional eigen columns
    Columns(const std::vector<int>& columns_) : columns(columns_) {}
    Columns(std::vector<int>&& columns_) : columns(std::move(columns_)) {}

    size_t size() const { return columns.size(); }
    int operator[](int i) const { return columns[i]; }
  };

  // Generic metadata element, which stores necessary information of SOA block.
  struct Block {
    void* ptr;
    Columns columns;
    bool isScalar;

    Block(void* ptr_, const Columns& columns_) : ptr(ptr_), columns(columns_) {
      // Use columns=0 to define scalar, but change to 1 to calculate correct size
      isScalar = (columns[0] == 0);
      if (isScalar)
        columns.columns[0] = 1;
    }

    Block(void* ptr_, Columns&& columns_) : ptr(ptr_), columns(std::move(columns_)) {
      isScalar = (columns[0] == 0);
      if (isScalar)
        columns.columns[0] = 1;
    }
  };

  // Element for support of multiblock SOA, used to create array of blocks for input metadata struct.
  // struct InputBlock : Block {
  //   bool used;

  //   // Constructor for scalar columns
  //   InputBlock(void* ptr_, int columns_)
  //       : Block(ptr_, Columns(columns_)), used(true) {}
  //   InputBlock(void* ptr_, int columns_, bool used_)
  //       : MetadataElement(ptr_, Columns(columns_)), used(used_) {}

  //   // Constructor for scalar or eigen columns
  //   InputBlock(void* ptr_, const Columns& columns_)
  //       : MetadataElement(ptr_, columns_), used(true) {}
  //   InputBlock(void* ptr_, const Columns& columns_, bool used_)
  //       : MetadataElement(ptr_, columns_), used(used_) {}
  // };

  // Wrapper of generic element for output SOA, with only one block per SOA
  struct OutputMetadata : Block {
    OutputMetadata(void* ptr_, int columns_) : Block(ptr_, Columns(columns_)) {}
    OutputMetadata(void* ptr_, const Columns& columns_) : Block(ptr_, columns_) {}
  };

  // Metadata for input SOA split into multiple blocks.
  // An order for the resulting tensors can be defined.
  // Blocks can be masked by setting "-1" as the order position.
  struct InputMetadata {
  private:
    std::map<std::string, Block> blocks;

  public:
    // Order of resulting tensor list
    std::vector<std::string> order;
    int nBlocks;

    // Constructor, if all blocks should be converted in initial ordering.
    InputMetadata(const std::map<std::string, Block> blocks_&) : blocks(blocks_) {
      nBlocks = blocks.size();

      for (int i = 0; i < nBlocks; i++) {
        order.push_back(i);
      }
    }

    // Constructor, if a special ordering should be created.
    InputMetadata(const std::map<std::string, Block> blocks&,
                  const std::vector<int>& order_)
        : blocks(blocks_), order(order_) {}

    InputMetadata(const std::vector<torch::ScalarType>& types,
                  const std::vector<Columns>& columns,
                  std::vector<int>&& order_)
        : blocks(blocks_), order(std::move(order_)) {}

    InputMetadata(const void* ptr, const Columns& columns) {
      blocks["default"] = Block(ptr, columns);
      order.push_back("default");
    }

    InputMetadataElement operator[](std::string key) const { return blocks[key]; }
  };

  // Metadata to run model with input SOA and fill output SOA.
  class ModelMetadata {
  public:
    int nElements;

    InputMetadata input;
    OutputMetadata output;

    ModelMetadata(int nElements_, const InputMetadata& input_, const OutputMetadata& output_)
        : nElements(nElements_), input(input_), output(output_) {}
  };

  // Static class to wrap raw SOA pointer in tensor object without copying.
  template <typename SOA_Layout>
  class Converter {
  public:
    // Calculate size and stride of data store based on InputMetadata and return list of IValue, which is parent class of torch::tensor.
    static std::vector<torch::IValue> convert_input(const ModelMetadata& mask, torch::Device device, void* arr);
    // Calculate size and stride of data store based on OutputMetadata and return single output tensor
    static torch::Tensor convert_output(const ModelMetadata& element, torch::Device device, std::byte* arr);

  private:
    static std::vector<long int> soa_get_stride(int nElements, Block& block);
    static std::vector<long int> soa_get_size(int nElements, Block& block);

    // Wrap raw pointer by torch::Tensor based on type, size and stride.
    static torch::Tensor array_to_tensor(torch::Device device,
                                         void* arr,
                                         const std::vector<long int>& size,
                                         const std::vector<long int>& stride);
  };

  // SOA_Layout is needed to calculate minimal size of columns, by using alignment info
  template <typename SOA_Layout>
  std::vector<long int> Converter<SOA_Layout>::soa_get_stride(int nElements, Block& block) {
    assert(SOA_Layout::alignment % bytes == 0);

    int N = columns.size() + 1;
    std::vector<long int> stride(N);
    int per_bunch = SOA_Layout::alignment / sizeof(*block.ptr());
    int bunches = std::ceil(1.0 * nElements / per_bunch);

    if (!isScalar)
      stride[0] = 1;
    else {
      // Jump no element per row, to fill with scalar value
      stride[0] = 0;
      bunches = 1;
    }
    stride[std::min(2, N - 1)] = bunches * per_bunch;

    // eigen are stored in column major, but still for every column.
    if (N > 2) {
      for (int i = 3; i < N; i++) {
        stride[i] = stride[i - 1] * columns[i - 2];
      }
      stride[1] = stride[N - 1] * columns[N - 2];
    }

    return stride;
  }

  template <typename SOA_Layout>
  std::vector<long int> Converter<SOA_Layout>::soa_get_size(int nElements, Block& block) {
    std::vector<long int> size(block.columns.size() + 1);
    size[0] = nElements;
    std::copy(block.columns.columns.begin(), block.columns.columns.end(), size.begin() + 1);

    return size;
  }

  template <typename SOA_Layout>
  torch::Tensor Converter<SOA_Layout>::array_to_tensor(torch::Device device,
                                                       void* arr,
                                                       const std::vector<long int>& size,
                                                       const std::vector<long int>& stride) {
    torch::Scalar type = torch::CppTypeToScalarType<typename std::remove_reference<decltype(*arr)>::type>::value;                                                    
    auto options = torch::TensorOptions().dtype(type).device(device).pinned_memory(true);
    return torch::from_blob(arr, size, stride, options);
  }

  template <typename SOA_Layout>
  std::vector<torch::IValue> Converter<SOA_Layout>::convert_input(const ModelMetadata& metadata,
                                                                  torch::Device device,
                                                                  void* arr) {
    assert(reinterpret_cast<intptr_t>(arr) % SOA_Layout::alignment == 0);
    std::vector<torch::IValue> tensors(metadata.input.nBlocks);

    // Initialize size and stride vector with default dimension for scalar block
    std::vector<long int> stride(2);
    std::vector<long int> size(2);
    torch::Tensor tensor;

    int N;
    int skip = 0;

    for (int i = 0; i < metadata.input.nBlocks; i++) {
      N = metadata.input[i].columns.size() + 1;

      // Resize if necessary
      // Is used for skip calculation, is therefore calculated also for masked block
      stride.resize(N);
      stride = Converter<SOA_Layout>::soa_get_stride(metadata.nElements, metadata.input[order[i]]);

      // Only calculate size and build tensor, if not masked
      size.resize(N);
      size = Converter<SOA_Layout>::soa_get_size(metadata.nElements, metadata.input[order[i]]);

      tensors.at(i) = std::move(Converter<SOA_Layout>::array_to_tensor(device, metadata.input[order[i]].ptr, size, stride));
    }
    return tensors;
  }

  template <typename SOA_Layout>
  torch::Tensor Converter<SOA_Layout>::convert_output(const ModelMetadata& metadata,
                                                      torch::Device device,
                                                      std::byte* arr) {
    assert(reinterpret_cast<intptr_t>(arr) % SOA_Layout::alignment == 0);
    std::vector<long int> stride = Converter<SOA_Layout>::soa_get_stride(metadata.nElements, metadata.output);
    std::vector<long int> size = Converter<SOA_Layout>::soa_get_size(metadata.nElements, metadata.output);

    return Converter<SOA_Layout>::array_to_tensor(device, metadata.output.ptr, size, stride);
  }
}  // namespace torch_alpaka
#endif  // PHYSICS_TOOLS__PYTORCH__INTERFACE__CONVERTER_H_
