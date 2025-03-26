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
    
    torch::ScalarType type;
    size_t bytes;
    bool isScalar;

    Block() : ptr(nullptr), columns(0) {}
    // Block(Block& block) : ptr(block.ptr), columns(block.columns), isScalar(block.isScalar) {}

    Block(void* ptr_, const Columns& columns_, torch::ScalarType type_, size_t bytes_) : ptr(ptr_), columns(columns_), type(type_), bytes(bytes_) {
      // Use columns=0 to define scalar, but change to 1 to calculate correct size
      isScalar = (columns[0] == 0);
      if (isScalar)
        columns.columns[0] = 1;
    }
  };

  template <typename T>
  static Block createBlock(const Columns& columns, T* ptr) {
    torch::ScalarType type = torch::CppTypeToScalarType<typename std::remove_reference<decltype(*ptr)>::type>();  
    size_t bytes = sizeof(T);
    return Block(ptr, columns, type, bytes);
  }

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
    InputMetadata(const std::map<std::string, Block>& blocks_) : blocks(blocks_) {
      nBlocks = blocks.size();

      for (const auto& [key, value] : blocks)
        order.push_back(key);
    }

    // Constructor, if a special ordering should be created.
    InputMetadata(const std::map<std::string, Block>& blocks_,
                  const std::vector<std::string>& order_)
        : blocks(blocks_), order(order_) {}

    InputMetadata(const std::map<std::string, Block>& blocks_,
                  std::vector<std::string>&& order_)
        : blocks(blocks_), order(std::move(order_)) {}

    InputMetadata(Block&& block) {
      blocks["default"] = std::move(block);
      order.push_back("default");
    }

    Block operator[](std::string key) const { 
      if (auto search = blocks.find(key); search != blocks.end()) {
        return search->second;
      }
      throw std::invalid_argument( "Not a key for SoA blocks" );
    }
  };

  // Metadata to run model with input SOA and fill output SOA.
  class ModelMetadata {
  public:
    int nElements;

    InputMetadata input;
    Block output;

    ModelMetadata(int nElements_, const InputMetadata& input_, const Block& output_)
        : nElements(nElements_), input(input_), output(output_) {}
  };

  // Static class to wrap raw SOA pointer in tensor object without copying.
  template <typename SOA_Layout>
  class Converter {
  public:
    // Calculate size and stride of data store based on InputMetadata and return list of IValue, which is parent class of torch::tensor.
    static std::vector<torch::IValue> convert_input(const ModelMetadata& mask, torch::Device device);
    // Calculate size and stride of data store based on OutputMetadata and return single output tensor
    static torch::Tensor convert_output(const ModelMetadata& element, torch::Device device);

  private:
    static std::vector<long int> soa_get_stride(int nElements, Block block);
    static std::vector<long int> soa_get_size(int nElements, Block block);

    // Wrap raw pointer by torch::Tensor based on type, size and stride.
    static torch::Tensor array_to_tensor(torch::Device device,
                                         Block block,
                                         const std::vector<long int>& size,
                                         const std::vector<long int>& stride);
  };

  // SOA_Layout is needed to calculate minimal size of columns, by using alignment info
  template <typename SOA_Layout>
  std::vector<long int> Converter<SOA_Layout>::soa_get_stride(int nElements, Block block) {
    assert(SOA_Layout::alignment % block.bytes == 0);

    int N = block.columns.size() + 1;
    std::vector<long int> stride(N);
    int per_bunch = SOA_Layout::alignment / block.bytes;
    int bunches = std::ceil(1.0 * nElements / per_bunch);

    if (!block.isScalar)
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
        stride[i] = stride[i - 1] * block.columns[i - 2];
      }
      stride[1] = stride[N - 1] * block.columns[N - 2];
    }

    return stride;
  }

  template <typename SOA_Layout>
  std::vector<long int> Converter<SOA_Layout>::soa_get_size(int nElements, Block block) {
    std::vector<long int> size(block.columns.size() + 1);
    size[0] = nElements;
    std::copy(block.columns.columns.begin(), block.columns.columns.end(), size.begin() + 1);

    return size;
  }

  template <typename SOA_Layout>
  torch::Tensor Converter<SOA_Layout>::array_to_tensor(torch::Device device,
                                                      Block block,
                                                       const std::vector<long int>& size,
                                                       const std::vector<long int>& stride) {                                                 
    auto options = torch::TensorOptions().dtype(block.type).device(device).pinned_memory(true);
    return torch::from_blob(block.ptr, size, stride, options);
  }

  template <typename SOA_Layout>
  std::vector<torch::IValue> Converter<SOA_Layout>::convert_input(const ModelMetadata& metadata,
                                                                  torch::Device device) {
    
    std::vector<torch::IValue> tensors(metadata.input.nBlocks);

    // Initialize size and stride vector with default dimension for scalar block
    std::vector<long int> stride(2);
    std::vector<long int> size(2);
    torch::Tensor tensor;

    int N;

    for (int i = 0; i < metadata.input.nBlocks; i++) {
      assert(reinterpret_cast<intptr_t>(metadata.input[metadata.input.order[i]].ptr) % SOA_Layout::alignment == 0);
      N = metadata.input[metadata.input.order[i]].columns.size() + 1;

      // Resize if necessary
      // Is used for skip calculation, is therefore calculated also for masked block
      stride.resize(N);
      stride = Converter<SOA_Layout>::soa_get_stride(metadata.nElements, metadata.input[metadata.input.order[i]]);

      // Only calculate size and build tensor, if not masked
      size.resize(N);
      size = Converter<SOA_Layout>::soa_get_size(metadata.nElements, metadata.input[metadata.input.order[i]]);

      tensors.at(i) = std::move(Converter<SOA_Layout>::array_to_tensor(device, metadata.input[metadata.input.order[i]], size, stride));
    }
    return tensors;
  }

  template <typename SOA_Layout>
  torch::Tensor Converter<SOA_Layout>::convert_output(const ModelMetadata& metadata,
                                                      torch::Device device) {
    assert(reinterpret_cast<intptr_t>(metadata.output.ptr) % SOA_Layout::alignment == 0);
    std::vector<long int> stride = Converter<SOA_Layout>::soa_get_stride(metadata.nElements, (Block&)metadata.output);
    std::vector<long int> size = Converter<SOA_Layout>::soa_get_size(metadata.nElements, (Block&)metadata.output);

    return Converter<SOA_Layout>::array_to_tensor(device, metadata.output, size, stride);
  }
}  // namespace torch_alpaka
#endif  // PHYSICS_TOOLS__PYTORCH__INTERFACE__CONVERTER_H_
