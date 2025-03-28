#ifndef PHYSICS_TOOLS__PYTORCH_ALPAKA__INTERFACE__CONVERTER_H_
#define PHYSICS_TOOLS__PYTORCH_ALPAKA__INTERFACE__CONVERTER_H_

#include <torch/torch.h>
#include <vector>

namespace torch_alpaka {

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
struct MetadataElement {
  torch::ScalarType type;
  Columns columns;
  int bytes;
  bool isScalar;

  MetadataElement(torch::ScalarType type_, const Columns& columns_) : type(type_), columns(columns_) {
    bytes = torch::elementSize(type);

    // Use columns=0 to define scalar, but change to 1 to calculate correct size
    isScalar = (columns[0] == 0);
    if (isScalar)
      columns.columns[0] = 1;
  }

  MetadataElement(torch::ScalarType type_, Columns&& columns_) : type(type_), columns(std::move(columns_)) {
    bytes = torch::elementSize(type);

    isScalar = (columns[0] == 0);
    if (isScalar)
      columns.columns[0] = 1;
  }
};

// Element for support of multiblock SOA, used to create array of blocks for input metadata struct.
struct InputMetadataElement : MetadataElement {
  bool used;

  // Constructor for scalar columns
  InputMetadataElement(torch::ScalarType type_, int columns_)
      : MetadataElement(type_, Columns(columns_)), used(true) {}
  InputMetadataElement(torch::ScalarType type_, int columns_, bool used_)
      : MetadataElement(type_, Columns(columns_)), used(used_) {}

  // Constructor for scalar or eigen columns
  InputMetadataElement(torch::ScalarType type_, const Columns& columns_)
      : MetadataElement(type_, columns_), used(true) {}
  InputMetadataElement(torch::ScalarType type_, const Columns& columns_, bool used_)
      : MetadataElement(type_, columns_), used(used_) {}
};

// Wrapper of generic element for output SOA, with only one block per SOA
struct OutputMetadata : MetadataElement {
  OutputMetadata() : MetadataElement({}, Columns(0)) {}
  OutputMetadata(torch::ScalarType type_, int columns_) : MetadataElement(type_, Columns(columns_)) {}
  OutputMetadata(torch::ScalarType type_, const Columns& columns_) : MetadataElement(type_, columns_) {}
};

// Metadata for input SOA split into multiple blocks.
// An order for the resulting tensors can be defined.
// Blocks can be masked by setting "-1" as the order position.
struct InputMetadata {
private:
  std::vector<InputMetadataElement> blocks;

public:
  // Order of resulting tensor list
  std::vector<int> order;
  int nBlocks;
  int nTensors;

  InputMetadata() : nBlocks(0), nTensors(0) {}

  // Constructor, if all blocks should be converted in initial ordering.
  InputMetadata(const std::vector<torch::ScalarType>& types, const std::vector<Columns>& columns) {
    nBlocks = std::min({types.size(), columns.size()});
    nTensors = 0;

    blocks.reserve(nBlocks);
    order.reserve(nBlocks);

    for (int i = 0; i < nBlocks; i++) {
      nTensors++;
      blocks.emplace_back(types[i], columns[i]);
      order.push_back(i);
    }
  }

  // Constructor, if a special ordering should be created.
  InputMetadata(const std::vector<torch::ScalarType>& types,
                const std::vector<Columns>& columns,
                const std::vector<int>& order_)
      : order(order_) {
    nBlocks = std::min({types.size(), columns.size(), order_.size()});
    nTensors = 0;
    blocks.reserve(nBlocks);

    for (int i = 0; i < nBlocks; i++) {
      nTensors += order[i] != -1;
      blocks.emplace_back(types[i], columns[i], order[i] != -1);
    }
  }

  InputMetadata(const std::vector<torch::ScalarType>& types,
                const std::vector<Columns>& columns,
                std::vector<int>&& order_)
      : order(std::move(order_)) {
    nBlocks = std::min({types.size(), columns.size(), order.size()});
    nTensors = 0;
    blocks.reserve(nBlocks);

    for (int i = 0; i < nBlocks; i++) {
      nTensors += order[i] != -1;
      blocks.emplace_back(types[i], columns[i], order[i] != -1);
    }
  }

  // Constructor if only one Block is present for the SOA
  InputMetadata(const torch::ScalarType types, const int columns) {
    nBlocks = 1;
    nTensors = 1;
    blocks.reserve(1);
    order.reserve(1);

    blocks.emplace_back(types, columns);
    order.push_back(0);
  }

  InputMetadata(const torch::ScalarType types, const Columns& columns) {
    nBlocks = 1;
    nTensors = 1;
    blocks.reserve(1);
    order.reserve(1);

    blocks.emplace_back(types, columns);
    order.push_back(0);
  }

  InputMetadataElement operator[](int i) const { return blocks[i]; }
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
  static std::vector<torch::IValue> convert_input(const ModelMetadata& mask, torch::Device device, std::byte* arr);
  // Calculate size and stride of data store based on OutputMetadata and return single output tensor
  static torch::Tensor convert_output(const ModelMetadata& element, torch::Device device, std::byte* arr);

private:
  static std::vector<long int> soa_get_stride(bool isScalar, int nElements, int bytes, const Columns& columns);
  static std::vector<long int> soa_get_size(int nElements, const Columns& columns);

  // Wrap raw pointer by torch::Tensor based on type, size and stride.
  static torch::Tensor array_to_tensor(torch::Device device,
                                        torch::ScalarType type,
                                        std::byte* arr,
                                        const std::vector<long int>& size,
                                        const std::vector<long int>& stride);
};

// SOA_Layout is needed to calculate minimal size of columns, by using alignment info
template <typename SOA_Layout>
std::vector<long int> Converter<SOA_Layout>::soa_get_stride(bool isScalar,
                                                            int nElements,
                                                            int bytes,
                                                            const Columns& columns) {
  assert(SOA_Layout::alignment % bytes == 0);

  int N = columns.size() + 1;
  std::vector<long int> stride(N);
  int per_bunch = SOA_Layout::alignment / bytes;
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
std::vector<long int> Converter<SOA_Layout>::soa_get_size(int nElements, const Columns& columns) {
  std::vector<long int> size(columns.size() + 1);
  size[0] = nElements;
  std::copy(columns.columns.begin(), columns.columns.end(), size.begin() + 1);

  return size;
}

template <typename SOA_Layout>
torch::Tensor Converter<SOA_Layout>::array_to_tensor(torch::Device device,
                                                      torch::ScalarType type,
                                                      std::byte* arr,
                                                      const std::vector<long int>& size,
                                                      const std::vector<long int>& stride) {
  auto options = torch::TensorOptions()
      .dtype(type)
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
      .device(device)
#endif 
      .pinned_memory(true);

  return torch::from_blob(arr, size, stride, options);
}

template <typename SOA_Layout>
std::vector<torch::IValue> Converter<SOA_Layout>::convert_input(const ModelMetadata& metadata,
                                                                torch::Device device,
                                                                std::byte* arr) {
  assert(reinterpret_cast<intptr_t>(arr) % SOA_Layout::alignment == 0);
  std::vector<torch::IValue> tensors(metadata.input.nTensors);

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
    stride = Converter<SOA_Layout>::soa_get_stride(
        metadata.input[i].isScalar, metadata.nElements, metadata.input[i].bytes, metadata.input[i].columns);

    // Only calculate size and build tensor, if not masked
    if (metadata.input[i].used) {
      size.resize(N);
      size = Converter<SOA_Layout>::soa_get_size(metadata.nElements, metadata.input[i].columns);

      tensors.at(metadata.input.order[i]) =
          std::move(Converter<SOA_Layout>::array_to_tensor(device, metadata.input[i].type, arr + skip, size, stride));
    }

    // Add block size in bytes to skip over it in next round
    skip += metadata.input[i].columns[0] * stride[1] * metadata.input[i].bytes;
  }
  return tensors;
}

template <typename SOA_Layout>
torch::Tensor Converter<SOA_Layout>::convert_output(const ModelMetadata& metadata,
                                                    torch::Device device,
                                                    std::byte* arr) {
  assert(reinterpret_cast<intptr_t>(arr) % SOA_Layout::alignment == 0);
  std::vector<long int> stride = Converter<SOA_Layout>::soa_get_stride(
      metadata.output.isScalar, metadata.nElements, metadata.output.bytes, metadata.output.columns);
  std::vector<long int> size = Converter<SOA_Layout>::soa_get_size(metadata.nElements, metadata.output.columns);

  return Converter<SOA_Layout>::array_to_tensor(device, metadata.output.type, arr, size, stride);
}

}  // namespace torch_alpaka

#endif  // PHYSICS_TOOLS__PYTORCH_ALPAKA__INTERFACE__CONVERTER_H_