
#include <torch/torch.h>
#include <torch/script.h>

struct Columns {
  std::vector<int> data;

  Columns(int columns_) { data.push_back(columns_); }
  Columns(const std::vector<int>& columns_) : data(columns_) {}
  Columns(std::vector<int>&& columns_) : data(std::move(columns_)) {}

  size_t size() const { return data.size(); }
  int operator[](int i) const { return data[i]; }
};

struct MetadataElement {
  torch::ScalarType type;
  Columns columns;
  int bytes;

  MetadataElement(torch::ScalarType type_, const Columns& columns_) : type(type_), columns(columns_) {
    bytes = torch::elementSize(type);
  }

  MetadataElement(torch::ScalarType type_, Columns&& columns_) : type(type_), columns(std::move(columns_)) {
    bytes = torch::elementSize(type);
  }
};

struct InputMetadataElement : MetadataElement {
  bool used;

  InputMetadataElement(torch::ScalarType type_, int columns_) : MetadataElement(type_, Columns(columns_)), used(true) {}
  InputMetadataElement(torch::ScalarType type_, int columns_, bool used_)
      : MetadataElement(type_, Columns(columns_)), used(used_) {}

  InputMetadataElement(torch::ScalarType type_, const Columns& columns_)
      : MetadataElement(type_, columns_), used(true) {}
  InputMetadataElement(torch::ScalarType type_, const Columns& columns_, bool used_)
      : MetadataElement(type_, columns_), used(used_) {}
};

struct OutputMetadata : MetadataElement {
  OutputMetadata(torch::ScalarType type_, int columns_) : MetadataElement(type_, Columns(columns_)) {}
  OutputMetadata(torch::ScalarType type_, const Columns& columns_) : MetadataElement(type_, columns_) {}
};

struct InputMetadata {
private:
  std::vector<InputMetadataElement> blocks;

public:
  std::vector<int> order;
  int nBlocks;
  int nTensors;

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

class ModelMetadata {
public:
  int nElements;

  InputMetadata input;
  OutputMetadata output;

  ModelMetadata(int nElements_, const InputMetadata& input_, const OutputMetadata& output_)
      : nElements(nElements_), input(input_), output(output_) {}
};

template <typename SOA_Layout>
class Converter {
public:
  static std::vector<torch::IValue> convert_input(const ModelMetadata& mask, torch::Device device, std::byte* arr);
  static torch::Tensor convert_output(const ModelMetadata& element, torch::Device device, std::byte* arr);

private:
  static std::vector<long int> soa_get_stride(int nElements, int bytes, const Columns& columns);
  static std::vector<long int> soa_get_size(int nElements, const Columns& columns);

  static torch::Tensor array_to_tensor(torch::Device device,
                                       torch::ScalarType type,
                                       std::byte* arr,
                                       const std::vector<long int>& size,
                                       const std::vector<long int>& stride);
};

template <typename SOA_Layout>
std::vector<long int> Converter<SOA_Layout>::soa_get_stride(int nElements, int bytes, const Columns& columns) {
  int N = columns.size() + 1;
  std::vector<long int> stride(N);
  int per_bunch = SOA_Layout::alignment / bytes;
  int bunches = std::ceil(1.0 * nElements / per_bunch);

  stride[0] = 1;
  stride[N - 1] = bunches * per_bunch;

  if (columns.size() > 1 && N > 2) {
    for (int i = N - 2; i > 0; i--) {
      stride[i] = stride[i + 1] * columns[i];
    }
  }

  return stride;
}

template <typename SOA_Layout>
std::vector<long int> Converter<SOA_Layout>::soa_get_size(int nElements, const Columns& columns) {
  std::vector<long int> size(columns.size() + 1);
  size[0] = nElements;
  std::copy(columns.data.begin(), columns.data.end(), size.begin() + 1);
  return size;
}

template <typename SOA_Layout>
torch::Tensor Converter<SOA_Layout>::array_to_tensor(torch::Device device,
                                                     torch::ScalarType type,
                                                     std::byte* arr,
                                                     const std::vector<long int>& size,
                                                     const std::vector<long int>& stride) {
  auto options = torch::TensorOptions().dtype(type).device(device).pinned_memory(true);
  return torch::from_blob(arr, size, stride, options);
}

template <typename SOA_Layout>
std::vector<torch::IValue> Converter<SOA_Layout>::convert_input(const ModelMetadata& metadata,
                                                                torch::Device device,
                                                                std::byte* arr) {
  std::vector<torch::IValue> tensors(metadata.input.nTensors);
  std::vector<long int> stride(2);
  std::vector<long int> size(2);
  torch::Tensor tensor;

  int N;
  int skip = 0;

  for (int i = 0; i < metadata.input.nBlocks; i++) {
    std::cout << "Block " << i << std::endl;
    N = metadata.input[i].columns.size() + 1;

    stride.resize(N);
    stride =
        Converter<SOA_Layout>::soa_get_stride(metadata.nElements, metadata.input[i].bytes, metadata.input[i].columns);
    std::cout << "Stride: { ";
    for (int n : stride)
      std::cout << n << ", ";
    std::cout << "} " << std::endl;

    if (metadata.input[i].used) {
      size.resize(N);
      size = Converter<SOA_Layout>::soa_get_size(metadata.nElements, metadata.input[i].columns);
      std::cout << "Size: { ";
      for (int n : size)
        std::cout << n << ", ";
      std::cout << "} " << std::endl;

      tensors.at(metadata.input.order[i]) =
          std::move(Converter<SOA_Layout>::array_to_tensor(device, metadata.input[i].type, arr + skip, size, stride));
    }

    skip += metadata.input[i].columns[0] * stride[N - 1] * metadata.input[i].bytes;
  }
  return tensors;
}

template <typename SOA_Layout>
torch::Tensor Converter<SOA_Layout>::convert_output(const ModelMetadata& metadata,
                                                    torch::Device device,
                                                    std::byte* arr) {
  std::vector<long int> stride =
      Converter<SOA_Layout>::soa_get_stride(metadata.nElements, metadata.output.bytes, metadata.output.columns);

  std::cout << "Stride: { ";
  for (int n : stride)
    std::cout << n << ", ";
  std::cout << "} " << std::endl;

  std::vector<long int> size = Converter<SOA_Layout>::soa_get_size(metadata.nElements, metadata.output.columns);
  std::cout << "Size: { ";
  for (int n : size)
    std::cout << n << ", ";
  std::cout << "} " << std::endl;

  return Converter<SOA_Layout>::array_to_tensor(device, metadata.output.type, arr, size, stride);
}
