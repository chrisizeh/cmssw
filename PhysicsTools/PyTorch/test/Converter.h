
#include <torch/torch.h>
#include <torch/script.h>

struct MetadataElement {
  torch::ScalarType type;
  int bytes;
  int columns;

  MetadataElement(torch::ScalarType type_, int columns_) : type(type_), columns(columns_) {
    bytes = torch::elementSize(type);
  }
};

struct InputMetadataElement : MetadataElement {
  bool used;

  InputMetadataElement(torch::ScalarType type_, int columns_) : MetadataElement(type_, columns_), used(true) {}
  InputMetadataElement(torch::ScalarType type_, int columns_, bool used_)
      : MetadataElement(type_, columns_), used(used_) {}
};

struct OutputMetadata : MetadataElement {
  OutputMetadata(torch::ScalarType type_, int columns_) : MetadataElement(type_, columns_) {}
};

struct InputMetadata {
private:
  std::vector<InputMetadataElement> blocks;

public:
  std::vector<int> order;
  int nBlocks;
  int nTensors;

  InputMetadata(const std::vector<torch::ScalarType>& types, const std::vector<int>& columns) {
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

  // TODO: How to allow option for filter but not order

  // InputMetadata(const std::vector<torch::ScalarType>& types,
  // 	const std::vector<int>& columns,
  // 	const std::vector<bool>& used) {
  // 	nBlocks = std::min({types.size(), columns.size(), used.size()});
  // 	nTensors = 0;

  // 	blocks.reserve(nBlocks);
  // 	order.reserve(nBlocks);

  // 	for (int i = 0; i < nBlocks; i++) {
  // 		nTensors += used;
  // 		blocks.emplace_back(types[i], columns[i], used[i]);
  // 		order.push_back(i);
  // 	}
  // }

  InputMetadata(const std::vector<torch::ScalarType>& types,
                const std::vector<int>& columns,
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

  InputMetadata(const std::vector<torch::ScalarType>& types, const std::vector<int>& columns, std::vector<int>&& order_)
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
  static std::array<long int, 2> soa_get_stride(int nElements, int bytes);
  static std::array<long int, 2> soa_get_size(int nElements, int bytes);

  template <size_t N>
  static torch::Tensor array_to_tensor(torch::Device device,
                                       torch::ScalarType type,
                                       std::byte* arr,
                                       const std::array<long int, 2>& size,
                                       const std::array<long int, 2>& stride);
};

template <typename SOA_Layout>
std::array<long int, 2> Converter<SOA_Layout>::soa_get_stride(int nElements, int bytes) {
  int per_bunch = SOA_Layout::alignment / bytes;
  int bunches = std::ceil(1.0 * nElements / per_bunch);
  std::array<long int, 2> stride{{1, bunches * per_bunch}};
  return stride;
}

template <typename SOA_Layout>
std::array<long int, 2> Converter<SOA_Layout>::soa_get_size(int nElements, int cols) {
  std::array<long int, 2> size{{nElements, cols}};
  return size;
}

template <typename SOA_Layout>
template <size_t N>
torch::Tensor Converter<SOA_Layout>::array_to_tensor(torch::Device device,
                                                     torch::ScalarType type,
                                                     std::byte* arr,
                                                     const std::array<long int, 2>& size,
                                                     const std::array<long int, 2>& stride) {
  long int arr_size[N];
  long int arr_stride[N];
  std::copy(size.begin(), size.end(), arr_size);
  std::copy(stride.begin(), stride.end(), arr_stride);

  auto options = torch::TensorOptions().dtype(type).device(device).pinned_memory(true);
  return torch::from_blob(arr, arr_size, arr_stride, options);
}

template <typename SOA_Layout>
std::vector<torch::IValue> Converter<SOA_Layout>::convert_input(const ModelMetadata& metadata,
                                                                torch::Device device,
                                                                std::byte* arr) {
  std::vector<torch::IValue> tensors(metadata.input.nTensors);
  int skip = 0;
  std::array<long int, 2> stride;
  std::array<long int, 2> size;
  torch::Tensor tensor;

  for (int i = 0; i < metadata.input.nBlocks; i++) {
    std::cout << "Block " << i << std::endl;
    stride = Converter<SOA_Layout>::soa_get_stride(metadata.nElements, metadata.input[i].bytes);
    std::cout << "Stride: {" << stride[0] << ", " << stride[1] << "}" << std::endl;

    if (metadata.input[i].used) {
      size = Converter<SOA_Layout>::soa_get_size(metadata.nElements, metadata.input[i].columns);
      std::cout << "Size: {" << size[0] << ", " << size[1] << "}" << std::endl;

      tensors.at(metadata.input.order[i]) = std::move(
          Converter<SOA_Layout>::array_to_tensor<2>(device, metadata.input[i].type, arr + skip, size, stride));
    }
    skip += metadata.input[i].columns * stride[1] * metadata.input[i].bytes;
  }
  return tensors;
}

template <typename SOA_Layout>
torch::Tensor Converter<SOA_Layout>::convert_output(const ModelMetadata& metadata,
                                                    torch::Device device,
                                                    std::byte* arr) {
  std::array<long int, 2> stride = Converter<SOA_Layout>::soa_get_stride(metadata.nElements, metadata.output.bytes);
  std::cout << "Stride: {" << stride[0] << ", " << stride[1] << "}" << std::endl;
  std::array<long int, 2> size = Converter<SOA_Layout>::soa_get_size(metadata.nElements, metadata.output.columns);
  std::cout << "Size: {" << size[0] << ", " << size[1] << "}" << std::endl;

  return Converter<SOA_Layout>::array_to_tensor<2>(device, metadata.output.type, arr, size, stride);
}
