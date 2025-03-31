#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace torch_alpaka {

  constexpr auto Byte = torch::kByte;
  constexpr auto Char = torch::kChar;
  constexpr auto Short = torch::kShort;
  constexpr auto Int = torch::kInt;
  constexpr auto Long = torch::kLong;
  constexpr auto UInt16 = torch::kUInt16;
  constexpr auto UInt32 = torch::kUInt32;
  constexpr auto UInt64 = torch::kUInt64;
  constexpr auto Half = torch::kHalf;
  constexpr auto Float = torch::kFloat;
  constexpr auto Double = torch::kDouble;

  // Wrapper struct to merge info about scalar columns and multidimensional eigen columns
  // template <size_t dim>
  struct Columns {
    // CHECK
    // std::array<int, 2> columns;

    std::vector<int> columns;

    // Constructor for scalar columns
    Columns(int columns_) { columns.push_back(columns_); }

    // Constructor for multidimensional eigen columns
    Columns(const std::vector<int>& columns_) : columns(columns_) {}
    Columns(std::vector<int>&& columns_) : columns(std::move(columns_)) {}

    size_t size() const { return columns.size(); }
    int operator[](int i) const { return columns[i]; }
    void push(int i) { columns.push_back(i); }
  };

  template <typename SOA_Layout>
  struct Block {
    std::vector<long int> stride;
    std::vector<long int> size;

    void* ptr;
    torch::ScalarType type;
    size_t bytes;
    bool is_scalar = false;

    Block() : ptr(nullptr) {}
    Block(int nElements, void* ptr_, const Columns& columns_, torch::ScalarType type_, size_t bytes_)
        : ptr(ptr_), type(type_), bytes(bytes_) {
      stride = std::move(create_stride(nElements, columns_, bytes_));
      size = std::move(create_size(nElements, columns_));
    };

    Block(int nElements, void* ptr_, torch::ScalarType type_, size_t bytes_) : ptr(ptr_), type(type_), bytes(bytes_) {
      stride = std::move(create_stride(nElements, 1, bytes_, true));
      size = std::move(create_size(nElements, 1));
    };

  private:
    static std::vector<long int> create_size(int nElements, const Columns& columns) {
      std::vector<long int> size(columns.size() + 1);
      size[0] = nElements;
      std::copy(columns.columns.begin(), columns.columns.end(), size.begin() + 1);

      return size;
    }

    static std::vector<long int> create_stride(int nElements,
                                               const Columns& columns,
                                               size_t bytes,
                                               bool is_scalar = false) {
      assert(SOA_Layout::alignment % bytes == 0);

      int N = columns.size() + 1;
      std::vector<long int> stride(N);

      int per_bunch = SOA_Layout::alignment / bytes;
      int bunches = std::ceil(1.0 * nElements / per_bunch);

      if (!is_scalar)
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
  };

  // Metadata for input SOA split into multiple blocks.
  // An order for the resulting tensors can be defined.
  // Blocks can be masked by setting "-1" as the order position.
  template <typename SOA_Layout>
  struct SoAMetadata {
  private:
    std::map<std::string, Block<SOA_Layout>> blocks;

    template <typename T>
    inline static torch::ScalarType getType() {
      return torch::CppTypeToScalarType<T>();
    }

  public:
    // Order of resulting tensor list
    std::vector<std::string> order;
    int nElements = 0;
    int nBlocks = 0;

    SoAMetadata(int nElements_) : nElements(nElements_) {}

    SoAMetadata(int nElements_,
                std::byte* ptr,
                const std::vector<torch::ScalarType>& types,
                const std::vector<Columns>& columns,
                std::vector<int>&& order_)
        : nElements(nElements_) {
      int N = std::min({types.size(), columns.size()});
      nBlocks = 0;
      int skip = 0;

      order.resize(order_.size());

      for (int i = 0; i < N; i++) {
        size_t bytes = torch::elementSize(types[i]);

        if (order_[i] != -1) {
          if (columns[i][0] > 0)
            blocks.try_emplace(std::to_string(i), nElements_, ptr + skip, columns[i], types[i], bytes);
          else
            blocks.try_emplace(std::to_string(i), nElements_, ptr + skip, types[i], bytes);

          order[order_[i]] = std::to_string(i);
          nBlocks += 1;

          skip += columns[i][0] * blocks[std::to_string(i)].stride[1] * bytes;
        } else {
          Block<SOA_Layout> block;
          if (columns[i][0] > 0)
            block = Block<SOA_Layout>(nElements_, ptr, columns[i], types[i], bytes);
          else
            block = Block<SOA_Layout>(nElements_, ptr, types[i], bytes);

          skip += columns[i][0] * block.stride[1] * bytes;
        }
      }
    }

    SoAMetadata(int nElements, void* ptr, const torch::ScalarType types, const Columns& columns) {
      nBlocks = 1;

      size_t bytes = torch::elementSize(types);
      blocks.try_emplace("0", nElements, ptr, columns, types, bytes);
      order.push_back("0");
    }

    template <typename T, int rows, int cols>
    void append_eigen_block(std::string name,
                            const int columns,
                            Eigen::Map<Eigen::Matrix<T, rows, cols>, 0, Eigen::InnerStride<>> ptr) {
      void* p = &ptr(0, 0);
      Columns col({columns, rows});
      if (cols > 1)
        col.push(cols);

      blocks.try_emplace(name, nElements, p, col, getType<T>(), sizeof(T));
      order.push_back(name);
      nBlocks += 1;
    }

    template <typename T>
    void append_block(std::string name, const Columns& columns, T* ptr) {
      blocks.try_emplace(name, nElements, ptr, columns, getType<T>(), sizeof(T));
      order.push_back(name);
      nBlocks += 1;
    }

    template <typename T>
    void append_block(std::string name, T& ptr) {
      blocks.try_emplace(name, nElements, &ptr, getType<T>(), sizeof(T));
      order.push_back(name);
      nBlocks += 1;
    }

    void change_order(const std::vector<std::string>& new_order) { order = new_order; }

    Block<SOA_Layout> operator[](std::string key) const {
      if (auto search = blocks.find(key); search != blocks.end()) {
        return search->second;
      }
      throw std::invalid_argument("Not a key for SoA blocks");
    }
  };
}  // namespace torch_alpaka