#ifndef PHYSICS_TOOLS__PYTORCH__INTERFACE__SOAMETADATA_H_
#define PHYSICS_TOOLS__PYTORCH__INTERFACE__SOAMETADATA_H_

#include <iostream>

#include <type_traits>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

using namespace cms::soa;

namespace torch_alpaka {
  template <typename T, typename... Others>
    concept SameTypes = (std::same_as<T, Others> && ...);

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
    void push(int i) { columns.push_back(i); }
  };

  // Block of SoA Columns with same type and element size.
  // Calculates size and stride and stores torch type.
  template <typename SOA_Layout>
  struct Block {
    std::vector<long int> stride;
    std::vector<long int> size;

    void* ptr;
    torch::ScalarType type;
    size_t bytes;
    bool is_scalar = false;

    Block() : ptr(nullptr) {}
    // Constructor for columns and eigen columns
    Block(int nElements, void* ptr_, const Columns& columns_, torch::ScalarType type_, size_t bytes_)
        : ptr(ptr_), type(type_), bytes(bytes_) {
      stride = std::move(create_stride(nElements, columns_, bytes_));
      size = std::move(create_size(nElements, columns_));
    };

    // Constructor for scalar columns
    Block(int nElements, void* ptr_, torch::ScalarType type_, size_t bytes_) : ptr(ptr_), type(type_), bytes(bytes_) {
      stride = std::move(create_stride(nElements, 1, bytes_, true));
      size = std::move(create_size(nElements, 1));
    };

    static int get_bytes_per_column(int nElements, size_t bytes) {
      int per_bunch = SOA_Layout::alignment / bytes;
      int bunches = std::ceil(1.0 * nElements / per_bunch);
      std::cout << "skip " << bunches * per_bunch << std::endl;
      return bunches * per_bunch * bytes;
    }

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
      std::cout << "stride " << stride[1] << std::endl;
      return stride;
    }
  };

  // Metadata for SOA split into multiple blocks.
  // An order for the resulting tensors can be defined.
  template <typename SOA_Layout>
  struct SoAMetadata {
  private:
    std::map<std::string, Block<SOA_Layout>> blocks;

    template <typename T>
    inline static torch::ScalarType get_type() {
      return torch::CppTypeToScalarType<T>();
    }

    inline static std::vector<int> standard_order(int size) {
      std::vector<int> order(size);
      for (int i = 0; i < size; i++) {
        order[i] = i;
      }
      return order;
    }

    template <typename T, typename... Others>
    bool check_location(int bytes, T column, T other_column, Others... others) {
      return check_location(bytes, other_column, others...) &&  (column.tupleOrPointer() + bytes) == other_column.tupleOrPointer();
    }

    template <typename T>
    bool check_location(int bytes, T column, T other_column) {
      return check_location(bytes, other_column, others...) &&  (column.tupleOrPointer() + bytes) == other_column.tupleOrPointer();
    }

    template <typename T>
    bool check_location(int bytes, T column) {
      return true;
    }

  public:
    // Order of resulting tensor list
    std::vector<std::string> order;
    int nElements;
    int nBlocks;

    SoAMetadata(int nElements_) : nElements(nElements_), nBlocks(0) {}

    // Constructor for defining blocks with custom order inline
    // Blocks can be masked by setting "-1" as the order position.
    // The name of the block is the position it is called on.
    SoAMetadata(int nElements_,
                std::byte* ptr,
                const std::vector<torch::ScalarType>& types,
                const std::vector<Columns>& columns,
                const std::vector<int>& order_)
        : order(order_.size()), nElements(nElements_), nBlocks(0) {
      int N = std::min({types.size(), columns.size()});
      int skip = 0;

      for (int i = 0; i < N; i++) {
        size_t bytes = torch::elementSize(types[i]);

        if (order_[i] != -1) {
          std::string name(std::to_string(i));
          if (columns[i][0] > 0)
            blocks.try_emplace(name, nElements_, ptr + skip, columns[i], types[i], bytes);
          else
            blocks.try_emplace(name, nElements_, ptr + skip, types[i], bytes);

          order[order_[i]] = name;
          nBlocks += 1;

          skip += columns[i][0] * blocks[name].stride[1] * bytes;
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

    SoAMetadata(int nElements_,
                std::byte* ptr,
                const std::vector<torch::ScalarType>& types,
                const std::vector<Columns>& columns,
                std::vector<int>&& order_)
        : SoAMetadata(nElements_, ptr, types, columns, order_) {}

    SoAMetadata(int nElements_,
                std::byte* ptr,
                const std::vector<torch::ScalarType>& types,
                const std::vector<Columns>& columns)
        : SoAMetadata(nElements_, ptr, types, columns, standard_order(types.size())) {}

    // Constructor for defining single block inline
    SoAMetadata(int nElements_, void* ptr, const torch::ScalarType types, const Columns& columns) : nElements(nElements_), nBlocks(1) {
      size_t bytes = torch::elementSize(types);
      blocks.try_emplace("0", nElements, ptr, columns, types, bytes);
      order.push_back("0");
    }

    // TODO: Check columns are contiguous
    template <typename T, typename... Others>
      requires (SameTypes<typename T::ValueType, typename Others::ValueType...> && T::columnType == SoAColumnType::eigen)
    void append_block(const std::string& name, T column, Others... others) {
      const auto [ptr, stride] = column.tupleOrPointer();

      Columns col({sizeof...(others) + 1, T::ValueType::RowsAtCompileTime});
      if (T::ValueType::ColsAtCompileTime > 1)
        col.push(T::ValueType::ColsAtCompileTime);

      blocks.try_emplace(name, nElements, ptr, col, get_type<typename T::ScalarType>(), sizeof(typename T::ScalarType));
      order.push_back(name);
      nBlocks += 1;
    }

    // TODO: Check columns are contiguous
    // Append a block based on a typed pointer and a column object.
    // Can be normal column or eigen column.
    template <typename T, typename... Others>
      requires (SameTypes<typename T::ScalarType, typename Others::ScalarType...> && T::columnType == SoAColumnType::column)
    void append_block(const std::string& name, T column, Others... others) {
      int bytes = Block<SOA_Layout>::get_bytes_per_column(nElements, sizeof(typename T::ScalarType));
      std::cout << check_location(bytes, column, others...) << std::endl;

      blocks.try_emplace(name, nElements, column.tupleOrPointer(), sizeof...(others) + 1, get_type<typename T::ScalarType>(), sizeof(typename T::ScalarType));
      order.push_back(name);
      nBlocks += 1;
    }

    // // General function to stop the variadic template
    // template <typename T>
    //   requires (T::columnType == SoAColumnType::column)
    // void append_block(const std::string& name, T column) {
    //   blocks.try_emplace(name, nElements, column.tupleOrPointer(), 1, get_type<typename T::ScalarType>(), sizeof(typename T::ScalarType));
    //   order.push_back(name);
    //   nBlocks += 1;
    // }

    // No column value indicates a scalar column, as they can't be stacked.
    template <SoAColumnType col_type, typename T>
      requires (std::is_arithmetic_v<T> && col_type == SoAColumnType::scalar)
    void append_block(const std::string& name, SoAParametersImpl<col_type, T> column) {
      blocks.try_emplace(name, nElements, column.tupleOrPointer(), get_type<T>(), sizeof(T));
      order.push_back(name);
      nBlocks += 1;
    }

    // The order is defined by the order append_block is called.
    // It can be changed by passing a vector of the block names afterwards.
    // All blocks have to be mentioned.
    void change_order(const std::vector<std::string>& new_order) { order = new_order; }
    void change_order(std::vector<std::string>&& new_order) { order = std::move(new_order); }

    inline Block<SOA_Layout> operator[](const std::string& key) const { return blocks.at(key); }
  };
}  // namespace torch_alpaka
#endif  // PHYSICS_TOOLS__PYTORCH__INTERFACE__SOAMETADATA_H_