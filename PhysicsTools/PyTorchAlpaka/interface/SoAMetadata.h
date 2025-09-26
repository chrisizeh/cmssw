#ifndef PhysicsTools_PyTorchAlpaka_interface_SoAMetadata_h
#define PhysicsTools_PyTorchAlpaka_interface_SoAMetadata_h

#include <cassert>
#include <cmath>
#include <cstddef>
#include <map>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <any>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <ATen/core/ScalarType.h>

#include "PhysicsTools/PyTorch/interface/TorchCompat.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"


namespace cms::torch::alpakatools {

  using namespace cms::soa;
  using namespace cms::alpakatools;

  template <typename T, typename... Others>
  concept SameTypes = (std::same_as<T, Others> && ...);

  // Wrapper struct to merge info about scalar columns and multidimensional eigen columns
  struct Columns {
    std::vector<int> columns;

	// Empty constructor, to fill iteratively
    Columns() {}

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
    ::torch::ScalarType type;
    size_t bytes;
    bool is_scalar = false;

    Block() : ptr(nullptr) {}

    // Constructor for columns
    Block(int nElements, void* ptr_, const Columns& columns_, ::torch::ScalarType type_, size_t bytes_)
        : ptr(ptr_), type(type_), bytes(bytes_) {
        stride = create_stride(nElements, columns_, bytes_);
        size = create_size(nElements, columns_);
    }

    // Constructor for scalar columns
    Block(int nElements, void* ptr_, ::torch::ScalarType type_, size_t bytes_) : ptr(ptr_), type(type_), bytes(bytes_) {
      stride = create_stride(nElements, 1, bytes_, true);
      size = create_size(nElements, 1);
    }

    static int get_elems_per_column(int nElements, size_t bytes) {
      int per_bunch = SOA_Layout::alignment / bytes;
      int bunches = std::ceil(1.0 * nElements / per_bunch);
      return bunches * per_bunch;
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
        bunches = 1 * nElements;
      }
      stride[std::min(2, N - 1)] = bunches * per_bunch;
      // eigen are stored in column major, but still for every column.
      if (N > 2) {
        for (int i = 3; i < N; i++) {
          stride[i] = stride[i - 1] * columns[i - 2];
        }
        stride[1] = stride[N - 1] * columns[N - 2];

		// if(columns[0] == 1)
		//   stride.erase(myvector.begin());
		//   // stride.pop_back();
      }
      return stride;
    }
  };

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
  struct HipMemcpyFallbackBase {
    virtual ~HipMemcpyFallbackBase() = default;
    virtual void copyToHost(ALPAKA_ACCELERATOR_NAMESPACE::Queue&) = 0;
    virtual void copyToDevice(ALPAKA_ACCELERATOR_NAMESPACE::Queue&) = 0;
    virtual void* hostPtr() = 0; 
  };

  template <typename T>
  struct HipMemcpyFallback : HipMemcpyFallbackBase {
    size_t size_;
    size_t stride_;
    void* d_ptr_;
    std::optional<host_buffer<T[]>> h_buf_;

    HipMemcpyFallback(const size_t size, const size_t stride, void* d_ptr)
       : size_(size), stride_(stride), d_ptr_(d_ptr) {
      const size_t n_elems = size * stride;
      h_buf_ = make_host_buffer<T[]>(n_elems);
    }

    void copyToHost(ALPAKA_ACCELERATOR_NAMESPACE::Queue &queue) {
      auto extent = Vec1D{size_ * stride_};
      auto d_view = alpaka::createView(
          alpaka::getDev(queue),
          static_cast<T*>(d_ptr_), 
          extent);

      auto h_view = alpaka::createView(
          cms::alpakatools::host(),
          alpaka::getPtrNative(h_buf_.value()),
          extent);
      alpaka::memcpy(queue, h_view, d_view);
    }

    void copyToDevice(ALPAKA_ACCELERATOR_NAMESPACE::Queue &queue) {
      auto extent = Vec1D{size_ * stride_};
      auto d_view = alpaka::createView(
            alpaka::getDev(queue),
            static_cast<T*>(d_ptr_), 
            extent);

      auto h_view = alpaka::createView(
          cms::alpakatools::host(),
          alpaka::getPtrNative(h_buf_.value()),
          extent);
      alpaka::memcpy(queue, d_view, h_view);
    }

    void* hostPtr() {
      return alpaka::getPtrNative(h_buf_.value());
    }
  };
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

  // Metadata for SOA split into multiple blocks.
  // An order for the resulting tensors can be defined.
  template <typename SOA_Layout>
  struct SoAMetadata {
  private:
    std::map<std::string, Block<SOA_Layout>> blocks;
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    std::vector<std::shared_ptr<HipMemcpyFallbackBase>> buffers_;
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

    template <typename T>
    inline static ::torch::ScalarType get_type() {
      return ::torch::CppTypeToScalarType<T>();
    }

    inline static std::vector<int> standard_order(int size) {
      std::vector<int> order(size);
      for (int i = 0; i < size; i++) {
        order[i] = i;
      }
      return order;
    }

    template <typename T, typename... Others>
    bool check_location(int elements, T* column, T* other_column, Others... others) {
      return check_location(elements, other_column, others...) && (column + elements) == other_column;
    }

    template <typename T>
    bool check_location(int elements, T* column, T* other_column) {
      return (column + elements) == other_column;
    }

    template <typename T>
    bool check_location(int elements, T* column) {
      return true;
    }

  public:
    // Order of resulting tensor list
    std::vector<std::string> order;
    int nElements;
    int nBlocks;

    SoAMetadata(int nElements_) : nElements(nElements_), nBlocks(0) {}

    template <typename T, typename... Others>
      requires(SameTypes<typename T::ValueType, typename Others::ValueType...> && T::columnType == SoAColumnType::eigen)
    void append_block(const std::string& name,
                      std::tuple<T, cms::soa::size_type> column,
                      std::tuple<Others, cms::soa::size_type>... others) {
      auto [ptr, stride] = std::get<0>(column).tupleOrPointer();

      int elems = Block<SOA_Layout>::get_elems_per_column(nElements, sizeof(typename T::ScalarType));
      assert(check_location(elems * T::ValueType::RowsAtCompileTime * T::ValueType::ColsAtCompileTime,
                            ptr,
                            std::get<0>(std::get<0>(others).tupleOrPointer())...));

	  Columns col;
	  if (sizeof...(others) > 0)
		col.push(sizeof...(others) + 1);
	  col.push(T::ValueType::RowsAtCompileTime);
	  
      if (T::ValueType::ColsAtCompileTime > 1) {
        col.push(T::ValueType::ColsAtCompileTime);
	  }

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
      auto hip_memcpy = std::make_shared<HipMemcpyFallback<typename T::ScalarType>>(
          1 + sizeof...(Others), 
          elems * T::ValueType::RowsAtCompileTime * T::ValueType::ColsAtCompileTime, 
          static_cast<void*>(ptr));
      buffers_.push_back(std::move(hip_memcpy));
    
      auto* target_ptr = buffers_.back()->hostPtr();
#else
      auto* target_ptr = ptr;
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

      blocks.try_emplace(name, 
                         nElements, 
                         target_ptr, 
                         col, 
                         get_type<typename T::ScalarType>(), 
                         sizeof(typename T::ScalarType));
      order.push_back(name);
      nBlocks += 1;
    }

    // Append a block based on a typed pointer and a column object.
    template <typename T, typename... Others>
      requires(SameTypes<typename T::ScalarType, typename Others::ScalarType...> &&
               T::columnType == SoAColumnType::column)
    void append_block(const std::string& name,
                      std::tuple<T, cms::soa::size_type> column,
                      std::tuple<Others, cms::soa::size_type>... others) {
      int elems = Block<SOA_Layout>::get_elems_per_column(nElements, sizeof(typename T::ScalarType));
      assert(check_location(elems, std::get<0>(column).tupleOrPointer(), std::get<0>(others).tupleOrPointer()...));

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
      auto hip_memcpy = std::make_shared<HipMemcpyFallback<typename T::ScalarType>>(
          1 + sizeof...(Others), 
          elems, 
          static_cast<void*>(std::get<0>(column).tupleOrPointer()));
      buffers_.push_back(std::move(hip_memcpy));
    
      auto* ptr = buffers_.back()->hostPtr();
#else
      auto *ptr = std::get<0>(column).tupleOrPointer();
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
      blocks.try_emplace(name,
                         nElements,
                         ptr,
                         sizeof...(others) + 1,
                         get_type<typename T::ScalarType>(),
                         sizeof(typename T::ScalarType));
      
      order.push_back(name);
      nBlocks += 1;
    }

    template <SoAColumnType col_type, typename T>
      requires(std::is_arithmetic_v<T> && col_type == SoAColumnType::scalar)
    void append_block(const std::string& name, std::tuple<SoAParametersImpl<col_type, T>, cms::soa::size_type> column) {
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
      auto hip_memcpy = std::make_shared<HipMemcpyFallback<T>>(
          1, 
          1, 
          static_cast<void*>(std::get<0>(column).tupleOrPointer()));
      buffers_.push_back(std::move(hip_memcpy));
    
      auto* ptr = buffers_.back()->hostPtr();
#else
      auto *ptr = std::get<0>(column).tupleOrPointer();
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
      
      blocks.try_emplace(name, 
                         nElements, 
                         ptr, 
                         get_type<T>(), 
                         sizeof(T));
      order.push_back(name);
      nBlocks += 1;
    }

    // The order is defined by the order append_block is called.
    // It can be changed by passing a vector of the block names afterwards.
    // All blocks have to be mentioned.
    void change_order(const std::vector<std::string>& new_order) { order = new_order; }
    void change_order(std::vector<std::string>&& new_order) { order = std::move(new_order); }

    inline Block<SOA_Layout> operator[](const std::string& key) const { return blocks.at(key); }

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    void copyToHost(ALPAKA_ACCELERATOR_NAMESPACE::Queue &queue) {
      for (int i = 0; i < nBlocks; i++) buffers_[i]->copyToHost(queue);
    } 

    void copyToDevice(ALPAKA_ACCELERATOR_NAMESPACE::Queue &queue) {
      for (int i = 0; i < nBlocks; i++) buffers_[i]->copyToDevice(queue);
    }
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
  };

  // Metadata to run model with input SOA and fill output SOA.
  template <typename SOA_Input, typename SOA_Output>
  class ModelMetadata {
  public:
    SoAMetadata<SOA_Input> input;
    SoAMetadata<SOA_Output> output;

    // Used in AOT model class to correctly choose multi or single output conversion
    // Default value true, as single value can be parsed with multi output
    bool multi_output;

    ModelMetadata(const SoAMetadata<SOA_Input>& input_,
                  const SoAMetadata<SOA_Output>& output_,
                  bool multi_output_ = true)
        : input(input_), output(output_), multi_output(multi_output_) {}

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    // For AMD CPU fallback both inputs and outputs are copied to host
    void copyToHost(ALPAKA_ACCELERATOR_NAMESPACE::Queue& queue) {
      input.copyToHost(queue);
      output.copyToHost(queue);
      alpaka::wait(queue);
    }

    // For AMD CPU fallback only outputs are copied to device, no need to copy inputs also
    void copyToDevice(ALPAKA_ACCELERATOR_NAMESPACE::Queue& queue) {
      output.copyToDevice(queue);
      alpaka::wait(queue);
    }
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
  };

}  // namespace cms::torch::alpakatools

#endif  // PhysicsTools_PyTorchAlpaka_interface_SoAMetadata_h
