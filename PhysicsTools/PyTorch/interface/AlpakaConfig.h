#ifndef PHYSICS_TOOLS__PYTORCH__INTERFACE__ALPAKA_CONFIG_H_
#define PHYSICS_TOOLS__PYTORCH__INTERFACE__ALPAKA_CONFIG_H_

#include <alpaka/alpaka.hpp>
#include <torch/script.h>
#include <torch/torch.h>

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#endif

#include <string>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"


namespace torch_alpaka {

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
constexpr c10::DeviceType kTorchDeviceType = c10::DeviceType::CUDA;
#elif ALPAKA_ACC_GPU_HIP_ENABLED
constexpr c10::DeviceType kTorchDeviceType = c10::DeviceType::HIP;
#elif ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
constexpr c10::DeviceType kTorchDeviceType = c10::DeviceType::CPU;
#elif ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
constexpr c10::DeviceType kTorchDeviceType = c10::DeviceType::CPU;
#else
#error "Could not define the torch device type."
#endif

namespace tools {

inline torch::Device device(const ALPAKA_ACCELERATOR_NAMESPACE::Device &dev) {
  return torch::Device(kTorchDeviceType, dev.getNativeHandle());
}

inline torch::Device device(const ALPAKA_ACCELERATOR_NAMESPACE::Queue &queue) {
  return torch::Device(kTorchDeviceType, alpaka::getDev(queue).getNativeHandle());
}

inline torch::jit::script::Module load(const std::string &model_path) {
  try {
    return torch::jit::load(model_path);
  } catch (const c10::Error &e) {
    throw std::runtime_error("Error loading the model: " + std::string(e.what()));
  }
}

inline int64_t queue_hash(const ALPAKA_ACCELERATOR_NAMESPACE::Queue &queue) {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  thread_local auto stream = c10::cuda::getStreamFromExternal(
    queue.getNativeHandle(), device(queue).index());
  return stream.id();
#elif ALPAKA_ACC_GPU_HIP_ENABLED
  return 0;
#endif
  return std::hash<std::thread::id>{}(std::this_thread::get_id());
}
     
}  // namespace torch_alpaka::tools

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

template <typename TQueue>
inline void set_guard(const TQueue &queue);
inline void reset_guard();

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

template <>
inline void set_guard(const alpaka_cuda_async::Queue &queue) {
  // TODO: fix cuBLAS context management
  const auto dev = tools::device(queue);
  thread_local auto stream = c10::cuda::getStreamFromExternal(queue.getNativeHandle(), dev.index());
  c10::cuda::setCurrentCUDAStream(stream);
}

inline void reset_guard() {
  c10::cuda::setCurrentCUDAStream(c10::cuda::getDefaultCUDAStream());
}

#elif ALPAKA_ACC_GPU_HIP_ENABLED

template <>
inline void set_guard(const alpaka_rocm_async::Queue &queue) {}
inline void reset_guard() {}

#elif ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED 

template <>
inline void set_guard(const alpaka_serial_sync::Queue &queue) {
  [[maybe_unused]] static bool initialized = [] {
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    return true;
  }();
}

inline void reset_guard() {}

#elif ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

template <>
inline void set_guard(const alpaka_tbb_async::Queue &queue) {
  [[maybe_unused]] static bool initialized = [] {
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    return true;
  }();
}

inline void reset_guard() {}

#else
#error "Automatic backend detection failed."
#endif

template <typename TBuf>
torch::Tensor toTensor(TBuf &alpakaBuff) {
  static_assert(1 == alpaka::Dim<TBuf>::value, "Current support limited to 1-dimension buffers");
  // Torch types are defined in (pytorch/torch/csrc/api/include/)torch/types.h
  // https://discuss.pytorch.org/t/mapping-a-template-type-to-a-scalartype/53174 to the rescue 😅
  // Apparently we are limited to signed types except for uint8_t.
  // Current master in gitlab supports more types. (see AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS macro in
  // c10/core/ScalarType.h
  auto options =
      torch::TensorOptions()
          .dtype(torch::CppTypeToScalarType<typename std::remove_reference<decltype(*alpakaBuff.data())>::type>::value)
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
          .device(c10::DeviceType::CUDA, alpaka::getDev(alpakaBuff).getNativeHandle())
#endif
          .pinned_memory(true);
  //    std::cout << "data type=" << typeid(*alpakaBuff.data()).name() << std::endl;
  //    std::cout << "data sizeof element=" << (unsigned)sizeof(*alpakaBuff.data()) << std::endl;
  //    std::cout << "buff extent product=" << alpaka::getExtentProduct(alpakaBuff) << std::endl;
  //    std::cout << "getExtends(buff)=[";
  //    for (auto s : alpaka::getExtents(alpakaBuff)) {
  //      std::cout << s << ", ";
  //    }
  //    std::cout << "]" << std::endl;
  return torch::from_blob(alpakaBuff.data(), {alpaka::getExtents(alpakaBuff)[0]}, options);
}

}  // namespace torch_alpaka

#endif  // PHYSICS_TOOLS__PYTORCH__INTERFACE__ALPAKA_CONFIG_H_
