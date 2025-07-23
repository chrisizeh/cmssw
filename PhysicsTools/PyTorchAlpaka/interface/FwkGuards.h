#ifndef PhysicsTools_PyTorchAlpaka_interface_FwkGuards_h
#define PhysicsTools_PyTorchAlpaka_interface_FwkGuards_h

#include <torch/torch.h>
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/Device.h"
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <c10/cuda/CUDAStream.h>
#endif

namespace cms::torch::alpaka {

  template <typename TQueue>
  struct FwkGuardTraits;

  template <typename TQueue>
  class FwkGuard {
  public:
    explicit FwkGuard(const TQueue &queue) : queue_(queue) {
      FwkGuardTraits<TQueue>::set(queue_);
    }

    ~FwkGuard() {
      FwkGuardTraits<TQueue>::reset();
    }

  private:
    const TQueue &queue_;
  };
\

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

  template <>
  struct FwkGuardTraits<alpaka_cuda_async::Queue> {
    static void set(const alpaka_cuda_async::Queue &queue) {
      auto dev = ALPAKA_ACCELERATOR_NAMESPACE::torch::alpakatools::device(queue);
      auto stream = c10::cuda::getStreamFromExternal(queue.getNativeHandle(), dev.index());
      c10::cuda::setCurrentCUDAStream(stream);
    }

    static void reset() { /**< optional: reset to previous state/stream. */ }
  };

#elif ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

  template <>
  struct FwkGuardTraits<alpaka_serial_sync::Queue> {
    static void set(const alpaka_serial_sync::Queue &) { /**< nothing to be done */ }
    static void reset() { /**< nothing to be done */ }
  };

#elif ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

  template <>
  struct FwkGuardTraits<alpaka_tbb_async::Queue> {
    static void set(const alpaka_tbb_async::Queue &) { /**< nothing to be done */ }
    static void reset() { /**< nothing to be done */ }
  };

#else
#error "TorchAlpaka guard for this backend is not defined."
#endif

}  // namespace cms::torch


namespace ALPAKA_ACCELERATOR_NAMESPACE::torch {

  using namespace cms::torch::alpaka;
  using Guard = FwkGuard<Queue>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torch

#endif  // PhysicsTools_PyTorchAlpaka_interface_FwkGuards_h