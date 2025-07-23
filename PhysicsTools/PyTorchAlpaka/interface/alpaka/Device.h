// ROCm/HIP backend not yet supported, see: https://github.com/pytorch/pytorch/blob/main/aten/CMakeLists.txt#L75
#ifndef PhysicsTools_PyTorchAlpaka_interface_alpaka_Device_h
#define PhysicsTools_PyTorchAlpaka_interface_alpaka_Device_h

#include <torch/torch.h>


namespace ALPAKA_ACCELERATOR_NAMESPACE::torch {

  constexpr auto kDevHost = c10::DeviceType::CPU;

  #ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    constexpr auto kDevice = c10::DeviceType::CUDA;
  // #elif ALPAKA_ACC_GPU_HIP_ENABLED
  //   constexpr auto kDevice = c10::DeviceType::HIP;
  #elif ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    constexpr auto kDevice = c10::DeviceType::CPU;
  #elif ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
    constexpr auto kDevice = c10::DeviceType::CPU;
  #else
  #error "Could not define the torch device type."
  #endif

  namespace alpakatools {

    inline ::torch::Device device(const Device &dev) {
      return ::torch::Device(kDevice, dev.getNativeHandle());
    }

    inline ::torch::Device device(const Queue &queue) {
      return ::torch::Device(kDevice, ::alpaka::getDev(queue).getNativeHandle());
    }

  }  // namespace alpakatools

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torch

#endif  // PhysicsTools_PyTorchAlpaka_interface_alpaka_Device_h