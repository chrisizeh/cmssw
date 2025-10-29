#ifndef PhysicsTools_PyTorchAlpakaTest_interface_alpaka_MaskDevice_h
#define PhysicsTools_PyTorchAlpakaTest_interface_alpaka_MaskDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorchAlpakaTest/interface/MaskHost.h"
#include "PhysicsTools/PyTorchAlpakaTest/interface/MaskSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::portabletest {

  using namespace ::portabletest;

  using MaskDevice = PortableCollection<Mask>;
  using ScalarMaskDevice = PortableCollection<ScalarMask>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::portabletest

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(portabletest::MaskDevice, portabletest::MaskHost);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(portabletest::ScalarMaskDevice, portabletest::ScalarMaskHost);

#endif  // PhysicsTools_PyTorchAlpakaTest_interface_alpaka_MaskDevice_h
