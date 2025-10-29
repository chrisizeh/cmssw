#ifndef PhysicsTools_PyTorchAlpakaTest_interface_MaskHost_h
#define PhysicsTools_PyTorchAlpakaTest_interface_MaskHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "PhysicsTools/PyTorchAlpakaTest/interface/MaskSoA.h"

namespace portabletest {

  using MaskHost = PortableHostCollection<Mask>;
  using ScalarMaskHost = PortableHostCollection<ScalarMask>;

}  // namespace portabletest

#endif  // PhysicsTools_PyTorchAlpakaTest_interface_MaskHost_h
