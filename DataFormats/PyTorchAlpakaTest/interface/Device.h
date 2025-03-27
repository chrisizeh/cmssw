#ifndef DATA_FORMATS__PYTORCH_ALPAKA_TEST__INTERFACE__DEVICE_H_
#define DATA_FORMATS__PYTORCH_ALPAKA_TEST__INTERFACE__DEVICE_H_

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/PyTorchAlpakaTest/interface/Layout.h"

template <typename TDev>
using SimpleInputCollectionDevice = PortableDeviceCollection<SimpleInputSoA, TDev>;

template <typename TDev>
using SimpleOutputCollectionDevice = PortableDeviceCollection<SimpleOutputSoA, TDev>;

#endif  // DATA_FORMATS__PYTORCH_ALPAKA_TEST__INTERFACE__DEVICE_H_