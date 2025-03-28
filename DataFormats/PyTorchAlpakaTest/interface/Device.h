#ifndef DATA_FORMATS__PYTORCH_ALPAKA_TEST__INTERFACE__DEVICE_H_
#define DATA_FORMATS__PYTORCH_ALPAKA_TEST__INTERFACE__DEVICE_H_

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/PyTorchAlpakaTest/interface/Layout.h"

template <typename TDev>
using ParticleCollectionDevice = PortableDeviceCollection<ParticleSoA, TDev>;

template <typename TDev>
using ClassificationCollectionDevice = PortableDeviceCollection<ClassificationSoA, TDev>;

template <typename TDev>
using RegressionCollectionDevice = PortableDeviceCollection<RegressionSoA, TDev>;

#endif  // DATA_FORMATS__PYTORCH_ALPAKA_TEST__INTERFACE__DEVICE_H_