#ifndef DATA_FORMATS__PYTORCH_ALPAKA_TEST__INTERFACE__ALPAKA__COLLECTIONS_H_
#define DATA_FORMATS__PYTORCH_ALPAKA_TEST__INTERFACE__ALPAKA__COLLECTIONS_H_

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/PyTorchAlpakaTest/interface/Device.h"
#include "DataFormats/PyTorchAlpakaTest/interface/Host.h"
#include "DataFormats/PyTorchAlpakaTest/interface/Layout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

using ParticleCollection =
    std::conditional_t<
        std::is_same_v<Device, alpaka::DevCpu>, 
        ParticleCollectionHost, 
        ParticleCollectionDevice<Device>>;

using ClassificationCollection =
    std::conditional_t<
        std::is_same_v<Device, alpaka::DevCpu>, 
        ClassificationCollectionHost, 
        ClassificationCollectionDevice<Device>>;

using RegressionCollection =
    std::conditional_t<
        std::is_same_v<Device, alpaka::DevCpu>, 
        RegressionCollectionHost, 
        RegressionCollectionDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(ParticleCollection, ParticleCollectionHost);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(ClassificationCollection, ClassificationCollectionHost);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(RegressionCollection, RegressionCollectionHost);

#endif  // DATA_FORMATS__PYTORCH_ALPAKA_TEST__INTERFACE__ALPAKA__COLLECTIONS_H_