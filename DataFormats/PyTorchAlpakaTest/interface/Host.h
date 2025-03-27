#ifndef DATA_FORMATS__PYTORCH_ALPAKA_TEST__INTERFACE__HOST_H_
#define DATA_FORMATS__PYTORCH_ALPAKA_TEST__INTERFACE__HOST_H_

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/PyTorchAlpakaTest/interface/Layout.h"

using SimpleInputCollectionHost = PortableHostCollection<SimpleInputSoA>;
using SimpleOutputCollectionHost = PortableHostCollection<SimpleOutputSoA>;

#endif  // DATA_FORMATS__PYTORCH_ALPAKA_TEST__INTERFACE__HOST_H_