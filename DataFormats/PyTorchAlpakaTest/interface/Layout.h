#ifndef DATA_FORMATS__PYTORCH_ALPAKA_TEST__INTERFACE__LAYOUT_H_
#define DATA_FORMATS__PYTORCH_ALPAKA_TEST__INTERFACE__LAYOUT_H_

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

GENERATE_SOA_LAYOUT(SimpleInLayout,
  SOA_COLUMN(float, x),
  SOA_COLUMN(float, y),
  SOA_COLUMN(float, z)
)
using SimpleInputSoA = SimpleInLayout<>;

GENERATE_SOA_LAYOUT(SimpleOutLayout,
  SOA_COLUMN(float, probs)
)
using SimpleOutputSoA = SimpleOutLayout<>;

#endif  // DATA_FORMATS__PYTORCH_ALPAKA_TEST__INTERFACE__LAYOUT_H_