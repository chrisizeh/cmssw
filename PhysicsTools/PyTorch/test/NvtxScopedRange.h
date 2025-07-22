#ifndef PhysicsTools_PyTorch_test_NvtxScopedRange_h
#define PhysicsTools_PyTorch_test_NvtxScopedRange_h

#include <nvtx3/nvToolsExt.h>

namespace torchtest {

  class NvtxScopedRange {
  public:
    explicit NvtxScopedRange(const char* msg) { id_ = nvtxRangeStartA(msg); }

    void end() {
      if (active_) {
        active_ = false;
        nvtxRangeEnd(id_);
      }
    }

    ~NvtxScopedRange() { end(); }

  private:
    nvtxRangeId_t id_;
    bool active_ = true;
  };

}  // namespace torchtest

#endif  // PhysicsTools_PyTorch_test_NvtxScopedRange_h