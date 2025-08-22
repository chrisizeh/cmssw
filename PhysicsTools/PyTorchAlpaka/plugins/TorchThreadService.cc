#include <iostream>

#include <torch/torch.h>
#include <torch/script.h>

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

class TorchThreadService {
public:
  TorchThreadService(const edm::ParameterSet& config, edm::ActivityRegistry& registry) {
    registry.watchPreGlobalBeginRun(this, &TorchThreadService::preGlobalBeginRun);
  };
  ~TorchThreadService() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void preGlobalBeginRun(edm::GlobalContext const&);
};

void TorchThreadService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  descriptions.add("TorchThreadService", desc);
  descriptions.setComment("Set single threading for pytorch after the module is loaded.");
}

void TorchThreadService::preGlobalBeginRun(edm::GlobalContext const&) {
  at::set_num_threads(1);
  at::set_num_interop_threads(1);
}

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(TorchThreadService);
