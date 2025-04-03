#ifndef PHYSICS_TOOLS__PYTORCH_TEST__PLUGINS__ALPAKA__DATA_LOADER_H_
#define PHYSICS_TOOLS__PYTORCH_TEST__PLUGINS__ALPAKA__DATA_LOADER_H_

#include "DataFormats/PyTorchTest/interface/alpaka/Collections.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

class DataLoader : public stream::EDProducer<> {
 public:
  DataLoader(const edm::ParameterSet &params);

  void produce(device::Event &event, const device::EventSetup &event_setup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:  
  const device::EDPutToken<torchportable::ParticleCollection> sic_put_token_;
  const std::string backend_;
  const uint32_t batch_size_;
};

}  // ALPAKA_ACCELERATOR_NAMESPACE

#endif  // PHYSICS_TOOLS__PYTORCH_TEST__PLUGINS__ALPAKA__DATA_LOADER_H_