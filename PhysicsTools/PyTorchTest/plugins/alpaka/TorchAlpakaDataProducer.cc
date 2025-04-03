#include "DataFormats/PyTorchTest/interface/alpaka/Collections.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

class TorchAlpakaDataProducer : public stream::EDProducer<> {
 public:
  TorchAlpakaDataProducer(const edm::ParameterSet &params);

  void produce(device::Event &event, const device::EventSetup &event_setup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:  
  const device::EDPutToken<torchportable::ParticleCollection> sic_put_token_;
  const uint32_t batch_size_;
};

///////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATION
///////////////////////////////////////////////////////////////////////////////

TorchAlpakaDataProducer::TorchAlpakaDataProducer(edm::ParameterSet const& params)
  : EDProducer<>(params),
    sic_put_token_{produces()},
    batch_size_(params.getParameter<uint32_t>("batchSize")) {}

void TorchAlpakaDataProducer::produce(device::Event &event, const device::EventSetup &event_setup) {
  // create dummy data
  auto collection = torchportable::ParticleCollection(batch_size_, event.queue());
  collection.zeroInitialise(event.queue());
  event.emplace(sic_put_token_, std::move(collection));
}

void TorchAlpakaDataProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<uint32_t>("batchSize");
  descriptions.addWithDefaultLabel(desc);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(TorchAlpakaDataProducer);
