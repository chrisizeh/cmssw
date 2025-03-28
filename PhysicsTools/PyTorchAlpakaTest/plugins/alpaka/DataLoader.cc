#include "PhysicsTools/PyTorchAlpakaTest/plugins/alpaka/DataLoader.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

DataLoader::DataLoader(edm::ParameterSet const& params)
  : EDProducer<>(params),
    sic_put_token_{produces()},
    batch_size_(params.getParameter<uint32_t>("batchSize")) {}

void DataLoader::produce(device::Event &event, const device::EventSetup &event_setup) {
  auto collection = ParticleCollection(batch_size_, event.queue());
  collection.zeroInitialise(event.queue());
  event.emplace(sic_put_token_, std::move(collection));
}

void DataLoader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<uint32_t>("batchSize");
  descriptions.addWithDefaultLabel(desc);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(DataLoader);
