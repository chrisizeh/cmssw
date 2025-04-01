#include "PhysicsTools/PyTorchTest/plugins/alpaka/Combinatorial.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

// using namespace torch_alpaka;

Combinatorial::Combinatorial(edm::ParameterSet const& params)
  : EDProducer<>(params),
    inputs_token_{consumes(params.getParameter<edm::InputTag>("inputs"))},
    outputs_token_{produces()},
    kernels_(std::make_unique<Kernels>()) {}

void Combinatorial::produce(device::Event &event, const device::EventSetup &event_setup) {
  std::cout << "(Combinatorial) hash=" << torch_alpaka::tools::queue_hash(event.queue()) << std::endl;
  const auto& inputs = event.get(inputs_token_);
  const size_t batch_size = inputs.const_view().metadata().size();
  auto outputs = ParticleCollection(batch_size, event.queue());
  std::cout << "(Combinatorial) kernel" << std::endl;  
  kernels_->FillParticleCollection(event.queue(), outputs, 0.32f);
  event.emplace(outputs_token_, std::move(outputs));
  std::cout << "(Combinatorial) OK" << std::endl; 
}

void Combinatorial::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputs");
  descriptions.addWithDefaultLabel(desc);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(Combinatorial);
