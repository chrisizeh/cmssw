#include <alpaka/alpaka.hpp>
#include <torch/torch.h>
#include <torch/script.h>

#include "DataFormats/PyTorchTest/interface/alpaka/Collections.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/AlpakaConfig.h"
#include "PhysicsTools/PyTorch/interface/Model.h"
#include "PhysicsTools/PyTorch/interface/SoAMetadata.h"
#include "PhysicsTools/PyTorchTest/plugins/alpaka/Kernels.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

class TorchAlpakaClassificationProducer : public stream::EDProducer<edm::GlobalCache<torch_alpaka::Model>> {
 public:
  TorchAlpakaClassificationProducer(const edm::ParameterSet &params, const torch_alpaka::Model *cache);

  static std::unique_ptr<torch_alpaka::Model> initializeGlobalCache(const edm::ParameterSet &params);
  static void globalEndJob(const torch_alpaka::Model *cache);

  void produce(device::Event &event, const device::EventSetup &event_setup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:  
  const device::EDGetToken<torchportable::ParticleCollection> inputs_token_;
  const device::EDPutToken<torchportable::ClassificationCollection> outputs_token_;
  std::unique_ptr<Kernels> kernels_ = nullptr;
};

///////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATION
///////////////////////////////////////////////////////////////////////////////

TorchAlpakaClassificationProducer::TorchAlpakaClassificationProducer(edm::ParameterSet const& params, const torch_alpaka::Model *cache)
  : EDProducer<edm::GlobalCache<torch_alpaka::Model>>(params),
    inputs_token_{consumes(params.getParameter<edm::InputTag>("inputs"))},
    outputs_token_{produces()},
    kernels_{std::make_unique<Kernels>()} {}

std::unique_ptr<torch_alpaka::Model> TorchAlpakaClassificationProducer::initializeGlobalCache(const edm::ParameterSet &param) {
  auto model_path = param.getParameter<edm::FileInPath>("modelPath").fullPath();
  return std::make_unique<torch_alpaka::Model>(model_path);
}

void TorchAlpakaClassificationProducer::globalEndJob(const torch_alpaka::Model *cache) {}

void TorchAlpakaClassificationProducer::produce(device::Event &event, const device::EventSetup &event_setup) {
  // guard torch internal operations to not conflict with fw execution scheme
  std::cout << "(Classification) qhash=" << torch_alpaka::tools::queue_hash(event.queue()) << std::endl;
  std::cout << "(Classification) chash=" << torch_alpaka::tools::current_stream_hash(event.queue()) << std::endl;
  torch_alpaka::set_guard(event.queue());
  std::cout << "(Classification::set_guard) chash=" << torch_alpaka::tools::current_stream_hash(event.queue()) << std::endl;
  // sanity check 
  assert(torch_alpaka::tools::queue_hash(event.queue()) == torch_alpaka::tools::current_stream_hash(event.queue()));

  // get data
  // TODO: const_cast should not be done by user
  // in principle should not be done by anyone
  // @see: torch::from_blob(void*) 
  std::cout << "(Classification::torchportable) chash=" << torch_alpaka::tools::current_stream_hash(event.queue()) << std::endl;
  auto& inputs =  const_cast<torchportable::ParticleCollection&>(event.get(inputs_token_));;
  const size_t batch_size = inputs.const_view().metadata().size();
  auto outputs = torchportable::ClassificationCollection(batch_size, event.queue());

  // metadata for automatic tensor conversion
  std::cout << "(Classification::metadata) chash=" << torch_alpaka::tools::current_stream_hash(event.queue()) << std::endl;
  torch_alpaka::SoAMetadata<torchportable::ParticleSoA> input_metadata(
    batch_size, inputs.buffer().data(), torch_alpaka::Float, 3);
  torch_alpaka::SoAMetadata<torchportable::ClassificationSoA> output_metadata(
    batch_size, outputs.buffer().data(), torch_alpaka::Float, 2);
  torch_alpaka::ModelMetadata model_metadata(input_metadata, output_metadata);

  // inference
  std::cout << "(Classification::forward) chash=" << torch_alpaka::tools::current_stream_hash(event.queue()) << std::endl;
  if (torch_alpaka::tools::device(event.queue()) != globalCache()->device()) 
    globalCache()->to(event.queue());
  assert(torch_alpaka::tools::device(event.queue()) == globalCache()->device());  
  globalCache()->forward(model_metadata);

  // assert output match expected 
  std::cout << "(Classification::assert) chash=" << torch_alpaka::tools::current_stream_hash(event.queue()) << std::endl;
  kernels_->AssertClassification(event.queue(), outputs);
  event.emplace(outputs_token_, std::move(outputs));
  std::cout << "(Classification) OK" << std::endl; 
}

void TorchAlpakaClassificationProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputs");
  desc.add<edm::FileInPath>("modelPath");
  descriptions.addWithDefaultLabel(desc);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(TorchAlpakaClassificationProducer);
