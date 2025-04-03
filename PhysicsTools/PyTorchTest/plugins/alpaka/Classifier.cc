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

class Classifier : public stream::EDProducer<edm::GlobalCache<torch_alpaka::Model>> {
 public:
  Classifier(const edm::ParameterSet &params, const torch_alpaka::Model *cache);

  static std::unique_ptr<torch_alpaka::Model> initializeGlobalCache(const edm::ParameterSet &params);
  static void globalEndJob(const torch_alpaka::Model *cache);

  void produce(device::Event &event, const device::EventSetup &event_setup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:  
  const device::EDGetToken<torchportable::ParticleCollection> inputs_token_;
  const device::EDPutToken<torchportable::ClassificationCollection> outputs_token_;
  const uint32_t number_of_classes_;
  const std::string backend_;
  std::unique_ptr<Kernels> kernels_ = nullptr;
};

///////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATION
///////////////////////////////////////////////////////////////////////////////

Classifier::Classifier(edm::ParameterSet const& params, const torch_alpaka::Model *cache)
  : EDProducer<edm::GlobalCache<torch_alpaka::Model>>(params),
    inputs_token_{consumes(params.getParameter<edm::InputTag>("inputs"))},
    outputs_token_{produces()},
    number_of_classes_{params.getParameter<uint32_t>("numberOfClasses")},
    backend_{params.getParameter<std::string>("backend")},
    kernels_{std::make_unique<Kernels>()} {}

std::unique_ptr<torch_alpaka::Model> Classifier::initializeGlobalCache(const edm::ParameterSet &param) {
  auto model_path = param.getParameter<edm::FileInPath>("classificationModelPath").fullPath();
  return std::make_unique<torch_alpaka::Model>(model_path);
}

void Classifier::globalEndJob(const torch_alpaka::Model *cache) {}

void Classifier::produce(device::Event &event, const device::EventSetup &event_setup) {
  std::cout << "(Classifier) queue_hash=" << torch_alpaka::tools::queue_hash(event.queue()) << std::endl;
  torch_alpaka::set_guard(event.queue());
  std::cout << "(Classifier -> set_guard) current_cuda_stream=" << torch_alpaka::tools::current_stream_hash() << std::endl;
  // TODO: const remove should not be done by user
  // in principle should not be done by anyone
  // @see: torch::from_blob(void*) 
  auto& inputs =  const_cast<torchportable::ParticleCollection&>(event.get(inputs_token_));;
  const size_t batch_size = inputs.const_view().metadata().size();
  auto outputs = torchportable::ClassificationCollection(batch_size, event.queue());

  std::cout << "(Classifier -> event.get) current_cuda_stream=" << torch_alpaka::tools::current_stream_hash() << std::endl;
  torch_alpaka::SoAMetadata<torchportable::ParticleSoA> input_metadata(batch_size, inputs.buffer().data(), torch_alpaka::Float, 3);
  torch_alpaka::SoAMetadata<torchportable::ClassificationSoA> output_metadata(batch_size, outputs.buffer().data(), torch_alpaka::Float, 2);
  torch_alpaka::ModelMetadata model_metadata(input_metadata, output_metadata);

  if (torch_alpaka::tools::device(event.queue()) != globalCache()->device()) 
    globalCache()->to(event.queue());
  assert(torch_alpaka::tools::device(event.queue()) == globalCache()->device());  

  std::cout << "(Classifier -> bind model) current_cuda_stream=" << torch_alpaka::tools::current_stream_hash() << std::endl;
  globalCache()->forward(model_metadata);
  std::cout << "(Classifier -> forward) current_cuda_stream=" << torch_alpaka::tools::current_stream_hash() << std::endl;

  // assert output match expected  
  kernels_->AssertClassification(event.queue(), outputs);
  std::cout << "(Classifier -> assert) current_cuda_stream=" << torch_alpaka::tools::current_stream_hash() << std::endl;
  event.emplace(outputs_token_, std::move(outputs));
  
  torch_alpaka::reset_guard();
  std::cout << "(Classifier -> reset_guard) current_cuda_stream=" << torch_alpaka::tools::current_stream_hash() << std::endl;
  std::cout << "(Classifier) OK" << std::endl; 
}

void Classifier::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputs");
  desc.add<edm::FileInPath>("classificationModelPath");
  desc.add<std::string>("backend");
  desc.add<uint32_t>("numberOfClasses");
  descriptions.addWithDefaultLabel(desc);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(Classifier);
