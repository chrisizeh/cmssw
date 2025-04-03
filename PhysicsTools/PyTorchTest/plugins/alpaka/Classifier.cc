#include "PhysicsTools/PyTorchTest/plugins/alpaka/Classifier.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

using namespace torch_alpaka;

Classifier::Classifier(edm::ParameterSet const& params, const Model *cache)
  : EDProducer<edm::GlobalCache<Model>>(params),
    inputs_token_{consumes(params.getParameter<edm::InputTag>("inputs"))},
    outputs_token_{produces()},
    number_of_classes_{params.getParameter<uint32_t>("numberOfClasses")},
    backend_{params.getParameter<std::string>("backend")},
    kernels_{std::make_unique<Kernels>()} {}

std::unique_ptr<Model> Classifier::initializeGlobalCache(const edm::ParameterSet &param) {
  auto model_path = param.getParameter<edm::FileInPath>("classificationModelPath").fullPath();
  return std::make_unique<Model>(model_path);
}

void Classifier::globalEndJob(const Model *cache) {}

void Classifier::produce(device::Event &event, const device::EventSetup &event_setup) {
  std::cout << "(Classifier) queue_hash=" << tools::queue_hash(event.queue()) << std::endl;
  set_guard(event.queue());
  std::cout << "(Classifier -> set_guard) current_cuda_stream=" << tools::current_stream_hash() << std::endl;
  // TODO: const remove should not be done by user
  // in principle should not be done by anyone
  // @see: torch::from_blob(void*) 
  auto& inputs =  const_cast<ParticleCollection&>(event.get(inputs_token_));;
  const size_t batch_size = inputs.const_view().metadata().size();
  auto outputs = ClassificationCollection(batch_size, event.queue());

  std::cout << "(Classifier -> event.get) current_cuda_stream=" << tools::current_stream_hash() << std::endl;
  SoAMetadata<ParticleSoA> input_metadata(batch_size, inputs.buffer().data(), Float, 3);
  SoAMetadata<ClassificationSoA> output_metadata(batch_size, outputs.buffer().data(), Float, 2);
  ModelMetadata model_metadata(input_metadata, output_metadata);

  if (tools::device(event.queue()) != globalCache()->device()) 
    globalCache()->to(event.queue());
  assert(tools::device(event.queue()) == globalCache()->device());  

  std::cout << "(Classifier -> bind model) current_cuda_stream=" << tools::current_stream_hash() << std::endl;
  globalCache()->forward(model_metadata);
  std::cout << "(Classifier -> forward) current_cuda_stream=" << tools::current_stream_hash() << std::endl;

  // assert output match expected  
  kernels_->AssertClassification(event.queue(), outputs);
  std::cout << "(Classifier -> assert) current_cuda_stream=" << tools::current_stream_hash() << std::endl;
  event.emplace(outputs_token_, std::move(outputs));
  
  reset_guard();
  std::cout << "(Classifier -> reset_guard) current_cuda_stream=" << tools::current_stream_hash() << std::endl;
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
