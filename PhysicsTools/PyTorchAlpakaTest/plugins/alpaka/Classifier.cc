#include "PhysicsTools/PyTorchAlpakaTest/plugins/alpaka/Classifier.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

using namespace torch_alpaka;

Classifier::Classifier(edm::ParameterSet const& params, const Model *cache)
  : EDProducer<edm::GlobalCache<Model>>(params),
    inputs_token_{consumes(params.getParameter<edm::InputTag>("inputs"))},
    outputs_token_{produces()},
    number_of_classes_{params.getParameter<uint32_t>("numberOfClasses")} {}

std::unique_ptr<Model> Classifier::initializeGlobalCache(const edm::ParameterSet &param) {
  auto model_path = param.getParameter<edm::FileInPath>("classificationModelPath").fullPath();
  return std::make_unique<Model>(model_path);
}

void Classifier::globalEndJob(const Model *cache) {}

void Classifier::produce(device::Event &event, const device::EventSetup &event_setup) {
  const auto& inputs = event.get(inputs_token_);
  const size_t batch_size = inputs.const_view().metadata().size();
  auto inputs_tmp = SimpleInputCollection(batch_size, event.queue());
  auto outputs = SimpleOutputCollection(batch_size, event.queue());

  InputMetadata input_metadata(Float, 3);
  OutputMetadata output_metadata(Float, 2);
  ModelMetadata model_metadata(batch_size, input_metadata, output_metadata);

  if (tools::device(event.queue()) != globalCache()->device()) 
    globalCache()->to(event.queue());
  std::cout << "(Classifier) model=" << globalCache()->device() << std::endl;  
  assert(tools::device(event.queue()) == globalCache()->device());  
  globalCache()->forward<SimpleInputSoA, SimpleOutputSoA>(
    model_metadata, inputs_tmp.buffer().data(), outputs.buffer().data());

  event.emplace(outputs_token_, std::move(outputs));
}

void Classifier::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputs");
  desc.add<edm::FileInPath>("classificationModelPath");
  desc.add<uint32_t>("numberOfClasses");
  descriptions.addWithDefaultLabel(desc);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(Classifier);
