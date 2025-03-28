#include "PhysicsTools/PyTorchAlpakaTest/plugins/alpaka/Regression.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

using namespace torch_alpaka;

Regression::Regression(edm::ParameterSet const& params, const Model *cache)
  : EDProducer<edm::GlobalCache<Model>>(params),
    inputs_token_{consumes(params.getParameter<edm::InputTag>("inputs"))},
    outputs_token_{produces()},
    backend_{params.getParameter<std::string>("backend")} {}

std::unique_ptr<Model> Regression::initializeGlobalCache(const edm::ParameterSet &param) {
  auto model_path = param.getParameter<edm::FileInPath>("regressionModelPath").fullPath();
  return std::make_unique<Model>(model_path);
}

void Regression::globalEndJob(const Model *cache) {}

void Regression::produce(device::Event &event, const device::EventSetup &event_setup) {
  set_guard(event.queue());
  std::cout << "(Regressor) hash=" << tools::queue_hash(event.queue()) << std::endl;
  const auto& inputs = event.get(inputs_token_);
  const size_t batch_size = inputs.const_view().metadata().size();
  auto inputs_tmp = ParticleCollection(batch_size, event.queue());
  auto outputs = RegressionCollection(batch_size, event.queue());

  InputMetadata input_metadata(Float, 3);
  OutputMetadata output_metadata(Float, 1);
  ModelMetadata model_metadata(batch_size, input_metadata, output_metadata);

  if (tools::device(event.queue()) != globalCache()->device()) 
    globalCache()->to(event.queue());
  std::cout << "(Regressor) model=" << globalCache()->device() << std::endl;  
  assert(tools::device(event.queue()) == globalCache()->device());  
  globalCache()->forward<ParticleSoA, RegressionSoA>(
    model_metadata, inputs_tmp.buffer().data(), outputs.buffer().data());

  event.emplace(outputs_token_, std::move(outputs));
  reset_guard();
}

void Regression::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputs");
  desc.add<edm::FileInPath>("regressionModelPath");
  desc.add<std::string>("backend");
  descriptions.addWithDefaultLabel(desc);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(Regression);
