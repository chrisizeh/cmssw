#include "PhysicsTools/PyTorchTest/plugins/alpaka/Regression.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

using namespace torch_alpaka;

Regression::Regression(edm::ParameterSet const& params, const Model *cache)
  : EDProducer<edm::GlobalCache<Model>>(params),
    inputs_token_{consumes(params.getParameter<edm::InputTag>("inputs"))},
    outputs_token_{produces()},
    backend_{params.getParameter<std::string>("backend")},
    kernels_{std::make_unique<Kernels>()} {}

std::unique_ptr<Model> Regression::initializeGlobalCache(const edm::ParameterSet &param) {
  auto model_path = param.getParameter<edm::FileInPath>("regressionModelPath").fullPath();
  return std::make_unique<Model>(model_path);
}

void Regression::globalEndJob(const Model *cache) {}

void Regression::produce(device::Event &event, const device::EventSetup &event_setup) {
  std::cout << "(Regressor) queue_hash=" << tools::queue_hash(event.queue()) << std::endl;
  set_guard(event.queue());
  std::cout << "(Regressor -> set_guard) current_cuda_stream=" << tools::current_stream_hash() << std::endl;
  auto& inputs =  const_cast<ParticleCollection&>(event.get(inputs_token_));;
  const size_t batch_size = inputs.const_view().metadata().size();
  auto outputs = RegressionCollection(batch_size, event.queue());

  std::cout << "(Regressor -> event.get) current_cuda_stream=" << tools::current_stream_hash() << std::endl;
  SoAMetadata<ParticleSoA> input_metadata(batch_size, inputs.buffer().data(), Float, 3);
  SoAMetadata<RegressionSoA> output_metadata(batch_size, outputs.buffer().data(), Float, 1);
  ModelMetadata model_metadata(input_metadata, output_metadata);

  if (tools::device(event.queue()) != globalCache()->device()) 
    globalCache()->to(event.queue());
  assert(tools::device(event.queue()) == globalCache()->device());  

  std::cout << "(Regressor -> bind model) current_cuda_stream=" << tools::current_stream_hash() << std::endl;
  globalCache()->forward(model_metadata);
  std::cout << "(Regressor -> forward) current_cuda_stream=" << tools::current_stream_hash() << std::endl;
  
  // assert output match expected  
  kernels_->AssertRegression(event.queue(), outputs);
  std::cout << "(Regressor -> assert) current_cuda_stream=" << tools::current_stream_hash() << std::endl;
  event.emplace(outputs_token_, std::move(outputs));
  reset_guard();
  std::cout << "(Regressor -> reset_guard) current_cuda_stream=" << tools::current_stream_hash() << std::endl;
  std::cout << "(Regressor) OK" << std::endl; 
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
