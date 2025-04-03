#include "PhysicsTools/PyTorchTest/plugins/alpaka/Regression.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

Regression::Regression(edm::ParameterSet const& params, const torch_alpaka::Model *cache)
  : EDProducer<edm::GlobalCache<torch_alpaka::Model>>(params),
    inputs_token_{consumes(params.getParameter<edm::InputTag>("inputs"))},
    outputs_token_{produces()},
    backend_{params.getParameter<std::string>("backend")},
    kernels_{std::make_unique<Kernels>()} {}

std::unique_ptr<torch_alpaka::Model> Regression::initializeGlobalCache(const edm::ParameterSet &param) {
  auto model_path = param.getParameter<edm::FileInPath>("regressionModelPath").fullPath();
  return std::make_unique<torch_alpaka::Model>(model_path);
}

void Regression::globalEndJob(const torch_alpaka::Model *cache) {}

void Regression::produce(device::Event &event, const device::EventSetup &event_setup) {
  std::cout << "(Regressor) queue_hash=" << torch_alpaka::tools::queue_hash(event.queue()) << std::endl;
  torch_alpaka::set_guard(event.queue());
  std::cout << "(Regressor -> set_guard) current_cuda_stream=" << torch_alpaka::tools::current_stream_hash() << std::endl;
  auto& inputs =  const_cast<torchportable::ParticleCollection&>(event.get(inputs_token_));;
  const size_t batch_size = inputs.const_view().metadata().size();
  auto outputs = torchportable::RegressionCollection(batch_size, event.queue());

  std::cout << "(Regressor -> event.get) current_cuda_stream=" << torch_alpaka::tools::current_stream_hash() << std::endl;
  torch_alpaka::SoAMetadata<torchportable::ParticleSoA> input_metadata(batch_size, inputs.buffer().data(), torch_alpaka::Float, 3);
  torch_alpaka::SoAMetadata<torchportable::RegressionSoA> output_metadata(batch_size, outputs.buffer().data(), torch_alpaka::Float, 1);
  torch_alpaka::ModelMetadata model_metadata(input_metadata, output_metadata);

  if (torch_alpaka::tools::device(event.queue()) != globalCache()->device()) 
    globalCache()->to(event.queue());
  assert(torch_alpaka::tools::device(event.queue()) == globalCache()->device());  

  std::cout << "(Regressor -> bind model) current_cuda_stream=" << torch_alpaka::tools::current_stream_hash() << std::endl;
  globalCache()->forward(model_metadata);
  std::cout << "(Regressor -> forward) current_cuda_stream=" << torch_alpaka::tools::current_stream_hash() << std::endl;
  
  // assert output match expected  
  kernels_->AssertRegression(event.queue(), outputs);
  std::cout << "(Regressor -> assert) current_cuda_stream=" << torch_alpaka::tools::current_stream_hash() << std::endl;
  event.emplace(outputs_token_, std::move(outputs));
  torch_alpaka::reset_guard();
  std::cout << "(Regressor -> reset_guard) current_cuda_stream=" << torch_alpaka::tools::current_stream_hash() << std::endl;
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
