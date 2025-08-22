#include <alpaka/alpaka.hpp>
#include <torch/torch.h>
#include <torch/script.h>

#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
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
#include "PhysicsTools/PyTorchAlpaka/test/NvtxScopedRange.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/Device.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/ModelJitAlpaka.h"
#include "PhysicsTools/PyTorchAlpaka/interface/FwkGuards.h"
#include "PhysicsTools/PyTorchAlpaka/plugins/alpaka/Kernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  using namespace ALPAKA_ACCELERATOR_NAMESPACE::torch;
  using namespace ::cms::torch;

  /**
   * @class JitRegressionProducer
   * @brief EDProducer that runs a regression model on input particles with Alpaka backend.
   *
   * Producer reads a collection of particles, runs inference on a ahead-of-time compiled model,
   * and writes the regression outputs to the event. 
   * 
   * Demonstrates the interoperability of Alpaka with PyTorch and the use of SoA data layout with AOT models.
   */
  class JitRegressionProducer : public stream::EDProducer<> {
  public:
    JitRegressionProducer(const edm::ParameterSet &params);

    void produce(device::Event &event, const device::EventSetup &event_setup) override;
    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  private:
    const device::EDGetToken<torchportabletest::ParticleCollection> inputs_token_;    /**< Token to get input data. */
    const device::EDPutToken<torchportabletest::RegressionCollection> outputs_token_; /**< Token to store output data. */
    std::unique_ptr<ModelJitAlpaka> model_;                                                 /**< Cache for the JIT model. */
  };

  JitRegressionProducer::JitRegressionProducer(edm::ParameterSet const &params)
      : EDProducer<>(params),
        inputs_token_{consumes(params.getParameter<edm::InputTag>("inputs"))},
        outputs_token_{produces()} {
    std::string path = params.getParameter<edm::FileInPath>("modelPath").fullPath();
    model_ = std::make_unique<ModelJitAlpaka>(path);
  }

  /**
   * @brief Performs inference on the event data.
   * @param event The event to process.
   * @param event_setup Event setup information.
   */
  void JitRegressionProducer::produce(device::Event &event, const device::EventSetup &event_setup) {
    auto t1 = std::chrono::steady_clock::now();

    // debug stream usage in concurrently scheduled modules
    auto msg = fmt::format("RegressionJit::produce [E: {}]", event.id().event());
    ::torchtest::NvtxScopedRange produce_range(msg.c_str());

    // guard torch internal operations to not conflict with cmssw fw scheme
    Guard guard(event.queue());
    // sanity check for debug
    //assert(cms::torch::alpaka::queue_hash(event.queue()) == cms::torch::alpaka::current_stream_hash(event.queue()));

    // get data
    // TODO: const_cast should not be done by user
    // in principle should not be done by anyone
    // @see: torch::from_blob(void*)
    auto &inputs = const_cast<torchportabletest::ParticleCollection &>(event.get(inputs_token_));
    const size_t batch_size = inputs.const_view().metadata().size();
    auto outputs = torchportabletest::RegressionCollection(batch_size, event.queue());

    // metadata for automatic tensor conversion
    auto input_records = inputs.view().records();
    auto output_records = outputs.view().records();
    SoAMetadata<torchportabletest::ParticleSoA> inputs_metadata(batch_size);
    inputs_metadata.append_block("features", input_records.pt(), input_records.eta(), input_records.phi());
    SoAMetadata<torchportabletest::RegressionSoA> outputs_metadata(batch_size);
    outputs_metadata.append_block("preds", output_records.reco_pt());
    ModelMetadata<torchportabletest::ParticleSoA, torchportabletest::RegressionSoA> metadata(
        inputs_metadata, outputs_metadata);

    // inference
    ::torchtest::NvtxScopedRange move_to_device("Regression::move_to_device");
    if (alpakatools::device(event.queue()) != model_->device()) {
      std::cout << "(RegressionJit) E: " << event.id().event() << " Model: " << model_->device() << " -> "
                << alpakatools::device(event.queue()) << std::endl;
      model_->to(event.queue());
    }
    assert(alpakatools::device(event.queue()) == model_->device());
    move_to_device.end();
    ::torchtest::NvtxScopedRange infer_range("Regression::inference");
    model_->forward(metadata);
    infer_range.end();

    // assert output match expected
    assertRegression(event.queue(), outputs);
    event.emplace(outputs_token_, std::move(outputs));
    ::alpaka::wait(event.queue());
    auto t2 = std::chrono::steady_clock::now();
    std::cout << "(RegressionJit) E: " << event.id().event() << " OK - "
              << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " us" << std::endl;
    produce_range.end();
  }

  /**
   * @brief Fills the parameter descriptions for this module.
   * @param descriptions Configuration descriptions object to fill.
   */
  void JitRegressionProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("inputs");
    desc.add<edm::FileInPath>("modelPath");
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

DEFINE_FWK_ALPAKA_MODULE(torchtest::JitRegressionProducer);
