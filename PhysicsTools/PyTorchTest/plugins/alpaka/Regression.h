#ifndef PHYSICS_TOOLS__PYTORCH_TEST__PLUGINS__ALPAKA__CLASSIFIER_H_
#define PHYSICS_TOOLS__PYTORCH_TEST__PLUGINS__ALPAKA__CLASSIFIER_H_

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
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/AlpakaConfig.h"
#include "PhysicsTools/PyTorch/interface/Model.h"
#include "PhysicsTools/PyTorchTest/plugins/alpaka/Kernels.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

using namespace torch_alpaka;  

class Regression : public stream::EDProducer<edm::GlobalCache<Model>> {
 public:
  Regression(const edm::ParameterSet &params, const Model *cache);

  static std::unique_ptr<Model> initializeGlobalCache(const edm::ParameterSet &params);
  static void globalEndJob(const Model *cache);

  void produce(device::Event &event, const device::EventSetup &event_setup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:  
  const device::EDGetToken<ParticleCollection> inputs_token_;
  const device::EDPutToken<RegressionCollection> outputs_token_;
  const std::string backend_;
  std::unique_ptr<Kernels> kernels_ = nullptr;
};

}  // ALPAKA_ACCELERATOR_NAMESPACE

#endif  // PHYSICS_TOOLS__PYTORCH_TEST__PLUGINS__ALPAKA__CLASSIFIER_H_