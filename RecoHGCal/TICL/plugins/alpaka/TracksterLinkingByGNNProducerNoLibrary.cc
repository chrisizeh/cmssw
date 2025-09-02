#include <alpaka/alpaka.hpp>
#include <torch/torch.h>
#include <torch/script.h>

#include "DataFormats/PyTorchTest/interface/alpaka/Collections.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterSoADeviceCollection.h"
#include "DataFormats/HGCalReco/interface/TracksterSoAHostCollection.h"
#include "DataFormats/HGCalReco/interface/TICLGraph.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
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

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TracksterLinkingByGNNNoLibrary : public stream::EDProducer<> {
  public:
    TracksterLinkingByGNNNoLibrary(const edm::ParameterSet &params);

    void produce(device::Event &event, const device::EventSetup &event_setup) override;
    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  private:
    const device::EDGetToken<TrackstersSoADeviceCollection> inputs_token_;
    const device::EDPutToken<TrackstersGNNOutputSoADeviceCollection> outputs_token_;
    ::torch::jit::script::Module model_;
  };

  TracksterLinkingByGNNNoLibrary::TracksterLinkingByGNNNoLibrary(edm::ParameterSet const &params)
      : EDProducer<>(params),
        inputs_token_{consumes(params.getParameter<edm::InputTag>("inputs"))},
        outputs_token_{produces()} {
    model_ = ::torch::jit::load(params.getParameter<edm::FileInPath>("modelPath").fullPath(), ::torch::Device(at::kCUDA, 0));
  }

  void TracksterLinkingByGNNNoLibrary::produce(device::Event &event, const device::EventSetup &event_setup) {
    // get data
    auto &inputs = const_cast<TrackstersSoADeviceCollection &>(event.get(inputs_token_));
    const int numNodes = inputs.const_view<GNNNodeSoA>().metadata().size();
    const int numEdges = inputs.const_view<GNNEdgeSoA>().metadata().size();

    std::array<int, 3> const sizes{{numNodes, numEdges, numEdges}};
    auto hostCollection = TrackstersSoAHostCollection(sizes, event.queue());
    auto& nodeView = hostCollection.view<GNNNodeSoA>();
    auto& edgeView = hostCollection.view<GNNEdgeSoA>();
    auto& edgeIndexView = hostCollection.view<GNNEdgeIndexSoA>();

    alpaka::memcpy(event.queue(), hostCollection.buffer(), inputs.buffer());
    alpaka::wait(event.queue());

    //auto nodeTensor = ::torch::tensor(hostCollection.buffer()).view({numNodes, 28});
    //auto edgeTensor = ::torch::tensor(hostCollection.buffer()).view({numEdges, 5});
    //auto indexTensor = ::torch::tensor(hostCollection.buffer()).view({numNodes, 2});

    auto nodeTensor = ::torch::rand({numNodes, 29});
    auto edgeTensor = ::torch::rand({numEdges, 5});
    auto indexTensor = ::torch::zeros({numEdges, 2}, ::torch::kLong);
    
    for (int i = 0; i < numNodes; i++) {
      nodeTensor[i][28] = nodeView.time()[i]; 
      nodeTensor[i][16] = nodeView.raw_energy()[i]; 
      nodeTensor[i][17] = nodeView.raw_em_energy()[i]; 

      nodeTensor[i][0] = nodeView.barycenter_x()[i]; 
      nodeTensor[i][1] = nodeView.barycenter_y()[i]; 
      nodeTensor[i][2] = nodeView.barycenter_z()[i]; 
      nodeTensor[i][3] = nodeView.barycenter_eta()[i]; 
      nodeTensor[i][4] = nodeView.barycenter_phi()[i]; 

      nodeTensor[i][5] = nodeView.eigenvector0_x()[i]; 
      nodeTensor[i][6] = nodeView.eigenvector0_y()[i]; 
      nodeTensor[i][7] = nodeView.eigenvector0_z()[i]; 

      nodeTensor[i][8] = nodeView.eigenvalue1()[i]; 
      nodeTensor[i][9] = nodeView.eigenvalue2()[i]; 
      nodeTensor[i][10] = nodeView.eigenvalue3()[i]; 

      nodeTensor[i][11] = nodeView.sigmasPCA1()[i]; 
      nodeTensor[i][12] = nodeView.sigmasPCA2()[i]; 
      nodeTensor[i][13] = nodeView.sigmasPCA3()[i]; 

      nodeTensor[i][18] = nodeView.photon_prob()[i]; 
      nodeTensor[i][19] = nodeView.electron_prob()[i]; 
      nodeTensor[i][20] = nodeView.muon_prob()[i]; 
      nodeTensor[i][21] = nodeView.neutral_pion_prob()[i]; 
      nodeTensor[i][22] = nodeView.charged_hadron_prob()[i]; 
      nodeTensor[i][23] = nodeView.neutral_hadron_prob()[i]; 

      nodeTensor[i][14] = nodeView.num_LCs()[i];
      nodeTensor[i][15] = nodeView.num_hits()[i];

      nodeTensor[i][24] = nodeView.z_min()[i]; 
      nodeTensor[i][25] = nodeView.z_max()[i]; 
      nodeTensor[i][26] = nodeView.LC_density()[i]; 
      nodeTensor[i][27] = nodeView.trackster_density()[i]; 
    }

    for (int i = 0; i < numEdges; i++) {
      edgeTensor[i][0] = edgeView.raw_energy()[i]; 
      edgeTensor[i][1] = edgeView.barycenter_z()[i]; 
      edgeTensor[i][2] = edgeView.barycenter_xy()[i]; 
      edgeTensor[i][3] = edgeView.eigenvector0()[i]; 
      edgeTensor[i][4] = edgeView.time()[i]; 

      indexTensor[i][0] = edgeIndexView.in()[i]; 
      indexTensor[i][1] = edgeIndexView.out()[i]; 
    }

	nodeTensor = nodeTensor.to(at::kCUDA, 0);
	edgeTensor = edgeTensor.to(at::kCUDA, 0);
	indexTensor = indexTensor.to(at::kCUDA, 0);

    auto outputTensor = model_.forward({{nodeTensor, edgeTensor, indexTensor}}).toTensor();
    auto outputs = TrackstersGNNOutputSoADeviceCollection(numEdges, event.queue());
    auto outputsHost = TrackstersGNNOutputSoAHostCollection(numEdges, event.queue());
    auto view = outputsHost.view();
    outputs.zeroInitialise(event.queue());
    for (int i = 0; i < numEdges; i++) {
      view.score()[i] = outputTensor[i][0].item<float>();
    }
    alpaka::memcpy(event.queue(), outputs.buffer(), outputsHost.buffer());
    alpaka::wait(event.queue());
    event.emplace(outputs_token_, std::move(outputs));
    alpaka::wait(event.queue());
  }

  void TracksterLinkingByGNNNoLibrary::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("inputs");
    desc.add<edm::FileInPath>("modelPath");
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(TracksterLinkingByGNNNoLibrary);
