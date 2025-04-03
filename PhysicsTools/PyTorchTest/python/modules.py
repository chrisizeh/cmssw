import FWCore.ParameterSet.Config as cms


TorchAlpakaDataProducerStruct = cms.EDProducer('TorchAlpakaDataProducer@alpaka',
  batchSize = cms.uint32(32),
  alpaka = cms.untracked.PSet(),
)

TorchAlpakaClassificationProducerStruct = cms.EDProducer('TorchAlpakaClassificationProducer@alpaka',
  inputs = cms.InputTag('TorchAlpakaDataProducerStruct'),
  modelPath = cms.FileInPath('model.pt'),
  alpaka = cms.untracked.PSet(),
)

TorchAlpakaRegressionProducerStruct = cms.EDProducer('TorchAlpakaRegressionProducer@alpaka',
  inputs = cms.InputTag('TorchAlpakaDataProducerStruct'),
  modelPath = cms.FileInPath('model.pt'),
  alpaka = cms.untracked.PSet(),
)

AlpakaCombinatoricsProducerStruct = cms.EDProducer('AlpakaCombinatoricsProducer@alpaka',
  inputs = cms.InputTag('TorchAlpakaDataProducerStruct'),
  alpaka = cms.untracked.PSet(),
)