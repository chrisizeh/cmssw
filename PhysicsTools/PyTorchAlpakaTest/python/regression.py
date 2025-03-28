import FWCore.ParameterSet.Config as cms

RegressionStruct = cms.EDProducer('Regression@alpaka',
  inputs = cms.InputTag('DataLoaderStruct'),
  regressionModelPath = cms.FileInPath('model.pt'),
  backend = cms.string("serial_sync"),
  alpaka = cms.untracked.PSet(),
)