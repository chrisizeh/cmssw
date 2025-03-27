import FWCore.ParameterSet.Config as cms

ClassifierStruct = cms.EDProducer('Classifier@alpaka',
  inputs = cms.InputTag('DataLoaderStruct'),
  classificationModelPath = cms.FileInPath('model.pt'),
  numberOfClasses = cms.uint32(10),
  alpaka = cms.untracked.PSet(),
)