import FWCore.ParameterSet.Config as cms

CombinatorialStruct = cms.EDProducer('Combinatorial@alpaka',
  inputs = cms.InputTag('DataLoaderStruct'),
  alpaka = cms.untracked.PSet(),
)