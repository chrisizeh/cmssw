import FWCore.ParameterSet.Config as cms

DataLoaderStruct = cms.EDProducer('DataLoader@alpaka',
  batchSize = cms.uint32(32),
  alpaka = cms.untracked.PSet(),
)