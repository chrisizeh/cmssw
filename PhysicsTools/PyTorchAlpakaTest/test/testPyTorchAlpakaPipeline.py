import FWCore.ParameterSet.Config as cms
from PhysicsTools.PyTorchAlpakaTest.options import args
  
args.parseArguments()
process = cms.Process("PyTorchAlpakaTestPipeline")

# enable multithreading
process.options.numberOfThreads = args.numberOfThreads if args.numberOfThreads > 1 else 1 
process.options.numberOfStreams = args.numberOfStreams if args.numberOfStreams > 1 else 1 

# enable alpaka and GPU support
process.load("Configuration.StandardSequences.Accelerators_cff")
process.load('HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi')

# process a limited number of events
process.maxEvents.input = args.numberOfEvents if args.numberOfEvents > 1 else 1 

# empty source 
process.source = cms.Source("EmptySource")

# print a message every event
process.MessageLogger.cerr.FwkReport.reportEvery = 1

# do not print the time and trigger reports at the end of the job
process.options.wantSummary = False

# load pipeline chain
process.load("PhysicsTools.PyTorchAlpakaTest.data_loader")
process.load("PhysicsTools.PyTorchAlpakaTest.classifier")

# setup chain configs
process.DataLoader = process.DataLoaderStruct.clone(
    batchSize = cms.uint32(args.batchSize if args.batchSize > 1 else 1),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
)
process.Classifier = process.ClassifierStruct.clone(
    inputs = cms.InputTag('DataLoader'),
    classificationModelPath = cms.FileInPath(args.classificationModelPath),
    numberOfClasses = cms.uint32(args.numberOfClasses),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
)

# schedule the modules
process.path = cms.Path(
    process.DataLoader + 
    process.Classifier
)

process.schedule = cms.Schedule(
    process.path
)