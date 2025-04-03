import os

import FWCore.ParameterSet.VarParsing as VarParsing


args = VarParsing.VarParsing("analysis")

args.register(
    "numberOfThreads",
    8,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,
    "Number of CMSSW threads"
)

args.register(
    "numberOfStreams",
    1,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,
    "Number of CMSSW streams"
)

args.register(
    "numberOfEvents",
    3,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,
    "Number of events to process"
)

args.register(
    "backend",
    "serial_sync",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,         
    "Accelerator backend: serial_sync, cuda_async, or rocm_async"
)

args.register(
    "batchSize",
    2**20,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,        
    "Batch size"
)

args.register(
    "classificationModelPath",
    "PhysicsTools/PyTorchTest/models/classification.pt",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,         
    "Classification model"
)

args.register(
    "regressionModelPath",
    "PhysicsTools/PyTorchTest/models/regression.pt",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,         
    "Regression model"
)

args.register(
    "numberOfClasses",
    2,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,         
    "Classification model number of classes"
)