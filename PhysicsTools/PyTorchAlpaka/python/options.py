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
    4,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,
    "Number of CMSSW streams"
)

args.register(
    "numberOfEvents",
    10,
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

# Classification model options

args.register(
    "classificationModelPath",
    "PhysicsTools/PyTorchAlpaka/python/create_linear_dnn.py",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,         
    "Classification model (just-in-time compiled)"
)

# Regression model options

args.register(
    "regressionModelPath",
    "PhysicsTools/PyTorch/models/jit_regression_model.pt",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,         
    "Regression model (just-in-time compiled)"
)
