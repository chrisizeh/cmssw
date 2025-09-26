## PhysicsTools/PyTorchAlpakaTest
A full implementation of a CMSSW pipeline with ML producers using the [PyTorchAlpaka](../PyTorchAlpaka) Wrapper.
To run the pipeline call 
```
cmsRun PhysicsTools/PyTorchAlpakaTest/test/testPyTorchAlpakaHeterogeneousPipeline.py backend=serial_sync batchSize=2
```
with the specific backend and batchSize.
