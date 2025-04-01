import sys
import os
import torch
import torch.nn as nn

# Prepare the data directory
if len(sys.argv) >= 2:
    datadir = sys.argv[1]
else:
    thisdir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(os.path.dirname(thisdir), "bin", "data")

os.makedirs(datadir, exist_ok=True)

# Define the model class
class ClassifierModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=2):
        super(ClassifierModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # Fully connected layer
        self.softmax = nn.Softmax(dim=1)  # Softmax activation
    
    def forward(self, input):
        x = self.fc(input)  # Linear transformation
        x = self.softmax(x)  # Apply softmax
        return x

# Instantiate the model
module = ClassifierModel()

# Sample input tensor (batch_size=n, input_dim=3)
n = 32  # Example batch size
x = torch.rand((n, 3))  # Random input tensor with 3 features
# print("Input:", x)
# print("Output:", module(x))

# Save the model using TorchScript
tm = torch.jit.trace(module.eval(), x)
tm.save(f"{datadir}/classifier.pt")
tm.save(f"/afs/cern.ch/user/l/lmichals/private/CMSSW_15_0_0/src/classifier.pt")

print(f"{datadir}/classifier.pt created successfully!")