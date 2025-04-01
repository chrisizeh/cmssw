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

# Define the regression model class
class RegressionModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=1):
        super(RegressionModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=True)  # Fully connected layer
        # repro: weights zero and bias to 0.5
        self.fc.weight.data.fill_(0)
        self.fc.bias.data.fill_(0.5)
    
    def forward(self, input):
        x = self.fc(input)  # Linear transformation
        return x

# Instantiate the model
module = RegressionModel()

# Sample input tensor (batch_size=n, input_dim=3)
n = 4  # Example batch size
x = torch.zeros((n, 3))  # Random input tensor with 3 features
# print("Input:", x)
# print("Output:", module(x))

# Save the model using TorchScript
tm = torch.jit.trace(module.eval(), x)
tm.save(f"{datadir}/regression.pt")
tm.save(f"/afs/cern.ch/user/l/lmichals/public/CMSSW_15_0_0/src/regression.pt")

print("regression.pt created successfully!")