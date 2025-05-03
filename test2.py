import torch
import torch.nn as nn
import brevitas.nn as qnn
from models import *
from torch_pruning.dependency import DependencyGraph
from torch_pruning.pruner.importance import MagnitudeImportance
from torch_pruning.pruner.function import prune_conv_out_channels


# Load model
model = QuantLeNet5()
model.load_state_dict(torch.load('models/weights/LeNet5-8/lenet_ckpt.pth'))

# Create example inputs
example_inputs = torch.randn(1, 1, 28, 28)

# Print model before pruning
print("Model before pruning:")
print(model)

# Create dependency graph
DG = DependencyGraph().build_dependency(model, example_inputs)

# Create pruner
imp = MagnitudeImportance()

# Print structure before pruning
print("\nStructure before pruning:")
for name, module in model.named_modules():
    if isinstance(module, (qnn.QuantConv2d, qnn.QuantLinear)):
        if isinstance(module, qnn.QuantConv2d):
            print(f"{name}: in_channels={module.in_channels}, out_channels={module.out_channels}")
        else:
            print(f"{name}: in_features={module.in_features}, out_features={module.out_features}")

# Prune conv2
conv_out_channels = model.conv2.out_channels
pruning_idxs = list(range(conv_out_channels // 2))  # Prune half of the channels
print(f"\nPruning conv2 channels: {pruning_idxs}")
group = DG.get_pruning_group(model.conv2, prune_conv_out_channels, pruning_idxs)
group.prune()

# Print structure after pruning
print("\nStructure after pruning:")
for name, module in model.named_modules():
    if isinstance(module, (qnn.QuantConv2d, qnn.QuantLinear)):
        if isinstance(module, qnn.QuantConv2d):
            print(f"{name}: in_channels={module.in_channels}, out_channels={module.out_channels}")
        else:
            print(f"{name}: in_features={module.in_features}, out_features={module.out_features}")

# Verify forward pass works
try:
    out = model(example_inputs)
    print(f"\nSuccessful forward pass! Output shape: {out.shape}")
except Exception as e:
    print(f"\nForward pass failed: {e}")