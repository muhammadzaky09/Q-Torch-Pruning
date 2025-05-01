import torch
from collections import OrderedDict

def remove_module_prefix(state_dict):
    """
    Removes the 'module.' prefix from state_dict keys.
    """
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        # Remove "module." prefix if it exists
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value
    return new_state_dict

# Load the original checkpoint (assumed saved as {'net': state_dict, 'acc': ..., 'epoch': ...})
checkpoint = torch.load('models/weights/LeNet5-8/lenet_w8a8.pth', map_location='cpu')

# Remove the prefix from the state dictionary
new_state_dict = remove_module_prefix(checkpoint['net'])
print(new_state_dict)
# Create a new checkpoint dictionary with the cleaned state dict


# Save the new checkpoint to a file
torch.save(new_state_dict, 'models/weights/lenet5_w8a8new.pth')
print("Saved the normal checkpoint as 'resnet50_w8a8.pth'.")
