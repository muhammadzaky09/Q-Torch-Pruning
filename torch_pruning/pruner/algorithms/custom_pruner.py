import torch
import torch.nn as nn
import torch_pruning as tp
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity
from torch_pruning.pruner.function import BasePruningFunc


# ============== QuantConv2d Pruner ==============
class QuantConvPruner(BasePruningFunc):
    TARGET_MODULES = QuantConv2d
    
    def prune_out_channels(self, layer: QuantConv2d, idxs: list) -> nn.Module:
        keep_idxs = list(set(range(layer.out_channels)) - set(idxs))
        keep_idxs.sort()
        
        # Update channels count
        layer.out_channels = layer.out_channels - len(idxs)
        
        # Prune weight tensor
        if not layer.transposed:
            layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
        else:
            layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 1)
        
        # Prune bias if exists
        if layer.bias is not None:
            layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, 0)
        
        # Handle per-channel quantization parameters
        if hasattr(layer, 'weight_quant') and layer.weight_scaling_per_output_channel:
            # For per-channel scaling
            if hasattr(layer.weight_quant, 'scaling_impl') and hasattr(layer.weight_quant.scaling_impl, 'scaling'):
                scaling = layer.weight_quant.scaling_impl.scaling
                if scaling is not None and scaling.shape[0] == len(keep_idxs) + len(idxs):
                    layer.weight_quant.scaling_impl.scaling = torch.nn.Parameter(
                        torch.index_select(scaling, 0, torch.tensor(keep_idxs).to(scaling.device))
                    )
            
            # Handle any other per-channel parameters the weight quantizer might have
            if hasattr(layer.weight_quant, 'threshold'):
                threshold = layer.weight_quant.threshold
                if threshold is not None and threshold.shape[0] == len(keep_idxs) + len(idxs):
                    layer.weight_quant.threshold = torch.nn.Parameter(
                        torch.index_select(threshold, 0, torch.tensor(keep_idxs).to(threshold.device))
                    )
        
        return layer

    def prune_in_channels(self, layer: QuantConv2d, idxs: list) -> nn.Module:
        keep_idxs = list(set(range(layer.in_channels)) - set(idxs))
        keep_idxs.sort()
        
        # Update channels count
        layer.in_channels = layer.in_channels - len(idxs)
        
        # Handle groups for depthwise convolutions
        if layer.groups > 1:
            if layer.groups == layer.in_channels + len(idxs):  # Depthwise convolution
                layer.groups = layer.groups - len(idxs)
            elif layer.groups > 1:  # Group convolution
                # Ensure groups divides in_channels evenly
                assert (layer.in_channels % layer.groups) == 0, "Pruned in_channels must be divisible by groups"
        
        # Prune weight tensor
        if not layer.transposed:
            layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 1)
        else:
            layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
        
        return layer

    def get_out_channels(self, layer):
        return layer.out_channels

    def get_in_channels(self, layer):
        return layer.in_channels
    
    def get_in_channel_groups(self, layer):
        return layer.groups
    
    def get_out_channel_groups(self, layer):
        return layer.groups

class QuantLinearPruner(BasePruningFunc):
    TARGET_MODULES = QuantLinear
    
    def prune_out_channels(self, layer: QuantLinear, idxs: list) -> nn.Module:
        keep_idxs = list(set(range(layer.out_features)) - set(idxs))
        keep_idxs.sort()
        
        # Update feature count
        layer.out_features = layer.out_features - len(idxs)
        
        # Prune weight tensor
        layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
        
        # Prune bias if exists
        if layer.bias is not None:
            layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, 0)
        
        # Handle per-channel quantization parameters
        if hasattr(layer, 'weight_quant'):
            # For per-output-channel scaling (if present)
            if hasattr(layer, 'per_output_channel_scaling') and layer.per_output_channel_scaling:
                if hasattr(layer.weight_quant, 'scaling_impl') and hasattr(layer.weight_quant.scaling_impl, 'scaling'):
                    scaling = layer.weight_quant.scaling_impl.scaling
                    if scaling is not None and scaling.shape[0] == len(keep_idxs) + len(idxs):
                        layer.weight_quant.scaling_impl.scaling = torch.nn.Parameter(
                            torch.index_select(scaling, 0, torch.tensor(keep_idxs).to(scaling.device))
                        )
            
            # Handle any thresholds or other parameters
            if hasattr(layer.weight_quant, 'threshold'):
                threshold = layer.weight_quant.threshold
                if threshold is not None and threshold.shape[0] == len(keep_idxs) + len(idxs):
                    layer.weight_quant.threshold = torch.nn.Parameter(
                        torch.index_select(threshold, 0, torch.tensor(keep_idxs).to(threshold.device))
                    )
        
        return layer

    def prune_in_channels(self, layer: QuantLinear, idxs: list) -> nn.Module:
        keep_idxs = list(set(range(layer.in_features)) - set(idxs))
        keep_idxs.sort()
        
        # Update feature count
        layer.in_features = layer.in_features - len(idxs)
        
        # Prune weight tensor
        layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 1)
        
        return layer

    def get_out_channels(self, layer):
        return layer.out_features

    def get_in_channels(self, layer):
        return layer.in_features

# ============== QuantReLU Pruner ==============
class QuantReLUPruner(BasePruningFunc):
    TARGET_MODULES = QuantReLU
    
    def prune_out_channels(self, layer: QuantReLU, idxs: list) -> nn.Module:
        # QuantReLU doesn't typically have channels to prune directly
        # It inherits the shape of the input tensor
        # No structural changes needed
        return layer

    # Make prune_in_channels equal to prune_out_channels for consistency
    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        # QuantReLU doesn't have a fixed number of channels
        # Returns None to indicate this is determined at runtime
        return None

    def get_in_channels(self, layer):
        # Same as out_channels
        return None

# ============== QuantIdentity Pruner ==============
class QuantIdentityPruner(BasePruningFunc):
    TARGET_MODULES = QuantIdentity
    
    def prune_out_channels(self, layer: QuantIdentity, idxs: list) -> nn.Module:
        # QuantIdentity is just a pass-through operation with quantization
        # No structural changes needed
        return layer

    # Make prune_in_channels equal to prune_out_channels for consistency
    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        # Returns None to indicate this is determined at runtime
        return None

    def get_in_channels(self, layer):
        # Same as out_channels
        return None
    

def register_brevitas_pruners(dependency_graph):
    
    
    quant_conv_pruner = QuantConvPruner()
    quant_linear_pruner = QuantLinearPruner()
    quant_relu_pruner = QuantReLUPruner()
    quant_identity_pruner = QuantIdentityPruner()
    
    dependency_graph.register_customized_layer(QuantConv2d, quant_conv_pruner)
    dependency_graph.register_customized_layer(QuantLinear, quant_linear_pruner)
    dependency_graph.register_customized_layer(QuantReLU, quant_relu_pruner)
    dependency_graph.register_customized_layer(QuantIdentity, quant_identity_pruner)