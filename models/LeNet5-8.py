import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity
from brevitas.quant import IntBias
from brevitas.core.restrict_val import RestrictValueType
from brevitas.quant import Int8ActPerTensorFloat, Uint8ActPerTensorFloat, Int32Bias
from brevitas.quant import Int8WeightPerTensorFloat, Int8WeightPerChannelFloat
from brevitas.quant.scaled_int import Int32Bias
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
import numpy as np
import time
import warnings
import os

# 8-bit weight quantization configurations
class Common8bitWeightPerTensorQuant(Int8WeightPerTensorFloat):
    scaling_min_val = 2e-16

class Common8bitWeightPerChannelQuant(Int8WeightPerChannelFloat):
    scaling_per_output_channel = True
    scaling_min_val = 2e-16

# 8-bit activation quantization configurations
class Common8bitActQuant(Int8ActPerTensorFloat):
    scaling_min_val = 2e-16
    restrict_scaling_type = RestrictValueType.LOG_FP

class Common8bitUintActQuant(Uint8ActPerTensorFloat):
    scaling_min_val = 2e-16
    restrict_scaling_type = RestrictValueType.LOG_FP
    
class QuantizedLeNet5_8bit(nn.Module):
    def __init__(self):
        super(QuantizedLeNet5_8bit, self).__init__()
        
        # Input quantization (8-bit)
        self.quant_inp = QuantIdentity(
            bit_width=8,
            return_quant_tensor=True,
            act_quant=Common8bitActQuant)
        
        # First convolutional layer (8-bit)
        self.conv1 = QuantConv2d(
            1, 6, kernel_size=5, stride=1, padding=2,
            weight_bit_width=8,
            bias=True,
            weight_quant=Common8bitWeightPerChannelQuant,
            input_quant=Common8bitActQuant,
            output_quant=Common8bitActQuant,
            return_quant_tensor=True)
        self.relu1 = QuantReLU(
            bit_width=8,
            act_quant=Common8bitUintActQuant,
            return_quant_tensor=True)
        
        # First average pooling layer
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Second convolutional layer (8-bit)
        self.conv2 = QuantConv2d(
            6, 16, kernel_size=5, stride=1, padding=0,
            weight_bit_width=8,
            bias=True,
            weight_quant=Common8bitWeightPerChannelQuant,
            input_quant=Common8bitActQuant,
            output_quant=Common8bitActQuant,
            return_quant_tensor=True)
        self.relu2 = QuantReLU(
            bit_width=8,
            act_quant=Common8bitUintActQuant,
            return_quant_tensor=True)
        
        # Second average pooling layer
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # First fully connected layer (8-bit)
        self.fc1 = QuantLinear(
            16 * 5 * 5, 120,
            bias=True,
            weight_bit_width=8,
            weight_quant=Common8bitWeightPerTensorQuant,
            input_quant=Common8bitActQuant,
            output_quant=Common8bitActQuant,
            return_quant_tensor=True)
        self.relu3 = QuantReLU(
            bit_width=8,
            act_quant=Common8bitUintActQuant,
            return_quant_tensor=True)
        
        # Second fully connected layer (8-bit)
        self.fc2 = QuantLinear(
            120, 84,
            bias=True,
            weight_bit_width=8,
            weight_quant=Common8bitWeightPerTensorQuant,
            input_quant=Common8bitActQuant,
            output_quant=Common8bitActQuant,
            return_quant_tensor=True)
        self.relu4 = QuantReLU(
            bit_width=8,
            act_quant=Common8bitUintActQuant,
            return_quant_tensor=True)
        
        # Output layer (8-bit)
        self.fc3 = QuantLinear(
            84, 10,
            bias=True,
            weight_bit_width=8,
            weight_quant=Common8bitWeightPerTensorQuant,
            input_quant=Common8bitActQuant,
            output_quant=Common8bitActQuant,
            return_quant_tensor=True)
        
        # Output quantization (8-bit)
        self.quant_out = QuantIdentity(
            bit_width=8,
            act_quant=Common8bitActQuant,
            return_quant_tensor=True)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.relu1(self.conv1(x))
        x = self.avg_pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.avg_pool2(x)
        x = x.view(x.size(0), -1)  
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        x = self.quant_out(x)
        return F.log_softmax(x, dim=1)