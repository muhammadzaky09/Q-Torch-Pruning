'''ResNet with Brevitas quantization support.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import brevitas.nn as qnn
from brevitas.nn import QuantIdentity
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int16Bias
from brevitas.quant import TruncTo8bit
from brevitas.quant_tensor import QuantTensor



class QuantBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, weight_bit_width=8, act_bit_width=8):
        super(QuantBasicBlock, self).__init__()
        
        # First quantized conv + bn + activation
        self.conv1 = qnn.QuantConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
            weight_quant=Int8WeightPerTensorFloat,
            weight_bit_width=weight_bit_width
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            bit_width=act_bit_width,
            return_quant_tensor=True
        )
        
        # Second quantized conv + bn
        self.conv2 = qnn.QuantConv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
            weight_quant=Int8WeightPerTensorFloat,
            weight_bit_width=weight_bit_width
        )
        self.bn2 = nn.BatchNorm2d(planes)
        
        # Activation after addition
        self.act2 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            bit_width=act_bit_width,
            return_quant_tensor=True
        )

        # QuantIdentity adder for skip connections
        self.adder = QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True)
        self.identity = nn.Identity()

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                qnn.QuantConv2d(
                    in_planes, planes, kernel_size=1, stride=stride, bias=False,
                    weight_quant=Int8WeightPerTensorFloat,
                    weight_bit_width=weight_bit_width
                ),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Handle skip connection with explicit quantization
        shortcut_out = self.shortcut(x)
        if isinstance(out, QuantTensor):
            out = self.adder(out)
        if isinstance(shortcut_out, QuantTensor):
            shortcut_out = self.adder(shortcut_out)
        else:
            shortcut_out = self.identity(shortcut_out)
            
        out = out + shortcut_out
        out = self.act2(out)
        return out


class QuantBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, weight_bit_width=8, act_bit_width=8):
        super(QuantBottleneck, self).__init__()
        
        # First quantized conv + bn + activation
        self.conv1 = qnn.QuantConv2d(
            in_planes, planes, kernel_size=1, bias=False,
            weight_quant=Int8WeightPerTensorFloat,
            weight_bit_width=weight_bit_width
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            bit_width=act_bit_width,
            return_quant_tensor=True
        )
        
        # Second quantized conv + bn + activation
        self.conv2 = qnn.QuantConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
            weight_quant=Int8WeightPerTensorFloat,
            weight_bit_width=weight_bit_width
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            bit_width=act_bit_width,
            return_quant_tensor=True
        )
        
        # Third quantized conv + bn
        self.conv3 = qnn.QuantConv2d(
            planes, self.expansion*planes, kernel_size=1, bias=False,
            weight_quant=Int8WeightPerTensorFloat,
            weight_bit_width=weight_bit_width
        )
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        
        # Activation after addition
        self.act3 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            bit_width=act_bit_width,
            return_quant_tensor=True
        )

        # QuantIdentity adder for skip connections
        self.adder = QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True)
        self.identity = nn.Identity()

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                qnn.QuantConv2d(
                    in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False,
                    weight_quant=Int8WeightPerTensorFloat,
                    weight_bit_width=weight_bit_width
                ),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # Handle skip connection with explicit quantization
        shortcut_out = self.shortcut(x)
        if isinstance(out, QuantTensor):
            out = self.adder(out)
        if isinstance(shortcut_out, QuantTensor):
            shortcut_out = self.adder(shortcut_out)
        else:
            shortcut_out = self.identity(shortcut_out)
            
        out = out + shortcut_out
        out = self.act3(out)
        return out


class QuantResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, weight_bit_width=8, act_bit_width=8):
        super(QuantResNet, self).__init__()
        self.in_planes = 64
        self.weight_bit_width = weight_bit_width
        self.act_bit_width = act_bit_width
        
        # Initialize final pool here so it's not recreated in every forward pass
        self.final_pool = qnn.TruncAvgPool2d(
            kernel_size=4,
            bit_width=act_bit_width,
            trunc_quant=TruncTo8bit,
            float_to_int_impl_type="ROUND"
        )

        # First quantized conv + bn + activation
        self.conv1 = qnn.QuantConv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False,
            weight_quant=Int8WeightPerTensorFloat,
            weight_bit_width=weight_bit_width
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            bit_width=act_bit_width,
            return_quant_tensor=True
        )
        
        # ResNet blocks
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Final linear layer
        self.linear = qnn.QuantLinear(
            512*block.expansion, num_classes, bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int16Bias,
            weight_bit_width=weight_bit_width
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, 
                                weight_bit_width=self.weight_bit_width,
                                act_bit_width=self.act_bit_width))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Use the pre-initialized TruncAvgPool2d with TruncTo8bit
        out = self.final_pool(out)
        
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def QuantResNet18(num_classes=10, weight_bit_width=8, act_bit_width=8):
    return QuantResNet(QuantBasicBlock, [2, 2, 2, 2], num_classes, weight_bit_width, act_bit_width)


def QuantResNet50(num_classes=10, weight_bit_width=8, act_bit_width=8):
    return QuantResNet(QuantBottleneck, [3, 4, 6, 3], num_classes, weight_bit_width, act_bit_width)