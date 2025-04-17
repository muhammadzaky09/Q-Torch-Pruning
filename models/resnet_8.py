'''ResNet with Brevitas quantization support.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int16Bias



class QuantBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, weight_bit_width=8, act_bit_width=8):
        super(QuantBasicBlock, self).__init__()
        
        # First quantized conv + bn + activation
        self.conv1 = qnn.QuantConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
            weight_quant=Int8WeightPerTensorFloat
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )
        
        # Second quantized conv + bn
        self.conv2 = qnn.QuantConv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
            weight_quant=Int8WeightPerTensorFloat
        )
        self.bn2 = nn.BatchNorm2d(planes)
        
        # Activation after addition
        self.act2 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                qnn.QuantConv2d(
                    in_planes, planes, kernel_size=1, stride=stride, bias=False,
                    weight_quant=Int8WeightPerTensorFloat
                ),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class QuantBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, weight_bit_width=8, act_bit_width=8):
        super(QuantBottleneck, self).__init__()
        
        # First quantized conv + bn + activation
        self.conv1 = qnn.QuantConv2d(
            in_planes, planes, kernel_size=1, bias=False,
            weight_quant=Int8WeightPerTensorFloat
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )
        
        # Second quantized conv + bn + activation
        self.conv2 = qnn.QuantConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
            weight_quant=Int8WeightPerTensorFloat
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )
        
        # Third quantized conv + bn
        self.conv3 = qnn.QuantConv2d(
            planes, self.expansion*planes, kernel_size=1, bias=False,
            weight_quant=Int8WeightPerTensorFloat
        )
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        
        # Activation after addition
        self.act3 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                qnn.QuantConv2d(
                    in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False,
                    weight_quant=Int8WeightPerTensorFloat
                ),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.act3(out)
        return out


class QuantResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, weight_bit_width=8, act_bit_width=8):
        super(QuantResNet, self).__init__()
        self.in_planes = 64

        # First quantized conv + bn + activation
        self.conv1 = qnn.QuantConv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False,
            weight_quant=Int8WeightPerTensorFloat
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
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
            bias_quant=Int16Bias
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # Provide all required arguments to avg_pool2d for Brevitas compatibility
        out = F.avg_pool2d(
            out, 
            kernel_size=4, 
            stride=4, 
            padding=0, 
            ceil_mode=False, 
            count_include_pad=True, 
            divisor_override=None
        )
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def QuantResNet18(num_classes=10, weight_bit_width=8, act_bit_width=8):
    return QuantResNet(QuantBasicBlock, [2, 2, 2, 2], num_classes, weight_bit_width, act_bit_width)


def QuantResNet50(num_classes=10, weight_bit_width=8, act_bit_width=8):
    return QuantResNet(QuantBottleneck, [3, 4, 6, 3], num_classes, weight_bit_width, act_bit_width)