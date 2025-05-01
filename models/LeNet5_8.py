import torch
import torch.nn as nn
import torch.nn.functional as F

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int16Bias
import argparse
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
import os
class QuantLeNet5(nn.Module):
    def __init__(self, num_classes=10, weight_bit_width=8, act_bit_width=8):
        super(QuantLeNet5, self).__init__()
        
        # First conv layer
        self.conv1 = qnn.QuantConv2d(
            1, 6, kernel_size=5, stride=1, padding=0, bias=False,
            weight_quant=Int8WeightPerTensorFloat
        )
        self.bn1 = nn.BatchNorm2d(6)
        self.act1 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )
        
        # Second conv layer
        self.conv2 = qnn.QuantConv2d(
            6, 16, kernel_size=5, stride=1, padding=0, bias=False,
            weight_quant=Int8WeightPerTensorFloat
        )
        self.bn2 = nn.BatchNorm2d(16)
        self.act2 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )
        
        # Fully connected layers
        self.fc1 = qnn.QuantLinear(
            16 * 4 * 4, 120, bias=True, 
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int16Bias
        )
        self.bn3 = nn.BatchNorm1d(120)
        self.act3 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )
        
        self.fc2 = qnn.QuantLinear(
            120, 84, bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int16Bias
        )
        self.bn4 = nn.BatchNorm1d(84)
        self.act4 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )
        
        self.fc3 = qnn.QuantLinear(
            84, num_classes, bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int16Bias
        )

    def forward(self, x):
        # First conv block
        out = self.act1(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = self.act2(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = torch.flatten(out, 1)
        out = self.act3(self.bn3(self.fc1(out)))
        out = self.act4(self.bn4(self.fc2(out)))
        out = self.fc3(out)
        
        return out
    
# parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
# parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true',
#                     help='resume from checkpoint')
# args = parser.parse_args()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# best_acc = 0  # best test accuracy
# start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# # Data
# print('==> Preparing data..')
# transform_train = transforms.Compose([
#     transforms.RandomCrop(28, padding=4),
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,)),
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,)),
# ])

# trainset = torchvision.datasets.MNIST(
#     root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=128, shuffle=True, num_workers=2)

# testset = torchvision.datasets.MNIST(
#     root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=100, shuffle=False, num_workers=2)

# # Model
# print('==> Building model..')
# net = QuantLeNet5()
# net.load_state_dict(torch.load('models/weights/LeNet5-8/lenet_ckpt.pth'))
# net = net.to(device)


# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


# # Simple progress bar function
# def progress_bar(batch_idx, total, message):
#     bar_length = 20
#     progress = float(batch_idx) / float(total)
#     bar = '=' * int(bar_length * progress) + '-' * (bar_length - int(bar_length * progress))
#     print(f'\r[{bar}] {int(100 * progress)}% | {message}', end='')


# # Training
# def train(epoch):
#     print('\nEpoch: %d' % epoch)
#     net.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         print(inputs[0].shape)
#         inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()

#         progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                      % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


# def test(epoch):
#     global best_acc
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)

#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#             progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                          % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

#     # Save checkpoint.
#     acc = 100.*correct/total
#     if acc > best_acc:
#         print('\nSaving..')
#         state = net.state_dict()
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, './checkpoint/lenet_ckpt.pth')
#         best_acc = acc


# for epoch in range(start_epoch, start_epoch+1):
#     # train(epoch)
#     test(epoch)
#     # scheduler.step()