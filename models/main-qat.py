import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import re

# Import the quantized ResNet models
from resnet_8 import QuantResNet18, QuantResNet50
from brevitas.export import export_onnx_qcdq

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 QAT Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default='resnet18', type=str, help='model type (resnet18 or resnet50)')
parser.add_argument('--weight_bit_width', default=8, type=int, help='bit width for weights')
parser.add_argument('--act_bit_width', default=8, type=int, help='bit width for activations')
parser.add_argument('--pretrained_path', required=True, type=str, help='path to pre-trained model')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs for QAT')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy

# Data preparation
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# Model creation
print('==> Building quantized model..')
if args.model == 'resnet18':
    net = QuantResNet18(num_classes=10, weight_bit_width=args.weight_bit_width, act_bit_width=args.act_bit_width)
elif args.model == 'resnet50':
    net = QuantResNet50(num_classes=10, weight_bit_width=args.weight_bit_width, act_bit_width=args.act_bit_width)
else:
    raise ValueError(f"Unsupported model type: {args.model}")

def map_weights_from_pytorch_to_brevitas(brevitas_model, pytorch_state_dict):
    brevitas_state_dict = brevitas_model.state_dict()
    mapped_state_dict = {}

    for k in brevitas_state_dict.keys():
        if 'act_quant' in k or 'weight_quant' in k:
            continue

        pytorch_key = k

        if pytorch_key in pytorch_state_dict:
            if brevitas_state_dict[k].shape == pytorch_state_dict[pytorch_key].shape:
                mapped_state_dict[k] = pytorch_state_dict[pytorch_key]
            else:
                print(f"Shape mismatch: {k} ({brevitas_state_dict[k].shape}) vs {pytorch_key} ({pytorch_state_dict[pytorch_key].shape})")
        else:
            print(f"Key not found in pretrained model: {pytorch_key}")

    for k in brevitas_state_dict.keys():
        if k not in mapped_state_dict:
            mapped_state_dict[k] = brevitas_state_dict[k]

    return mapped_state_dict

checkpoint = torch.load(args.pretrained_path, map_location=device)
if 'net' in checkpoint:
    pytorch_state_dict = checkpoint['net']
else:
    pytorch_state_dict = checkpoint

try:

    mapped_state_dict = map_weights_from_pytorch_to_brevitas(net, pytorch_state_dict)
    

    missing_keys, unexpected_keys = net.load_state_dict(mapped_state_dict, strict=False)
    
    if missing_keys:
        print(f"First few missing keys: {missing_keys[:5]}")
    if unexpected_keys:
        print(f"First few unexpected keys: {unexpected_keys[:5]}")
except Exception as e:
    print(f"Error loading pretrained weights: {e}")
    print("Starting with random initialization for the quantized model")

# Move model to device
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# Training function
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print('Train Batch: %d/%d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                 % (batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# Testing function
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                print('Test Batch: %d/%d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (batch_idx, len(testloader), test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print(f'Accuracy: {acc:.2f}%')
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict() if not isinstance(net, torch.nn.DataParallel) else net.module.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/quantized_{args.model}_w{args.weight_bit_width}_a{args.act_bit_width}.pth')
        best_acc = acc
        
        

# Start training
for epoch in range(args.epochs):
    train(epoch)
    test(epoch)
    scheduler.step()

print(f"Best accuracy: {best_acc:.2f}%")