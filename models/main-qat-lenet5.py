import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse

# Import the quantized LeNet-5 model
from LeNet5_8 import QuantLeNet5

parser = argparse.ArgumentParser(description='PyTorch MNIST QAT Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--weight_bit_width', default=8, type=int, help='bit width for weights')
parser.add_argument('--act_bit_width', default=8, type=int, help='bit width for activations')
parser.add_argument('--pretrained_path', required=True, type=str, help='path to pre-trained model')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs for QAT')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy

# Data preparation for MNIST
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# Model creation
print('==> Building quantized LeNet-5 model..')
net = QuantLeNet5(num_classes=10, weight_bit_width=args.weight_bit_width, act_bit_width=args.act_bit_width)

def map_weights_from_pytorch_to_brevitas(brevitas_model, pytorch_state_dict):
    brevitas_state_dict = brevitas_model.state_dict()
    mapped_state_dict = {}

    for k in brevitas_state_dict.keys():
        if 'act_quant' in k or 'weight_quant' in k:
            continue

        # For LeNet-5, handle the case where FP32 model has '.weight' but brevitas has '.weight.tensor'
        pytorch_key = k
        if '.weight.tensor' in k:
            pytorch_key = k.replace('.weight.tensor', '.weight')
        if '.bias.tensor' in k:
            pytorch_key = k.replace('.bias.tensor', '.bias')

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

# Load pretrained FP32 model
print('==> Loading pretrained FP32 weights..')
checkpoint = torch.load(args.pretrained_path, map_location=device)
if 'net' in checkpoint:
    pytorch_state_dict = checkpoint['net']
elif 'state_dict' in checkpoint:
    pytorch_state_dict = checkpoint['state_dict']
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

    # Save checkpoint.onn
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
        torch.save(state, f'./checkpoint/quantized_lenet5_w{args.weight_bit_width}_a{args.act_bit_width}.pth')
        best_acc = acc


# Start training
for epoch in range(args.epochs):
    train(epoch)
    test(epoch)
    scheduler.step()

print(f"Best accuracy: {best_acc:.2f}%")

