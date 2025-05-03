import torch
import brevitas.nn as qnn
import time
from finetune import evaluate, finetune, get_dataloaders_mnist
from torch_pruning.utils import utils
from torch_pruning.pruner import importance
from torch_pruning.pruner.algorithms import base_pruner
from torchvision import datasets, transforms
from models import *



def save_pruned_model(model, path, pruning_history=None):
    """Save a pruned model including its structure and pruning history"""
    save_dict = {
        'state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'pruning_history': pruning_history
    }
    torch.save(save_dict, path)
def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return train_dataset
pruning_config = {
    'model': QuantLeNet5(),
    'state_dict': 'models/weights/LeNet5-8/lenet_ckpt.pth',
    'example_inputs': torch.randn(1, 1, 28, 28),
    'target_pruning_ratio': 0.9,
    'iterative_steps': 15,
    'epochs': 10,
    'round_to': 8,
    'quant': True
}
# pruning_config = {
#     'model': LeNet5(),
#     'state_dict': 'models/weights/LeNet5/LeNet5_new.pth',
#     'example_inputs': torch.randn(1, 1, 28, 28),
#     'target_pruning_ratio': 0.9,
#     'iterative_steps': 15,
#     'epochs': 10,
#     'round_to': 8,
#     'quant': False
# }


def iterative_pruning():
    device = torch.device("cpu")
    output_log = []
    # Get data
    trainloader, testloader = get_dataloaders_mnist(batch_size=32)
    
    # Load model (modify for CIFAR-10 - changing first conv and classifier)
    model = pruning_config['model']
    state_dict = torch.load(pruning_config['state_dict'])
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    example_inputs = pruning_config['example_inputs'].to(device)
    base_macs, base_params = utils.count_ops_and_params(model, example_inputs)
    print(f"Original model: {base_macs} MACs, {base_params} parameters")
    
    # imp = importance.GroupMagnitudeImportance(p=1)
    imp = importance.GroupActivationImportance(
       model=model,             # Must be the same model you'll prune
       dataset= load_mnist_data(),         # Must provide (input, label) tuples
       num_classes=10,          # Must match your dataset's classes
       batch_size=32,           # Smaller may avoid memory issues
       num_samples=32,          # Fewer samples = faster but less accurate
       critical_percentile=10   # Threshold for critical neurons
   )
    ignored_layers = []
    for m in model.modules():
        if (isinstance(m, torch.nn.Linear) or isinstance(m, qnn.QuantLinear)) and m.out_features == 10:
            ignored_layers.append(m)
    
    pruner = base_pruner.BasePruner(
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=pruning_config['target_pruning_ratio'],
        ignored_layers=ignored_layers,
        iterative_steps=pruning_config['iterative_steps'],
        round_to=pruning_config['round_to'],  
    )
    
    results = []
    for i in range(pruning_config['iterative_steps']):
        print(f"\n{'='*50}\nPruning Step {i+1}/{pruning_config['iterative_steps']}")
        
        pruner.step()
            

        current_ratio = pruner.per_step_pruning_ratio[pruner.current_step-1]
        macs, params = utils.count_ops_and_params(model, example_inputs)
        compression_ratio = base_params / params
        print(f"Pruned model: {macs} MACs, {params} parameters")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        
        acc_before = evaluate(model, testloader, device)
        print(f"Accuracy before finetuning: {acc_before:.2f}%")
        
        print("Finetuning...")
        start_time = time.time()
        best_acc = finetune(model, trainloader, testloader, device, pruning_config['epochs'])
        finetune_time = time.time() - start_time
        
        results.append({
            'step': i+1,
            'pruning_ratio': current_ratio,
            'params': params,
            'macs': macs,
            'compression_ratio': compression_ratio,
            'accuracy_before_finetuning': acc_before,
            'accuracy_after_finetuning': best_acc,
            'finetune_time': finetune_time
        })
        if pruning_config['quant']:
            # Inside your iterative_pruning function
            pruning_history = pruner.pruning_history()
            save_pruned_model(model, f'pruned_model_step_{i+1}.pth', pruning_history)
        else:
            torch.save(model, f'pruned_model_step_{i+1}.pth')
    

    output_log.append("\n" + "="*50 + "\n")
    output_log.append("Iterative Pruning Summary:\n")
    output_log.append("="*50 + "\n")
    header = f"{'Step':^5}|{'Pruning Ratio':^15}|{'Params'}|{'MACs'}|{'Acc Before':^12}|{'Acc After':^12}|{'Compression':^12}\n"
    separator = "-"*80 + "\n"
    output_log.append(header)
    output_log.append(separator)
    
    for r in results:
        line = f"{r['step']:^5}|{r['pruning_ratio']:^15.2f}|{r['params']}|{r['macs']}|{r['accuracy_before_finetuning']:^12.2f}|{r['accuracy_after_finetuning']:^12.2f}|{r['compression_ratio']:^12.2f}x\n"
        output_log.append(line)
    

    with open("pruning_summary.txt", "w") as f:
        f.writelines(output_log)

if __name__ == "__main__":
    iterative_pruning()