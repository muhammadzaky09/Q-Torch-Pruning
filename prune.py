import torch
import time
from finetune import evaluate, finetune, get_dataloaders
from torch_pruning.utils import utils
from torch_pruning.pruner import importance
from torch_pruning.pruner.algorithms import base_pruner
from models import *

def save_pruned_model(model, path, pruning_history=None):
    """Save a pruned model including its structure and pruning history"""
    save_dict = {
        'state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'pruning_history': pruning_history
    }
    torch.save(save_dict, path)
    
pruning_config = {
    'model': QuantResNet50(),
    'state_dict': 'models/weights/resnet50_w8a8.pth',
    'example_inputs': torch.randn(1, 3, 32, 32),
    'target_pruning_ratio': 0.9,
    'iterative_steps': 15,
    'epochs': 10,
    'round_to': 8,
    'quant': True
}

def iterative_pruning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_log = []
    # Get data
    trainloader, testloader = get_dataloaders(batch_size=32)
    
    # Load model (modify for CIFAR-10 - changing first conv and classifier)
    model = pruning_config['model']
    state_dict = torch.load(pruning_config['state_dict'])
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    example_inputs = pruning_config['example_inputs'].to(device)
    base_macs, base_params = utils.count_ops_and_params(model, example_inputs)
    print(f"Original model: {base_macs/1e9:.2f} GMACs, {base_params/1e6:.2f}M parameters")
    
    imp = importance.GroupMagnitudeImportance(p=2)
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 10:
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
        print(f"Current pruning ratio: {current_ratio:.2f}")
        
        macs, params = utils.count_ops_and_params(model, example_inputs)
        compression_ratio = base_params / params
        print(f"Pruned model: {macs/1e9:.2f} GMACs, {params/1e6:.2f}M parameters")
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
            torch.save(model.state_dict(), f'pruned_model_step_{i+1}.pth')
    

    output_log.append("\n" + "="*50 + "\n")
    output_log.append("Iterative Pruning Summary:\n")
    output_log.append("="*50 + "\n")
    header = f"{'Step':^5}|{'Pruning Ratio':^15}|{'Params (M)':^12}|{'MACs (G)':^10}|{'Acc Before':^12}|{'Acc After':^12}|{'Compression':^12}\n"
    separator = "-"*80 + "\n"
    output_log.append(header)
    output_log.append(separator)
    
    for r in results:
        line = f"{r['step']:^5}|{r['pruning_ratio']:^15.2f}|{r['params']/1e6:^12.2f}|{r['macs']/1e9:^10.2f}|{r['accuracy_before_finetuning']:^12.2f}|{r['accuracy_after_finetuning']:^12.2f}|{r['compression_ratio']:^12.2f}x\n"
        output_log.append(line)
    

    with open("pruning_summary.txt", "w") as f:
        f.writelines(output_log)

if __name__ == "__main__":
    iterative_pruning()