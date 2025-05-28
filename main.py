import argparse
import os
import torch
from torch_geometric.loader import DataLoader
from src.loadData import GraphDataset
from src.utils import set_seed
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from src.losses import GCOD_loss
from src.models import GNN 
import optuna

# Set the random seed
set_seed()

def add_zeros(data):
    # Create tensor on CPU initially
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def get_data_loader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,  # Reduce number of workers
        pin_memory=True,  # More efficient GPU transfer
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2  # Reduce prefetching
    )

def train(data_loader, model, optimizer, criterion, device, save_checkpoints, checkpoint_path, current_epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    # Clear memory after each batch
    for data in tqdm(data_loader, desc="Iterating training graphs", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        
        # Move predictions to CPU and convert to numpy immediately
        pred = output.argmax(dim=1).cpu()
        correct += (pred == data.y.cpu()).sum().item()
        total += data.y.size(0)
        total_loss += loss.item()
        
        # Clear memory
        del data, output, loss, pred
        torch.cuda.empty_cache()

    # Save checkpoints if required
    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    return total_loss / len(data_loader),  correct / total



def evaluate(data_loader, model, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    total_loss = 0
    criterion = GCOD_loss()
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            # Move all data attributes to the correct device
            data = data.to(device)
            # Ensure x is on the same device as model
            if hasattr(data, 'x'):
                data.x = data.x.to(device)
            
            output = model(data)
            pred = output.argmax(dim=1)
            
            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                total_loss += criterion(output, data.y).item()
            else:
                predictions.extend(pred.cpu().numpy())
            
            # Clear memory after each batch
            torch.cuda.empty_cache()
            
    if calculate_accuracy:
        accuracy = correct / total
        return total_loss / len(data_loader), accuracy
    return predictions

def save_predictions(predictions, test_path):
    script_dir = os.getcwd() 
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path))
    
    os.makedirs(submission_folder, exist_ok=True)
    
    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")
    
    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })
    
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")


def plot_training_progress(train_losses, train_accuracies, output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch')

    # Save plots in the current directory
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"))
    plt.close()

def objective(trial, args, train_loader, val_loader, device):
    # Reduce parameter search space
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)  # Narrower range
    dropout = trial.suggest_float("dropout", 0.2, 0.5)     # Narrower range
    num_layers = trial.suggest_int("num_layers", 2, 6)     # Fewer layers
    emb_dim = trial.suggest_categorical("emb_dim", [64, 128, 256])  # Smaller dimensions
    
    # Create model with smaller capacity
    model = GNN(
        gnn_type=args.gnn,
        num_class=6,
        num_layer=num_layers,
        emb_dim=emb_dim,
        drop_ratio=dropout,
        virtual_node='virtual' in args.gnn
    ).to(device)
    
    # Clear memory before training
    torch.cuda.empty_cache()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = GCOD_loss()
    
    # Training loop
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train(
            train_loader, model, optimizer, criterion, device,
            save_checkpoints=False, checkpoint_path=None, current_epoch=epoch
        )
        
        val_loss, val_acc = evaluate(val_loader, model, device, calculate_accuracy=True)
        
        # Report intermediate value
        trial.report(val_acc, epoch)
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    return best_val_acc

# Add this function to check and fix model attributes
def ensure_model_attributes(model):
    """Ensure all required attributes are present in the model."""
    if not hasattr(model, 'node_encoder'):
        print("Initializing missing node_encoder")
        model.node_encoder = torch.nn.Embedding(1, model.emb_dim)
    return model

# Modify your main function to include Optuna study
def main(args):
    print("\n=== Starting Main Execution ===")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Model type: {args.gnn}")
    
    script_dir = os.getcwd() 
    
    # Device setup with proper error handling
    if torch.cuda.is_available():
        n_cuda_devices = torch.cuda.device_count()
        if args.device >= n_cuda_devices:
            print(f"Warning: Selected GPU {args.device} not available. Using GPU 0 instead.")
            device = torch.device("cuda:0")
        else:
            device = torch.device(f"cuda:{args.device}")
    else:
        print("Warning: No GPU available. Using CPU instead.")
        device = torch.device("cpu")
    
    print(f"Device: {device}")
    num_checkpoints = args.num_checkpoints if args.num_checkpoints else 3
    
    print("\n=== Model Configuration ===")
    print(f"Number of layers: {args.num_layer}")
    print(f"Embedding dimension: {args.emb_dim}")
    print(f"Dropout ratio: {args.drop_ratio}")
    print(f"Batch size: {args.batch_size}")
    
    # Model initialization
    print("\n=== Initializing Model ===")
    if args.gnn == 'gin':
        model = GNN(gnn_type='gin', num_class=6, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type='gin', num_class=6, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type='gcn', num_class=6, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type='gcn', num_class=6, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=True).to(device)
    else:
        raise ValueError('Invalid GNN type')
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Setting up directories
    print("\n=== Setting up Directories ===")
    test_dir_name = os.path.basename(os.path.dirname(args.test_path))
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
    
    os.makedirs(logs_folder, exist_ok=True)
    os.makedirs(checkpoints_folder, exist_ok=True)
    print(f"Logs directory: {logs_folder}")
    print(f"Checkpoints directory: {checkpoints_folder}")
    
    # Load pre-trained model or prepare for training
    if os.path.exists(checkpoint_path):
        print("\n=== Loading Pre-trained Model ===")
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model = ensure_model_attributes(model)
            if check_model_compatibility(state_dict, model):
                model.load_state_dict(state_dict)
                print("Model loaded successfully with matching architecture")
            else:
                print("Warning: Model architecture mismatch. Training new model.")
                model = ensure_model_attributes(model)
        except Exception as e:
            print(f"Warning: Error loading model: {e}")
            print("Creating new model instance...")
            model = GNN(
                gnn_type=args.gnn,
                num_class=6,
                num_layer=args.num_layer,
                emb_dim=args.emb_dim,
                drop_ratio=args.drop_ratio,
                virtual_node='virtual' in args.gnn
            ).to(device)
            model = ensure_model_attributes(model)
    
    # Prepare test dataset
    print("\n=== Preparing Test Dataset ===")
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Test dataset size: {len(test_dataset)}")
    
    if args.train_path:
        print("\n=== Training Mode ===")
        print("Preparing training and validation datasets...")
        full_dataset = GraphDataset(args.train_path, transform=add_zeros)
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        print(f"Full dataset size: {len(full_dataset)}")
        print(f"Training size: {train_size}")
        print(f"Validation size: {val_size}")
        
        if args.optimize:
            print("\n=== Starting Hyperparameter Optimization ===")
            print(f"Number of trials: {args.n_trials}")
            print(f"Timeout: {args.timeout} seconds")
            
            # ... rest of the optimization code ...
            
        print("\n=== Starting Training ===")
        print(f"Number of epochs: {args.epochs}")
        print(f"Number of checkpoints: {num_checkpoints}")
        
        # ... rest of the training code ...
        
    print("\n=== Generating Predictions ===")
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        # Ensure model and its attributes are on the correct device
        model = model.to(device)
        model = ensure_model_attributes(model)
        if hasattr(model, 'node_encoder'):
            model.node_encoder = model.node_encoder.to(device)
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully for predictions")
    except Exception as e:
        print(f"Warning: Error loading model for predictions: {e}")
        print("Using current model state")
        model = ensure_model_attributes(model)
        model = model.to(device)
    
    predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
    save_predictions(predictions, args.test_path)
    print("=== Execution Complete ===\n")

def check_model_compatibility(state_dict, current_model):
    """Check if a state dict is compatible with current model architecture."""
    model_keys = set(current_model.state_dict().keys())
    state_dict_keys = set(state_dict.keys())
    
    missing_keys = model_keys - state_dict_keys
    unexpected_keys = state_dict_keys - model_keys
    
    if missing_keys or unexpected_keys:
        print("\nModel Architecture Mismatch:")
        if missing_keys:
            print(f"Missing keys in state dict: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys in state dict: {len(unexpected_keys)}")
        return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--num_checkpoints", type=int, help="Number of checkpoints to save during training.")
    parser.add_argument('--device', type=int, default=1, help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin', help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300, help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--optimize', action='store_true', help='Use Optuna for hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--timeout', type=int, default=3600, help='Timeout for optimization in seconds')
    
    args = parser.parse_args()
    main(args)
