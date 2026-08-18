import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import load_config, set_seed
from dataset import KLADataset
from baseline_model import BaselineRestorationNet, get_model_info

def main():
    # Load configuration
    config_path = "configs/baseline.yaml"
    config = load_config(config_path)
    
    # Set seed
    set_seed(config["seed"])
    
    # Device detection
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"CUDA available: {'YES' if cuda_available else 'NO'}")
    print(f"Device: {device}")
    if cuda_available:
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        
    # Initialize datasets
    print("\nInitializing datasets...")
    train_dataset = KLADataset(
        dataset_root=config["dataset_root"],
        split="train",
        csv_path=config["train_split_path"]
    )
    val_dataset = KLADataset(
        dataset_root=config["dataset_root"],
        split="train",
        csv_path=config["val_split_path"]
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=cuda_available
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=cuda_available
    )
    
    # Initialize model
    print("\nInitializing model...")
    model_cfg = config.get("model", {})
    model = BaselineRestorationNet(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        num_features=model_cfg.get("num_features", 64),
        num_blocks=model_cfg.get("num_blocks", 4)
    )
    model.to(device)
    
    # Print model parameters
    tot_params, train_params, model_size = get_model_info(model)
    print(f"Total parameters: {tot_params:,}")
    print(f"Trainable parameters: {train_params:,}")
    print(f"Model size: {model_size:.3f} MB")
    
    # Set Loss and Optimizer
    loss_type = config.get("loss_type", "L1")
    if loss_type == "L1":
        criterion = nn.L1Loss()
    elif loss_type == "L2":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    print(f"Using loss function: {loss_type}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Create checkpoints dir
    os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)
    
    # Training Loop
    epochs = config["epochs"]
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    
    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        for batch in train_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_train_loss / len(train_dataset)
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = running_val_loss / len(val_dataset)
        
        # Save history
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        
        # Checkpoint if best
        is_best = epoch_val_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_val_loss
            # Save checkpoint dictionary containing training state as well
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": epoch_val_loss,
                "config": config
            }
            torch.save(checkpoint, config["checkpoint_path"])
            
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.6f} | "
            f"Val Loss: {epoch_val_loss:.6f} | "
            f"Best Val Loss: {best_val_loss:.6f}"
            + (" (Saved Best Checkpoint)" if is_best else "")
        )
        
    # Save training history to file
    os.makedirs("outputs/baseline", exist_ok=True)
    history_path = "outputs/baseline/loss_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"\nTraining complete. Loss history saved to: {history_path}")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"Checkpoint saved to: {config['checkpoint_path']}")

if __name__ == "__main__":
    main()
