import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import load_config, set_seed
from dataset import KLADataset
from baseline_model import BaselineRestorationNet, get_model_info

def main():
    # Verify CUDA availability first
    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL ERROR: CUDA is not available! Stopping execution to prevent fallback to CPU.")
        
    # Load configuration
    config_path = "configs/baseline.yaml"
    config = load_config(config_path)
    
    # Set seed
    set_seed(config["seed"])
    
    # Device detection and startup printing
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Device: {device}")
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    
    # Setup GPU baseline directories
    checkpoint_dir = "outputs/baseline_gpu/checkpoints"
    metrics_dir = "outputs/baseline_gpu/metrics"
    logs_dir = "outputs/baseline_gpu/logs"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    log_file_path = os.path.join(logs_dir, "training.log")
    log_file = open(log_file_path, "w")
    
    def log_print(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
        
    log_print(f"Device: {device}")
    log_print(f"GPU: {gpu_name}")
    log_print(f"PyTorch: {torch.__version__}")
    log_print(f"CUDA: {torch.version.cuda}")
    
    # Initialize datasets
    log_print("\nInitializing datasets...")
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
    
    log_print(f"Train samples: {len(train_dataset)}")
    log_print(f"Validation samples: {len(val_dataset)}")
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    
    # Model
    log_print("\nInitializing model...")
    model_cfg = config.get("model", {})
    model = BaselineRestorationNet(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        num_features=model_cfg.get("num_features", 64),
        num_blocks=model_cfg.get("num_blocks", 4)
    )
    model.to(device)
    log_print("Model moved to selected device.")
    
    # Print model params
    tot_params, train_params, model_size = get_model_info(model)
    log_print(f"Total parameters: {tot_params:,}")
    log_print(f"Trainable parameters: {train_params:,}")
    log_print(f"Model size: {model_size:.3f} MB")
    
    # Loss and optimizer
    loss_type = config.get("loss_type", "L1")
    if loss_type == "L1":
        criterion = nn.L1Loss()
    elif loss_type == "L2":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    log_print(f"Using loss function: {loss_type}")
    
    learning_rate = config["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    epochs = config["epochs"]
    best_val_loss = float("inf")
    best_epoch = 0
    history = {
        "train_loss": [], 
        "val_loss": [], 
        "lr": [], 
        "gpu_memory_mb": [],
        "epoch_time_sec": []
    }
    
    # Reset peak memory tracking
    torch.cuda.reset_peak_memory_stats(device)
    
    log_print(f"\nStarting GPU Baseline Training for {epochs} epochs...")
    start_train_time = time.perf_counter()
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.perf_counter()
        
        # Verify CUDA is still available
        if not torch.cuda.is_available():
            raise RuntimeError("CRITICAL ERROR: CUDA became unavailable during training!")
            
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
        
        # GPU Memory usage (in MB)
        mem_allocated = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        epoch_elapsed = time.perf_counter() - epoch_start_time
        
        # Record stats
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["lr"].append(learning_rate)
        history["gpu_memory_mb"].append(mem_allocated)
        history["epoch_time_sec"].append(epoch_elapsed)
        
        # Checkpoint if best
        is_best = epoch_val_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": epoch_val_loss,
                "config": config
            }
            best_path = os.path.join(checkpoint_dir, "baseline_gpu_best.pth")
            torch.save(checkpoint, best_path)
            
        log_print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.6f} | "
            f"Val Loss: {epoch_val_loss:.6f} | "
            f"Learning Rate: {learning_rate:.6f} | "
            f"GPU Memory: {mem_allocated:.2f} MB | "
            f"Time: {epoch_elapsed:.1f}s"
            + (" (Saved Best)" if is_best else "")
        )
        
    total_train_time = time.perf_counter() - start_train_time
    
    # Save final checkpoint
    final_checkpoint = {
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": epoch_val_loss,
        "config": config
    }
    final_path = os.path.join(checkpoint_dir, "baseline_gpu_final.pth")
    torch.save(final_checkpoint, final_path)
    
    # Save history
    history_path = os.path.join(metrics_dir, "loss_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    # Save metadata summary
    summary = {
        "train_time_seconds": total_train_time,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_train_loss": epoch_train_loss,
        "final_val_loss": epoch_val_loss
    }
    summary_path = os.path.join(metrics_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
        
    log_print(f"\nTraining completed in {total_train_time:.1f} seconds ({total_train_time/60:.2f} minutes).")
    log_print(f"Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.6f}")
    log_print(f"Final Train Loss: {epoch_train_loss:.6f} | Final Val Loss: {epoch_val_loss:.6f}")
    
    log_file.close()

if __name__ == "__main__":
    main()
