import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils import load_config, set_seed
from dataset import KLADataset
from echo_model import BaselineECHOModel, get_model_info
from echo_loss import ECHOLoss

def parse_args():
    parser = argparse.ArgumentParser(description="Train ECHO Prototype Model")
    parser.add_argument("--smoke-test", action="store_true", help="Run only a 2-epoch smoke test and exit")
    return parser.parse_known_args()[0]

def main():
    args = parse_args()
    
    # Verify CUDA availability first
    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL ERROR: CUDA is not available! Stopping execution to prevent fallback to CPU.")
        
    # Load configuration
    config_path = "configs/echo.yaml"
    config = load_config(config_path)
    
    # Set seed
    set_seed(config["seed"])
    
    # Device and print stats
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Device: {device}")
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    
    # Setup directories
    checkpoint_dir = "outputs/echo_phase4/checkpoints"
    metrics_dir = "outputs/echo_phase4/metrics"
    logs_dir = "outputs/echo_phase4/logs"
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
    
    # Initialize model
    log_print("\nInitializing model...")
    model_cfg = config.get("model", {})
    ablation_cfg = config.get("ablation", {})
    model = BaselineECHOModel(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        num_features=model_cfg.get("num_features", 64),
        num_blocks=model_cfg.get("num_blocks", 6),
        ablation=ablation_cfg
    )
    model.to(device)
    log_print("Model moved to selected device.")
    
    # Print model params
    tot_params, train_params, model_size = get_model_info(model)
    log_print(f"Total parameters: {tot_params:,}")
    log_print(f"Trainable parameters: {train_params:,}")
    log_print(f"Model size: {model_size:.3f} MB")
    
    # Verify forward pass and loss calculations
    log_print("\nRunning verification checks...")
    sample_batch = next(iter(train_loader))
    sample_in = sample_batch["input"].to(device)
    sample_tgt = sample_batch["target"].to(device)
    
    # Forward check
    with torch.no_grad():
        sample_out, sample_E = model(sample_in)
    log_print(f"Input shape: {list(sample_in.shape)}")
    log_print(f"Output shape: {list(sample_out.shape)}")
    log_print(f"Evidence map shape: {list(sample_E.shape)}")
    
    # Shape verification
    if list(sample_out.shape) != [sample_in.size(0), 1, 256, 256]:
        raise ValueError(f"CRITICAL ERROR: Output shape {list(sample_out.shape)} is incorrect! Expected [batch, 1, 256, 256].")
    log_print("Output shape verification: PASSED")
    
    # Loss check
    criterion = ECHOLoss(weights=config["loss_weights"])
    loss_val, loss_items = criterion(sample_out, sample_tgt)
    log_print(f"Composite loss calculation: PASSED (Initial loss: {loss_val.item():.6f})")
    
    # Gradient check
    model.train()
    sample_out, _ = model(sample_in)
    loss_val, _ = criterion(sample_out, sample_tgt)
    loss_val.backward()
    
    # Check for finite gradients
    finite_grads = True
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if not torch.isfinite(param.grad).all():
                finite_grads = False
                log_print(f"Gradient warning: {name} contains non-finite values.")
    
    if not finite_grads:
        raise ValueError("CRITICAL ERROR: Non-finite gradients detected during verification!")
    log_print("Gradient calculation verification: PASSED")
    
    # Optimizer and epochs
    epochs = 2 if args.smoke_test else config["epochs"]
    learning_rate = config["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float("inf")
    best_epoch = 0
    history = {
        "train_loss": [], 
        "train_loss_pixel": [],
        "train_loss_edge": [],
        "train_loss_ssim": [],
        "val_loss": [], 
        "val_loss_pixel": [],
        "val_loss_edge": [],
        "val_loss_ssim": [],
        "lr": [], 
        "gpu_memory_mb": [],
        "epoch_time_sec": []
    }
    
    torch.cuda.reset_peak_memory_stats(device)
    
    log_print(f"\nStarting ECHO Prototype Training for {epochs} epochs...")
    start_train_time = time.perf_counter()
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.perf_counter()
        
        # Verify CUDA is still available
        if not torch.cuda.is_available():
            raise RuntimeError("CRITICAL ERROR: CUDA became unavailable during training!")
            
        # Training Phase
        model.train()
        running_loss = 0.0
        running_pixel = 0.0
        running_edge = 0.0
        running_ssim = 0.0
        
        for batch in train_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss_val, items = criterion(outputs, targets)
            loss_val.backward()
            optimizer.step()
            
            running_loss += items["loss"] * inputs.size(0)
            running_pixel += items["loss_pixel"] * inputs.size(0)
            running_edge += items["loss_edge"] * inputs.size(0)
            running_ssim += items["loss_ssim"] * inputs.size(0)
            
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_pixel = running_pixel / len(train_dataset)
        epoch_train_edge = running_edge / len(train_dataset)
        epoch_train_ssim = running_ssim / len(train_dataset)
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        running_val_pixel = 0.0
        running_val_edge = 0.0
        running_val_ssim = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                
                outputs, _ = model(inputs)
                loss_val, items = criterion(outputs, targets)
                
                running_val_loss += items["loss"] * inputs.size(0)
                running_val_pixel += items["loss_pixel"] * inputs.size(0)
                running_val_edge += items["loss_edge"] * inputs.size(0)
                running_val_ssim += items["loss_ssim"] * inputs.size(0)
                
        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_pixel = running_val_pixel / len(val_dataset)
        epoch_val_edge = running_val_edge / len(val_dataset)
        epoch_val_ssim = running_val_ssim / len(val_dataset)
        
        # GPU stats
        mem_allocated = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        epoch_elapsed = time.perf_counter() - epoch_start_time
        
        # Save history
        history["train_loss"].append(epoch_train_loss)
        history["train_loss_pixel"].append(epoch_train_pixel)
        history["train_loss_edge"].append(epoch_train_edge)
        history["train_loss_ssim"].append(epoch_train_ssim)
        history["val_loss"].append(epoch_val_loss)
        history["val_loss_pixel"].append(epoch_val_pixel)
        history["val_loss_edge"].append(epoch_val_edge)
        history["val_loss_ssim"].append(epoch_val_ssim)
        history["lr"].append(learning_rate)
        history["gpu_memory_mb"].append(mem_allocated)
        history["epoch_time_sec"].append(epoch_elapsed)
        
        # Checkpoint if best (based on validation loss)
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
            best_path = os.path.join(checkpoint_dir, "echo_best.pth")
            torch.save(checkpoint, best_path)
            
        log_print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.6f} (L1: {epoch_train_pixel:.4f}, Edge: {epoch_train_edge:.4f}, SSIM: {epoch_train_ssim:.4f}) | "
            f"Val Loss: {epoch_val_loss:.6f} (L1: {epoch_val_pixel:.4f}, Edge: {epoch_val_edge:.4f}, SSIM: {epoch_val_ssim:.4f}) | "
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
    final_path = os.path.join(checkpoint_dir, "echo_final.pth")
    torch.save(final_checkpoint, final_path)
    
    # Save training curve plot
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss", color="blue")
    plt.plot(history["val_loss"], label="Validation Loss", color="red")
    plt.title("ECHO Prototype Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Composite)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/echo_phase4/training_curve.png", dpi=150)
    plt.close()
    
    # Save history stats
    with open(os.path.join(metrics_dir, "loss_history.json"), "w") as f:
        json.dump(history, f, indent=4)
        
    # Save training config reproducibility metadata
    metadata = {
        "seed": config["seed"],
        "dataset_root": config["dataset_root"],
        "train_split_path": config["train_split_path"],
        "val_split_path": config["val_split_path"],
        "model_config": model_cfg,
        "ablation_config": ablation_cfg,
        "loss_weights": config["loss_weights"],
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "batch_size": config["batch_size"],
        "epochs": epochs,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": gpu_name,
        "parameter_count": tot_params,
        "trainable_parameters": train_params,
        "model_size_mb": model_size,
        "total_training_time_seconds": total_train_time,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_train_loss": epoch_train_loss,
        "final_val_loss": epoch_val_loss
    }
    with open("outputs/echo_phase4/config.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    log_print(f"\nTraining completed in {total_train_time:.1f} seconds ({total_train_time/60:.2f} minutes).")
    log_print(f"Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.6f}")
    
    log_file.close()

if __name__ == "__main__":
    main()
