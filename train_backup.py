import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Import the models and data
from data_old import SweepDataset, SweepEvalDataset, imagenet_transform
from model import create_model  # Using the factory function

def train_model(
    model_type='resnet18',
    train_csv="/mnt/Data/hackathon/final_train.csv",
    val_csv="/mnt/Data/hackathon/final_valid.csv",
    epochs=100,
    batch_size=8,
    n_sweeps_val=8,
    learning_rate=1e-4,
    reduced_dim=128,
    fine_tune_backbone=True,
    pretrained=True,
    save_dir="checkpoints"
):
    """
    Train any GA prediction model with single-sweep training and multi-sweep validation.
    
    Args:
        model_type: 'resnet18', 'resnet50', or 'convnext_tiny'
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        epochs: Number of training epochs
        batch_size: Batch size
        n_sweeps_val: Number of sweeps to use during validation
        learning_rate: Initial learning rate
        reduced_dim: Dimension for attention reduced features
        fine_tune_backbone: Whether to fine-tune the backbone
        pretrained: Use pretrained weights
        save_dir: Directory to save checkpoints
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training {model_type} model...")
    
    # Create timestamp for unique experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{model_type}_{timestamp}"
    
    # Create directories
    log_dir = os.path.join("logs", experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Datasets and loaders
    print("Loading datasets...")
    train_dataset = SweepDataset(train_csv, transform=imagenet_transform)
    val_dataset = SweepEvalDataset(val_csv, n_sweeps=n_sweeps_val, transform=imagenet_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print(f"Creating {model_type} model...")
    model = create_model(
        model_type=model_type,
        reduced_dim=reduced_dim,
        fine_tune_backbone=fine_tune_backbone,
        pretrained=pretrained
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss, optimizer, scheduler
    criterion = nn.L1Loss()  # MAE loss
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    best_val_mae = float('inf')
    best_val_loss = float('inf')
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    global_step = 0
    for epoch in range(epochs):
        # ---------- Training ----------
        model.train()
        train_loss = 0.0
        train_mae_epoch = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        
        for frames, labels in train_pbar:
            frames = frames.to(device)
            labels = labels.float().to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs, _ = model(frames)  # Single sweep training
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Compute MAE
            mae = torch.mean(torch.abs(outputs - labels)).item()
            
            # Update metrics
            batch_size_actual = frames.size(0)
            train_loss += loss.item() * batch_size_actual
            train_mae_epoch += mae * batch_size_actual
            
            # TensorBoard logging
            writer.add_scalar(f"Train/{model_type}/Batch_Loss", loss.item(), global_step)
            writer.add_scalar(f"Train/{model_type}/Batch_MAE", mae, global_step)
            writer.add_scalar(f"Train/{model_type}/Learning_Rate", optimizer.param_groups[0]['lr'], global_step)
            global_step += 1
            
            train_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mae": f"{mae:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Calculate epoch training metrics
        train_loss /= len(train_loader.dataset)
        train_mae_epoch /= len(train_loader.dataset)
        
        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        val_mae_epoch = 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        
        with torch.no_grad():
            for sweeps, labels in val_pbar:
                sweeps = sweeps.to(device)
                labels = labels.float().to(device).unsqueeze(1)
                
                # For validation: flatten sweeps (S sweeps × T frames)
                B, S, T, C, H, W = sweeps.shape
                sweeps_flat = sweeps.view(B, S * T, C, H, W)
                
                outputs, _ = model(sweeps_flat)  # Multi-sweep validation
                loss = criterion(outputs, labels)
                mae = torch.mean(torch.abs(outputs - labels)).item()
                
                batch_size_actual = sweeps.size(0)
                val_loss += loss.item() * batch_size_actual
                val_mae_epoch += mae * batch_size_actual
                
                val_pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "mae": f"{mae:.4f}"
                })
        
        # Calculate epoch validation metrics
        val_loss /= len(val_loader.dataset)
        val_mae_epoch /= len(val_loader.dataset)
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train MAE: {train_mae_epoch:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val MAE:   {val_mae_epoch:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # TensorBoard epoch logging
        writer.add_scalar(f"Train/{model_type}/Epoch_Loss", train_loss, epoch)
        writer.add_scalar(f"Train/{model_type}/Epoch_MAE", train_mae_epoch, epoch)
        writer.add_scalar(f"Val/{model_type}/Epoch_Loss", val_loss, epoch)
        writer.add_scalar(f"Val/{model_type}/Epoch_MAE", val_mae_epoch, epoch)
        
        # Save best model
        if val_mae_epoch < best_val_mae:
            best_val_mae = val_mae_epoch
            best_val_loss = val_loss
            
            # Save model checkpoint
            checkpoint_path = os.path.join(save_dir, f"best_model_{model_type}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae_epoch,
                'train_loss': train_loss,
                'train_mae': train_mae_epoch,
                'model_type': model_type,
                'model_config': {
                    'reduced_dim': reduced_dim,
                    'fine_tune_backbone': fine_tune_backbone,
                    'pretrained': pretrained
                }
            }, checkpoint_path)
            
            print(f"✅ Saved new best model (Val MAE: {val_mae_epoch:.4f}) to {checkpoint_path}")
    
    # Close TensorBoard writer
    writer.close()
    
    # Final summary
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Best validation MAE: {best_val_mae:.4f}")
    print(f"Best validation Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {os.path.join(save_dir, f'best_model_{model_type}.pth')}")
    print(f"TensorBoard logs: {log_dir}")
    
    return best_val_mae, best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Train GA prediction models')
    
    # Model selection
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50', 'convnext_tiny'],
                        help='Model backbone to use')
    
    # Data paths
    parser.add_argument('--train_csv', type=str, 
                        default="/mnt/Data/hackathon/final_train.csv",
                        help='Path to training CSV')
    parser.add_argument('--val_csv', type=str,
                        default="/mnt/Data/hackathon/final_valid.csv",
                        help='Path to validation CSV')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--reduced_dim', type=int, default=128,
                        help='Dimension for attention reduced features')
    
    # Model configuration
    parser.add_argument('--no_fine_tune', action='store_false', dest='fine_tune_backbone',
                        help='Freeze backbone (do not fine-tune)')
    parser.add_argument('--no_pretrained', action='store_false', dest='pretrained',
                        help='Do not use pretrained weights')
    
    # Validation
    parser.add_argument('--n_sweeps_val', type=int, default=8,
                        help='Number of sweeps to use during validation')
    
    # Output
    parser.add_argument('--save_dir', type=str, default="checkpoints",
                        help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Train the model
    train_model(
        model_type=args.model,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_sweeps_val=args.n_sweeps_val,
        learning_rate=args.lr,
        reduced_dim=args.reduced_dim,
        fine_tune_backbone=args.fine_tune_backbone,
        pretrained=args.pretrained,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()