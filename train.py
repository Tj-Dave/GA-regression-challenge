import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from data import SweepDataset, SweepEvalDataset
from model import HierarchicalGA
import warnings

warnings.filterwarnings("ignore")


def train_and_validate_hierarchical(train_csv, val_csv, 
                                    epochs=100, 
                                    batch_size=8, 
                                    train_sweeps=2,
                                    val_sweeps=8, 
                                    save_path='checkpoints/hierarchical_best_model.pth'):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs("logs_hierarchical", exist_ok=True)
    writer = SummaryWriter(log_dir="logs_hierarchical")

    # ----------------------------------
    # Datasets — NO transform needed now
    # ----------------------------------
    train_dataset = SweepDataset(train_csv, n_sweeps=train_sweeps)
    val_dataset = SweepEvalDataset(val_csv, n_sweeps=val_sweeps)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # ----------------------------------
    # Model
    # ----------------------------------
    model = HierarchicalGA(
        backbone_type='convnext_tiny',
        feature_dim=768,
        reduced_dim=128,
        num_heads=4,
        fine_tune_backbone=True,
        pretrained=True
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)

    scaler = torch.cuda.amp.GradScaler()

    best_val_mae = float('inf')

    global_step = 0

    for epoch in range(epochs):

        # --------------------
        # Training
        # --------------------
        model.train()
        train_mae_epoch = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for sweeps, labels in train_pbar:

            sweeps = sweeps.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs, _ = model(sweeps)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            mae = torch.mean(torch.abs(outputs - labels)).item()

            train_mae_epoch += mae * sweeps.size(0)

            writer.add_scalar("Train/Batch_Loss", loss.item(), global_step)
            writer.add_scalar("Train/Batch_MAE", mae, global_step)

            global_step += 1

        train_mae_epoch /= len(train_loader.dataset)
        print(f"Epoch {epoch+1} | Train MAE: {train_mae_epoch:.4f}")
        writer.add_scalar("Train/Epoch_MAE", train_mae_epoch, epoch + 1)

        # --------------------
        # Validation
        # --------------------
        model.eval()
        val_mae_epoch = 0.0
        last_attention = None

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")

        with torch.no_grad():
            for sweeps, labels in val_pbar:

                sweeps = sweeps.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                with torch.cuda.amp.autocast():
                    outputs, att_dict = model(sweeps, return_attention=True)
                    loss = criterion(outputs, labels)

                mae = torch.mean(torch.abs(outputs - labels)).item()
                val_mae_epoch += mae * sweeps.size(0)

                last_attention = att_dict

        val_mae_epoch /= len(val_loader.dataset)
        print(f"Epoch {epoch+1} | Val MAE: {val_mae_epoch:.4f}")
        writer.add_scalar("Val/Epoch_MAE", val_mae_epoch, epoch + 1)

        # Log attention every 5 epochs
        if last_attention and epoch % 5 == 0:
            att = last_attention["frame_attention"][0]  # (S, T)
            for s in range(att.shape[0]):
                writer.add_histogram(f"Attention/Sweep_{s}", att[s], global_step)

        scheduler.step(val_mae_epoch)

        # Save checkpoint
        if val_mae_epoch < best_val_mae:
            best_val_mae = val_mae_epoch
            print(f"🔥 Saving best model (Val MAE: {val_mae_epoch:.4f})")

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_mae": val_mae_epoch,
            }, save_path)

    writer.close()
    print(f"Training finished. Best MAE = {best_val_mae:.4f}")
    return best_val_mae
