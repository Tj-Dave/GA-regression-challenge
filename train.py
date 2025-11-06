import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from data import SweepDataset, SweepEvalDataset, imagenet_transform
from model import NEJMbaseline
import warnings

warnings.filterwarnings("ignore")


def train_and_validate(train_csv, val_csv, epochs=100, batch_size=8, n_sweeps_val=8, save_path='best_model.pth'):
    """
    Train and validate the NEJMbaseline model.
    Logs metrics to TensorBoard for both train and validation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------
    # Create log directory & writer
    # ----------------------------
    os.makedirs("logs", exist_ok=True)
    writer = SummaryWriter(log_dir="logs")

    # Datasets and loaders
    train_dataset = SweepDataset(train_csv, transform=imagenet_transform)
    val_dataset = SweepEvalDataset(val_csv, n_sweeps=n_sweeps_val, transform=imagenet_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model, loss, optimizer
    model = NEJMbaseline().to(device)
    criterion = nn.L1Loss()  # MAE
    mse_loss = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')

    # ----------------------------
    # Training loop
    # ----------------------------
    global_step = 0
    for epoch in range(epochs):
        # ---------------- Training ----------------
        model.train()
        train_loss = 0.0
        train_mae_epoch, train_mse_epoch = 0.0, 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)

        for batch_idx, (frames, labels) in enumerate(train_pbar):
            frames = frames.to(device)
            labels = labels.float().to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs, _ = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Compute metrics
            mae = torch.mean(torch.abs(outputs - labels)).item()
            mse = torch.mean((outputs - labels) ** 2).item()

            # Batchwise logs
            writer.add_scalar("Train/Batch_Loss", loss.item(), global_step)
            writer.add_scalar("Train/Batch_MAE", mae, global_step)
            writer.add_scalar("Train/Batch_MSE", mse, global_step)
            global_step += 1

            # Update running totals
            train_loss += loss.item() * frames.size(0)
            train_mae_epoch += mae * frames.size(0)
            train_mse_epoch += mse * frames.size(0)
            train_pbar.set_postfix({"batch_loss": loss.item(), "batch_mae": mae})

        # Epoch-wise averages
        train_loss /= len(train_loader.dataset)
        train_mae_epoch /= len(train_loader.dataset)
        train_mse_epoch /= len(train_loader.dataset)
        print(f"Epoch {epoch+1} | Train MAE: {train_mae_epoch:.4f} | MSE: {train_mse_epoch:.4f}")

        # Log epoch-level metrics
        writer.add_scalar("Train/Epoch_Loss", train_loss, epoch + 1)
        writer.add_scalar("Train/Epoch_MAE", train_mae_epoch, epoch + 1)
        writer.add_scalar("Train/Epoch_MSE", train_mse_epoch, epoch + 1)

        # ---------------- Validation ----------------
        model.eval()
        val_loss = 0.0
        val_mae_epoch, val_mse_epoch = 0.0, 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)

        with torch.no_grad():
            for sweeps, labels in val_pbar:
                sweeps = sweeps.to(device)
                B, S, T, C, H, W = sweeps.shape
                sweeps = sweeps.view(B, S * T, C, H, W)
                labels = labels.float().to(device).unsqueeze(1)

                outputs, _ = model(sweeps)
                loss = criterion(outputs, labels)

                # Metrics
                mae = torch.mean(torch.abs(outputs - labels)).item()
                mse = torch.mean((outputs - labels) ** 2).item()

                val_loss += loss.item() * B
                val_mae_epoch += mae * B
                val_mse_epoch += mse * B

                # Batchwise logs
                writer.add_scalar("Val/Batch_Loss", loss.item(), global_step)
                writer.add_scalar("Val/Batch_MAE", mae, global_step)
                writer.add_scalar("Val/Batch_MSE", mse, global_step)
                val_pbar.set_postfix({"batch_loss": loss.item(), "batch_mae": mae})

        # Epoch-wise averages
        val_loss /= len(val_loader.dataset)
        val_mae_epoch /= len(val_loader.dataset)
        val_mse_epoch /= len(val_loader.dataset)
        print(f"Epoch {epoch+1} | Val MAE: {val_mae_epoch:.4f} | MSE: {val_mse_epoch:.4f}")

        # Log epoch-level metrics
        writer.add_scalar("Val/Epoch_Loss", val_loss, epoch + 1)
        writer.add_scalar("Val/Epoch_MAE", val_mae_epoch, epoch + 1)
        writer.add_scalar("Val/Epoch_MSE", val_mse_epoch, epoch + 1)

        # ---------------- Save best model ----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"✅ Saving new best model (Val MAE: {val_loss:.4f})")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_architecture': model,
                'epoch': epoch + 1,
                'val_loss': val_loss
            }, save_path)

    # Close writer at the end
    writer.close()


if __name__ == "__main__":
    train_and_validate(
        train_csv="/mnt/Data/hackathon/final_train.csv",
        val_csv='/mnt/Data/hackathon/final_valid.csv',
        epochs=50,
        batch_size=8,
        n_sweeps_val=8,
        save_path="checkpoints/best_model.pth"
    )

