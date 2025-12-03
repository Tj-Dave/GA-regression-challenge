import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import SweepEvalDataset, imagenet_transform
from model import HierarchicalGA


def load_model(checkpoint_path, device):
    """
    Load HierarchicalGA model for inference.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = HierarchicalGA(
        backbone_type='convnext_tiny',
        feature_dim=768,
        reduced_dim=128,
        num_heads=4,
        fine_tune_backbone=False,  # freeze backbone at inference
        pretrained=False           # weights come from checkpoint
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def run_inference(csv_path,
                  checkpoint_path,
                  output_csv="test_predictions.csv",
                  n_sweeps=None,
                  batch_size=4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path, device)

    print("Preparing dataset...")
    dataset = SweepEvalDataset(
        csv_path,
        n_sweeps=n_sweeps,     # use all sweeps unless specified
        transform=imagenet_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    all_preds = []
    all_labels = []
    all_ids = []

    print("Running inference...")
    with torch.no_grad():
        for sweeps, labels in tqdm(loader):
            sweeps = sweeps.to(device)              # (B, S, T, C, H, W)
            outputs, _ = model(sweeps)              # predictions
            preds = outputs.squeeze(1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    df = pd.read_csv(csv_path)
    df["pred_ga"] = all_preds
    df["true_ga"] = all_labels

    df.to_csv(output_csv, index=False)
    print(f"\nInference complete! Results saved to: {output_csv}")

    return output_csv


if __name__ == "__main__":
    run_inference(
        csv_path="/mnt/Data/hackathon/final_test.csv",
        checkpoint_path="checkpoints/hierarchical_best_model.pth",
        output_csv="outputs/test_ug_predictions.csv",
        n_sweeps=None,
        batch_size=4
    )
