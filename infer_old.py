import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import SweepEvalDataset, imagenet_transform
from model import NEJMbaseline

def infer_test(test_csv, model_path='best_model.pth', n_sweeps_test=8, output_csv='test_predictions.csv'):
    """
    Run inference on test set using a trained NEJMbaseline model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = checkpoint['model_architecture']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Prepare test data
    test_df = pd.read_csv(test_csv)
    test_dataset = SweepEvalDataset(test_csv, n_sweeps=n_sweeps_test, transform=imagenet_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    predictions = []
    study_ids = []
    site = []

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing", leave=False)
        for i, (sweeps, _) in enumerate(test_pbar):
            sweeps = sweeps.to(device)
            B, S, T, C, H, W = sweeps.shape
            sweeps = sweeps.view(B, S*T, C, H, W)
            outputs, _ = model(sweeps)
            preds = outputs.squeeze(1).cpu().numpy()
            predictions.extend(preds)

            start_idx = i * test_loader.batch_size
            end_idx = min(start_idx + B, len(test_df))
            study_ids.extend(test_df.iloc[start_idx:end_idx]['study_id'].tolist())
            site.extend(test_df.iloc[start_idx:end_idx]['site'].tolist())

    # Save predictions to CSV
    result_df = pd.DataFrame({'study_id': study_ids, 'site': site, 'predicted_ga': predictions})
    result_df.to_csv(output_csv, index=False)
    print(f"✅ Saved predictions to {output_csv}")



if __name__ == "__main__":
    infer_test(
        test_csv='/mnt/Data/hackathon/final_test.csv',
        model_path="checkpoints/best_model.pth",
        n_sweeps_test=8,
        output_csv="outputs/test_predictions.csv"
    )

