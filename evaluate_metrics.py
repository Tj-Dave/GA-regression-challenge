import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")


def compute_metrics(pred_csv, gt_csv):
    # Load data
    pred_df = pd.read_csv(pred_csv)
    gt_df = pd.read_csv(gt_csv)

    # Ensure columns exist
    if 'study_id' not in pred_df.columns or 'ga' not in pred_df.columns:
        raise ValueError("Prediction CSV must contain 'study_id' and 'ga' columns.")
    if 'study_id' not in gt_df.columns or 'ga' not in gt_df.columns:
        raise ValueError("Ground truth CSV must contain 'study_id' and 'ga' columns.")

    # Merge on study_id
    merged = pd.merge(gt_df, pred_df, on='study_id', suffixes=('_true', '_pred'))
    if merged.empty:
        raise ValueError("No matching study_id values found between CSVs.")

    y_true = merged['ga_true'].values
    y_pred = merged['ga_pred'].values

    # Compute metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)

    print("\n📊 Evaluation Metrics")
    print("---------------------------")
    print(f"MAE   : {mae:.4f}")
    print(f"MSE   : {mse:.4f}")
    print(f"RMSE  : {rmse:.4f}")
    print(f"R²    : {r2:.4f}")
    print(f"Corr  : {corr:.4f}")
    print("---------------------------")
    print(f"Samples evaluated: {len(y_true)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute regression metrics between predictions and ground truth CSVs.")
    parser.add_argument("--pred", required=True, help="Path to predictions CSV (columns: study_id, ga)")
    parser.add_argument("--gt", required=True, help="Path to ground truth CSV (columns: study_id, ga)")
    args = parser.parse_args()

    compute_metrics(args.pred, args.gt)

