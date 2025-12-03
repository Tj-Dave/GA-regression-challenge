import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

def compute_metrics_by_trimester(pred_csv, gt_csv):
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

    y_true = merged['ga_true']
    y_pred = merged['ga_pred']

    # Determine trimester thresholds (based on ground truth)
    max_ga = y_true.max()
    min_ga = y_true.min()
    step = (max_ga - min_ga) / 3

    trimester_bins = [min_ga, min_ga + step, min_ga + 2*step, max_ga]
    trimester_labels = ['Trimester 1', 'Trimester 2', 'Trimester 3']

    merged['trimester'] = pd.cut(y_true, bins=trimester_bins, labels=trimester_labels, include_lowest=True)

    # Store results
    results = []

    for trimester in trimester_labels:
        df_t = merged[merged['trimester'] == trimester]
        if df_t.empty:
            continue

        y_t_true = df_t['ga_true'].values
        y_t_pred = df_t['ga_pred'].values

        mae = mean_absolute_error(y_t_true, y_t_pred)
        mse = mean_squared_error(y_t_true, y_t_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_t_true, y_t_pred)
        corr, _ = pearsonr(y_t_true, y_t_pred)

        results.append({
            "Trimester": trimester,
            "Samples": len(y_t_true),
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R²": r2,
            "Corr": corr
        })

    results_df = pd.DataFrame(results)
    print("\n📊 Metrics by Trimester")
    print(results_df.round(4).to_string(index=False))

    # Overall metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)

    print("\n📈 Overall Metrics")
    print("---------------------------")
    print(f"MAE   : {mae:.4f}")
    print(f"MSE   : {mse:.4f}")
    print(f"RMSE  : {rmse:.4f}")
    print(f"R²    : {r2:.4f}")
    print(f"Corr  : {corr:.4f}")
    print("---------------------------")
    print(f"Samples evaluated: {len(y_true)}")

    return results_df
