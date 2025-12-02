import pandas as pd
import numpy as np


# -------------------------
# BASIC METRICS
# -------------------------
def compute_basic_metrics(df):
    mae = np.mean(np.abs(df["pred_ga"] - df["true_ga"]))
    mse = np.mean((df["pred_ga"] - df["true_ga"]) ** 2)
    return mae, mse


# -------------------------
# TRIMESTER METRICS
# -------------------------
def compute_trimester_metrics(df):
    """
    Split GA range into 3 equal bins (not fixed weeks).
    """
    min_ga = df["true_ga"].min()
    max_ga = df["true_ga"].max()
    bins = np.linspace(min_ga, max_ga, 4)  # 3 bins → 4 edges

    df["trimester"] = pd.cut(df["true_ga"], bins=bins, labels=["T1", "T2", "T3"], include_lowest=True)

    results = {}
    for t in ["T1", "T2", "T3"]:
        sub = df[df["trimester"] == t]
        if len(sub) == 0:
            results[t] = {"MAE": None, "MSE": None, "N": 0}
        else:
            mae = np.mean(np.abs(sub["pred_ga"] - sub["true_ga"]))
            mse = np.mean((sub["pred_ga"] - sub["true_ga"]) ** 2)
            results[t] = {"MAE": mae, "MSE": mse, "N": len(sub)}

    return results


# -------------------------
# SITE METRICS
# -------------------------
def compute_site_metrics(df):
    sites = df["site"].unique()
    results = {}

    for s in sites:
        sub = df[df["site"] == s]

        mae = np.mean(np.abs(sub["pred_ga"] - sub["true_ga"]))
        mse = np.mean((sub["pred_ga"] - sub["true_ga"]) ** 2)

        results[s] = {
            "MAE": mae,
            "MSE": mse,
            "N": len(sub)
        }

    return results


# -------------------------
# MAIN EVAL FUNCTION
# -------------------------
def evaluate_predictions(csv_path, save_summary="evaluation_summary.txt"):
    df = pd.read_csv(csv_path)

    basic_mae, basic_mse = compute_basic_metrics(df)
    trimester_results = compute_trimester_metrics(df)
    site_results = compute_site_metrics(df)

    with open(save_summary, "w") as f:
        f.write("=== Overall Metrics ===\n")
        f.write(f"MAE: {basic_mae:.4f}\n")
        f.write(f"MSE: {basic_mse:.4f}\n\n")

        f.write("=== Trimester Metrics ===\n")
        for t, vals in trimester_results.items():
            f.write(f"{t}: N={vals['N']}  MAE={vals['MAE']}  MSE={vals['MSE']}\n")
        f.write("\n")

        f.write("=== Site Metrics ===\n")
        for s, vals in site_results.items():
            f.write(f"{s}: N={vals['N']}  MAE={vals['MAE']:.4f}  MSE={vals['MSE']:.4f}\n")

    print(f"Evaluation summary saved to {save_summary}")

    return {
        "overall": (basic_mae, basic_mse),
        "trimester": trimester_results,
        "site": site_results
    }


if __name__ == "__main__":
    evaluate_predictions(
        csv_path="test_predictions.csv",
        save_summary="evaluation_summary.txt"
    )
