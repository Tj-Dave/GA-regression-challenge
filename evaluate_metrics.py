"""
Evaluation script for multi-center gestational age prediction.

This script evaluates submissions using:
- Mean Absolute Error (MAE) across all predictions
- Fairness penalty based on performance disparity across sites

Final Score = Overall MAE + penalty_weight * (max_site_MAE - min_site_MAE)

Lower scores are better.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import json


def calculate_mae(predictions, ground_truth):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(predictions - ground_truth))


def evaluate_submission(submission_csv, ground_truth_csv, penalty_weight=0.3, verbose=True):
    """
    Evaluate a submission file against ground truth.
    
    Args:
        submission_path: Path to submission CSV with columns [patient_id, site, predicted_GA]
        ground_truth_path: Path to ground truth CSV with columns [patient_id, site, gestational_age_days]
        penalty_weight: Weight for fairness penalty (default: 0.3)
        verbose: Whether to print detailed results
    
    Returns:
        results: dict with overall metrics and per-site breakdown
    """
    
    # Load data
    submission = pd.read_csv(submission_csv)
    ground_truth = pd.read_csv(ground_truth_csv)
    
    # Validate submission format
    required_columns = ['patient_id', 'site', 'predicted_ga']
    missing_columns = set(required_columns) - set(submission.columns)
    if missing_columns:
        raise ValueError(f"Submission missing required columns: {missing_columns}")
    
    # Merge submission with ground truth
    merged = ground_truth.merge(
        submission,
        on=['patient_id', 'site'],
        how='inner',
        suffixes=('_true', '_pred')
    )
    
    # Check for missing predictions
    if len(merged) < len(ground_truth):
        missing_count = len(ground_truth) - len(merged)
        print(f"[!] Warning: {missing_count} patients missing from submission")
        missing_patients = set(ground_truth['patient_id']) - set(merged['patient_id'])
        print(f"  Missing patient IDs: {list(missing_patients)[:10]}...")
    
    # Check for extra predictions
    if len(submission) > len(ground_truth):
        extra_count = len(submission) - len(merged)
        print(f"[!] Warning: {extra_count} extra predictions in submission (will be ignored)")
    
    if len(merged) == 0:
        raise ValueError("No valid predictions found in submission")
    
    # Calculate overall MAE
    overall_mae = calculate_mae(
        merged['predicted_ga'].values,
        merged['ga'].values
    )
    
    # Calculate per-site MAE
    sites = sorted(merged['site'].unique())
    site_maes = {}
    site_counts = {}
    
    for site in sites:
        site_data = merged[merged['site'] == site]
        site_mae = calculate_mae(
            site_data['predicted_ga'].values,
            site_data['ga'].values
        )
        site_maes[site] = site_mae
        site_counts[site] = len(site_data)
    
    # Calculate disparity and fairness metrics
    max_site_mae = max(site_maes.values())
    min_site_mae = min(site_maes.values())
    site_disparity = max_site_mae - min_site_mae
    
    # Calculate standard deviation and coefficient of variation
    site_mae_values = list(site_maes.values())
    site_mae_std = np.std(site_mae_values)
    coefficient_of_variation = site_mae_std / overall_mae if overall_mae > 0 else 0
    
    # Calculate final score using NORMALIZED FORMULA
    # Score = (1 - λ) × MAE + λ × Disparity
    final_score = (1 - penalty_weight) * overall_mae + penalty_weight * site_disparity
    fairness_penalty = penalty_weight * site_disparity
    
    # Compile results
    results = {
        'final_score': final_score,
        'overall_mae': overall_mae,
        'fairness_penalty': fairness_penalty,
        'site_disparity': site_disparity,
        'max_site_mae': max_site_mae,
        'min_site_mae': min_site_mae,
        'site_mae_std': site_mae_std,
        'coefficient_of_variation': coefficient_of_variation,
        'penalty_weight': penalty_weight,
        'num_predictions': len(merged),
        'num_sites': len(sites),
        'per_site_metrics': {}
    }
    
    for site in sites:
        results['per_site_metrics'][site] = {
            'mae': site_maes[site],
            'count': site_counts[site]
        }
    
    # Print results if verbose
    if verbose:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"\n[INFO] FINAL SCORE: {final_score:.4f} days")
        print(f"   (Lower is better)")
        print("\nScore Formula:")
        print(f"  Score = {1-penalty_weight:.1f} x MAE + {penalty_weight:.1f} x Disparity")
        print(f"  Score = {1-penalty_weight:.1f} x {overall_mae:.4f} + {penalty_weight:.1f} x {site_disparity:.4f}")
        print(f"  Score = {final_score:.4f} days")
        print("\n[SUMMARY] Performance Summary:")
        print(f"  Overall MAE:        {overall_mae:.4f} days")
        print(f"  Site Std Dev:       {site_mae_std:.4f} days")
        print(f"  Coeff of Variation: {coefficient_of_variation:.4f}")
        print(f"  Total Predictions:  {len(merged)}")
        print(f"  Number of Sites:    {len(sites)}")
        print(f"\n[SITES] Per-Site Performance:")
        print(f"  {'Site':<15} {'MAE (days)':<12} {'Count':<8}")
        print(f"  {'-'*15} {'-'*12} {'-'*8}")
        
        for site in sorted(sites):
            mae = site_maes[site]
            count = site_counts[site]
            # Highlight best and worst
            marker = ""
            if mae == min_site_mae:
                marker = " [*] (best)"
            elif mae == max_site_mae:
                marker = " [!] (worst)"
            print(f"  {site:<15} {mae:<12.4f} {count:<8}{marker}")
        
        print(f"\n[FAIRNESS] Fairness Analysis:")
        print(f"  Best Site MAE:      {min_site_mae:.4f} days")
        print(f"  Worst Site MAE:     {max_site_mae:.4f} days")
        print(f"  Disparity:          {site_disparity:.4f} days")
        print(f"  Site Std Dev:       {site_mae_std:.4f} days")
        print(f"  Fairness Penalty:   {fairness_penalty:.4f} days")
        print(f"    (Weight: {penalty_weight:.1%} of score from fairness)")
        
        # Interpretation
        print(f"\n[NOTE] Interpretation:")
        if site_disparity < 3.0:
            print(f"  [*] Excellent fairness - very consistent across sites (disparity < 3 days)")
        elif site_disparity < 5.0:
            print(f"  [*] Good fairness - consistent performance across sites (disparity < 5 days)")
        elif site_disparity < 8.0:
            print(f"  [!] Moderate disparity - some sites perform notably worse (disparity 5-8 days)")
        else:
            print(f"  [!] High disparity - significant performance gaps between sites (disparity > 8 days)")
        
        if coefficient_of_variation < 0.1:
            print(f"  [*] Very low variation (CV < 0.1)")
        elif coefficient_of_variation < 0.2:
            print(f"  [*] Low variation (CV < 0.2)")
        else:
            print(f"  [!] High variation (CV > 0.2)")
        
        print("="*60 + "\n")
    
    return results


def main(args):
    """Main evaluation function."""
    
    print(f"Evaluating submission: {args.submission}")
    print(f"Ground truth: {args.ground_truth}")
    print(f"Penalty weight: {args.penalty_weight}")
    
    try:
        results = evaluate_submission(
            submission_csv=args.submission,
            ground_truth_csv=args.ground_truth,
            penalty_weight=args.penalty_weight,
            verbose=True
        )
        
        # Save results to JSON if requested
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"[*] Results saved to: {output_path}")
        
        # Return exit code based on success
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate gestational age predictions with fairness penalty',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python evaluate.py --submission submission.csv --ground_truth test.csv
  
  python evaluate.py --submission team1.csv --ground_truth gt.csv --penalty_weight 0.5
  
Submission CSV format:
  patient_id,site,predicted_GA
  P001,site_A,245.5
  P002,site_B,198.2
  
Ground truth CSV format:
  patient_id,site,gestational_age_days
  P001,site_A,248.0
  P002,site_B,196.0
        """
    )
    
    parser.add_argument(
        '--submission',
        type=str,
        required=True,
        help='Path to submission CSV file'
    )
    
    parser.add_argument(
        '--ground_truth',
        type=str,
        required=True,
        help='Path to ground truth CSV file'
    )
    
    parser.add_argument(
        '--penalty_weight',
        type=float,
        default=0.3,
        help='Weight for fairness penalty (default: 0.3). Higher values penalize site disparity more.'
    )
    
    parser.add_argument(
        '--output_json',
        type=str,
        default='outputs/evaluation_results.json',
        help='Optional path to save results as JSON'
    )
    
    args = parser.parse_args()
    
    exit(main(args))