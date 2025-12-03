# infer_new.py
import torch
import pandas as pd
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torchvision.transforms as T
import argparse
import os
from model import create_model  # Using our new factory function

# ImageNet transforms
imagenet_transform = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def load_sweep_frames(path, target_frames=16):
    """Load and preprocess a single sweep."""
    img = nib.load(path).get_fdata().astype(np.float32)
    num_frames = img.shape[-1]
    
    if num_frames >= target_frames:
        indices = np.linspace(0, num_frames - 1, target_frames, dtype=int)
        sampled_img = img[..., indices]
    else:
        repeat_factor = int(np.ceil(target_frames / num_frames))
        repeated_img = np.tile(img, (1, 1, 1, repeat_factor))
        sampled_img = repeated_img[..., :target_frames]
    
    # Apply transforms to each frame
    frames = []
    for f in range(sampled_img.shape[-1]):
        frame = np.repeat(sampled_img[:, :, :, f], 3, axis=2)
        frame = imagenet_transform(frame)
        frames.append(frame)
    
    return torch.stack(frames, dim=0)  # (T, C, H, W)

def load_model(checkpoint_path, device):
    """
    Load model from checkpoint.
    Automatically detects model type and architecture from checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration from checkpoint
    model_type = checkpoint.get('model_type', 'resnet18')
    architecture = checkpoint.get('architecture', 'flat')
    model_config = checkpoint.get('model_config', {})
    
    print(f"Loading {architecture}_{model_type} model...")
    print(f"Model config: {model_config}")
    
    # Create model
    model = create_model(
        model_type=model_type,
        architecture=architecture,
        reduced_dim=model_config.get('reduced_dim', 128),
        num_heads=model_config.get('num_heads', 4),
        fine_tune_backbone=False,  # Freeze for inference
        pretrained=False  # Weights come from checkpoint
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, architecture

def run_inference(
    csv_path,
    checkpoint_path,
    output_csv="test_predictions.csv",
    n_sweeps=8,
    batch_size=4
):
    """
    Run inference on test data.
    
    Args:
        csv_path: Path to test CSV
        checkpoint_path: Path to model checkpoint
        output_csv: Output CSV path
        n_sweeps: Number of sweeps to use per study
        batch_size: Batch size for inference
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, architecture = load_model(checkpoint_path, device)
    
    # Load test CSV
    test_df = pd.read_csv(csv_path)
    print(f"Loaded {len(test_df)} test samples")
    
    # Get sweep column names
    sweep_cols = [c for c in test_df.columns if c.startswith('path_nifti')]
    print(f"Found {len(sweep_cols)} sweep columns")
    
    predictions = []
    
    with torch.no_grad():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
            # Get sweep paths (up to n_sweeps)
            sweep_paths = [row[col] for col in sweep_cols if pd.notna(row[col])]
            sweep_paths = sweep_paths[:n_sweeps]
            
            if not sweep_paths:
                print(f"⚠️  No sweep paths found for study {row.get('study_id', idx)}")
                predictions.append(0.0)
                continue
            
            # Load all sweeps
            all_sweep_frames = []
            for path in sweep_paths:
                try:
                    frames = load_sweep_frames(path)
                    all_sweep_frames.append(frames)
                except Exception as e:
                    print(f"⚠️  Error loading {path}: {e}")
                    # Add zero frames as fallback
                    dummy_frames = torch.zeros((16, 3, 224, 224))
                    all_sweep_frames.append(dummy_frames)
            
            # Stack sweeps: (S, T, C, H, W)
            sweeps = torch.stack(all_sweep_frames, dim=0)
            
            # Prepare input based on architecture
            if architecture == 'flat':
                # Flat models: flatten sweeps (S, T, C, H, W) -> (1, S*T, C, H, W)
                S, T, C, H, W = sweeps.shape
                data = sweeps.unsqueeze(0).view(1, S * T, C, H, W).to(device)
            else:  # hierarchical
                # Hierarchical models: keep as (1, S, T, C, H, W)
                data = sweeps.unsqueeze(0).to(device)
            
            # Predict
            output, _ = model(data)
            pred = output.item()
            predictions.append(pred)
    
    # Create output DataFrame
    result_df = pd.DataFrame({
        'study_id': test_df['study_id'],
        'site': test_df['site'] if 'site' in test_df.columns else ['unknown'] * len(test_df),
        'predicted_ga': predictions
    })
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result_df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Saved {len(result_df)} predictions to {output_csv}")
    
    # Summary statistics
    print(f"\n📊 Prediction Statistics:")
    print(f"  Mean GA: {result_df['predicted_ga'].mean():.1f} days")
    print(f"  Std GA: {result_df['predicted_ga'].std():.1f} days")
    print(f"  Min GA: {result_df['predicted_ga'].min():.1f} days")
    print(f"  Max GA: {result_df['predicted_ga'].max():.1f} days")
    
    # Show first few predictions
    print(f"\n🔍 First 5 predictions:")
    print(result_df.head())
    
    return result_df

def batch_inference(
    test_csv,
    checkpoint_dir="checkpoints",
    output_dir="outputs",
    n_sweeps=8,
    batch_size=4
):
    """
    Run inference with multiple checkpoints.
    """
    # Find all checkpoints
    import glob
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoints)} checkpoints")
    
    results = {}
    for checkpoint_path in checkpoints:
        checkpoint_name = os.path.basename(checkpoint_path).replace('.pth', '')
        output_csv = os.path.join(output_dir, f"{checkpoint_name}_predictions.csv")
        
        print(f"\n{'='*60}")
        print(f"Processing: {checkpoint_name}")
        print(f"{'='*60}")
        
        try:
            result_df = run_inference(
                csv_path=test_csv,
                checkpoint_path=checkpoint_path,
                output_csv=output_csv,
                n_sweeps=n_sweeps,
                batch_size=batch_size
            )
            results[checkpoint_name] = result_df
        except Exception as e:
            print(f"❌ Error with {checkpoint_name}: {e}")
    
    # Compare results if multiple models
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print('='*60)
        
        comparison = []
        for name, df in results.items():
            comparison.append({
                'model': name,
                'mean_ga': df['predicted_ga'].mean(),
                'std_ga': df['predicted_ga'].std(),
                'min_ga': df['predicted_ga'].min(),
                'max_ga': df['predicted_ga'].max(),
                'num_predictions': len(df)
            })
        
        comparison_df = pd.DataFrame(comparison)
        print(comparison_df)
        
        # Save comparison
        comparison_csv = os.path.join(output_dir, "model_comparison.csv")
        comparison_df.to_csv(comparison_csv, index=False)
        print(f"\n📊 Comparison saved to: {comparison_csv}")

def main():
    parser = argparse.ArgumentParser(description='Run inference on GA prediction models')
    
    # Input/Output
    parser.add_argument('--test_csv', type=str, required=True,
                        help='Path to test CSV file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--output_csv', type=str, default='predictions.csv',
                        help='Output CSV path')
    
    # Inference parameters
    parser.add_argument('--n_sweeps', type=int, default=8,
                        help='Number of sweeps to use per study')
    parser.add_argument('--batch_mode', action='store_true',
                        help='Process all checkpoints in a directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory with checkpoints (for batch mode)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory (for batch mode)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for processing')
    
    args = parser.parse_args()
    
    if args.batch_mode:
        # Batch inference with all models
        batch_inference(
            test_csv=args.test_csv,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            n_sweeps=args.n_sweeps,
            batch_size=args.batch_size
        )
    else:
        # Single model inference
        run_inference(
            csv_path=args.test_csv,
            checkpoint_path=args.checkpoint,
            output_csv=args.output_csv,
            n_sweeps=args.n_sweeps,
            batch_size=args.batch_size
        )

if __name__ == "__main__":
    main()