# train_sequential.py
import time
from train import train_model

# List of models to train in hierarchical mode
models_to_train = [
    {
        'model_type': 'resnet18',
        'architecture': 'hierarchical',
        'train_sweeps': 2,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'reduced_dim': 128,
        'num_heads': 4,
        'save_dir': 'checkpoints_hierarchical',
        'experiment_name': 'hierarchical_resnet18'
    },
    {
        'model_type': 'resnet50',
        'architecture': 'hierarchical',
        'train_sweeps': 2,
        'batch_size': 4,
        'learning_rate': 1e-4,
        'reduced_dim': 256,
        'num_heads': 4,
        'save_dir': 'checkpoints_hierarchical',
        'experiment_name': 'hierarchical_resnet50'
    },
    {
        'model_type': 'convnext_tiny',
        'architecture': 'hierarchical',
        'train_sweeps': 2,
        'batch_size': 8,
        'learning_rate': 3e-5,
        'reduced_dim': 384,
        'num_heads': 4,
        'save_dir': 'checkpoints_hierarchical',
        'experiment_name': 'hierarchical_convnext'
    }
]

print("Starting sequential hierarchical model training...")
print("="*60)

for i, config in enumerate(models_to_train):
    print(f"\nTraining Model {i+1}/{len(models_to_train)}: {config['model_type']}")
    print(f"   Architecture: {config['architecture']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Reduced dim: {config['reduced_dim']}")
    print(f"   Train sweeps: {config['train_sweeps']}")
    print("-" * 40)
    
    try:
        best_mae, best_loss = train_model(
            model_type=config['model_type'],
            architecture=config['architecture'],
            train_csv="/mnt/Data/hackathon/final_train.csv",
            val_csv="/mnt/Data/hackathon/final_valid.csv",
            epochs=50,
            batch_size=config['batch_size'],
            train_sweeps=config['train_sweeps'],
            n_sweeps_val=8,
            learning_rate=config['learning_rate'],
            reduced_dim=config['reduced_dim'],
            num_heads=config['num_heads'],
            fine_tune_backbone=True,
            pretrained=True,
            save_dir=config['save_dir']
        )
        
        print(f"{config['model_type']} completed!")
        print(f"   Best MAE: {best_mae:.4f}")
        print(f"   Best Loss: {best_loss:.4f}")
        
    except Exception as e:
        print(f"{config['model_type']} failed: {e}")
    
    # Wait between models (optional)
    if i < len(models_to_train) - 1:
        print("\nWaiting 5 seconds before next model...")
        time.sleep(5)

print("\n" + "="*60)
print("All hierarchical models training completed!")
print("="*60)