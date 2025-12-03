import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Try to import ConvNeXt (available in torchvision >= 0.13)
try:
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    CONVNEXT_AVAILABLE = True
except ImportError:
    CONVNEXT_AVAILABLE = False
    convnext_tiny = None
    ConvNeXt_Tiny_Weights = None


# ============================================================================
# ATTENTION MODULES
# ============================================================================

class WeightedAverageAttention(nn.Module):
    """
    Original attention: Flattens all frames, applies single attention.
    Input: (B, T, D) where T = total frames (S*T if multiple sweeps)
    Output: (B, D')
    """
    def __init__(self, feature_dim=512, reduced_dim=128):
        super().__init__()
        self.W = nn.Linear(feature_dim, 64)
        self.V = nn.Linear(64, 1)
        self.Q = nn.Linear(feature_dim, reduced_dim)

    def forward(self, features):
        attn_scores = self.V(torch.tanh(self.W(features)))  # (B,T,1)
        attn_weights = F.softmax(attn_scores, dim=1)        # (B,T,1)
        reduced_features = self.Q(features)                 # (B,T,reduced_dim)
        weighted_sum = torch.sum(attn_weights * reduced_features, dim=1)
        return weighted_sum, attn_weights.squeeze(-1)


class IntraSweepAttention(nn.Module):
    """
    Intra-sweep attention: Processes frames within each sweep separately.
    Input: (B, S, T, D) where S = sweeps, T = frames per sweep
    Output: (B, S, D') - per-sweep embeddings
    """
    def __init__(self, feature_dim=512, reduced_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.reduced_dim = reduced_dim
        self.num_heads = num_heads
        
        # Multi-head attention for intra-sweep processing
        self.mha = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        
        # Learnable query for summarizing each sweep
        self.query = nn.Parameter(torch.randn(1, 1, feature_dim))
        
        # Projection to reduced dimension
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, reduced_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(reduced_dim)
        )
        
    def forward(self, features):
        """
        Args:
            features: (B, S, T, D) - batch, sweeps, frames, features
        Returns:
            sweep_embeddings: (B, S, D') - per-sweep embeddings
            frame_attention: (B, S, T) - attention weights per frame
        """
        B, S, T, D = features.shape
        
        # Reshape for batch processing: (B, S, T, D) -> (B*S, T, D)
        features_flat = features.view(B * S, T, D)
        
        # Create query: (B*S, 1, D)
        query = self.query.repeat(B * S, 1, 1)
        
        # Multi-head attention: summarize each sweep
        attended, attn_weights = self.mha(query, features_flat, features_flat)
        # attended: (B*S, 1, D), attn_weights: (B*S, 1, T)
        
        # Squeeze and project
        attended = attended.squeeze(1)  # (B*S, D)
        sweep_embeddings = self.projection(attended)  # (B*S, D')
        
        # Reshape back
        sweep_embeddings = sweep_embeddings.view(B, S, self.reduced_dim)  # (B, S, D')
        frame_attention = attn_weights.squeeze(1).view(B, S, T)  # (B, S, T)
        
        return sweep_embeddings, frame_attention


# ============================================================================
# REGRESSION HEADS
# ============================================================================

class SimpleRegressionHead(nn.Module):
    """Simple linear regression head (original)"""
    def __init__(self, input_dim=128):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.fc(x)


class MLPRegressionHead(nn.Module):
    """MLP regression head with GELU and dropout"""
    def __init__(self, input_dim=128, hidden_dims=[256, 128], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.mlp(x)


# ============================================================================
# BASE MODEL CLASSES
# ============================================================================

class BaseFlatModel(nn.Module):
    """
    Base class for FLAT models (original architecture).
    Flattens all frames, applies single attention.
    """
    def __init__(self, feature_extractor, feature_dim, reduced_dim=128, 
                 fine_tune_backbone=True, use_mlp_head=False):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_dim = feature_dim
        
        # Optionally freeze backbone
        if not fine_tune_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        # Attention module
        self.attention = WeightedAverageAttention(
            feature_dim=feature_dim, 
            reduced_dim=reduced_dim
        )
        
        # Regression head
        if use_mlp_head:
            self.reg_head = MLPRegressionHead(input_dim=reduced_dim)
        else:
            self.reg_head = SimpleRegressionHead(input_dim=reduced_dim)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, C, H, W) OR (B, S, T, C, H, W)
        Returns:
            output: Predicted GA (B, 1)
            attn_weights: Attention weights (B, T) or (B, S*T)
        """
        # Handle both single-sweep (B, T, C, H, W) and multi-sweep (B, S, T, C, H, W)
        if x.dim() == 5:
            # Single sweep: (B, T, C, H, W)
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
        elif x.dim() == 6:
            # Multi-sweep: (B, S, T, C, H, W) -> flatten sweeps
            B, S, T, C, H, W = x.shape
            x = x.view(B * S * T, C, H, W)
            T = S * T  # Update total frames
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Reshape features to (B, T, D)
        if features.dim() == 4:  # (B*T, D, 1, 1) for ResNet
            features = features.view(B, T, self.feature_dim)
        elif features.dim() == 2:  # (B*T, D) for ConvNeXt
            features = features.view(B, T, self.feature_dim)
        else:
            features = features.view(B, T, -1)
        
        # Apply attention
        aggregated, attn_weights = self.attention(features)
        
        # Regression
        output = self.reg_head(aggregated)
        
        return output, attn_weights


class BaseHierarchicalModel(nn.Module):
    """
    Base class for HIERARCHICAL models (new architecture).
    Processes sweeps separately with intra-sweep attention.
    """
    def __init__(self, feature_extractor, feature_dim, reduced_dim=128,
                 num_heads=4, fine_tune_backbone=True, use_mlp_head=True):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_dim = feature_dim
        
        # Optionally freeze backbone
        if not fine_tune_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        # Intra-sweep attention
        self.intra_sweep_attention = IntraSweepAttention(
            feature_dim=feature_dim,
            reduced_dim=reduced_dim,
            num_heads=num_heads
        )
        
        # Simple sweep aggregator (mean)
        self.sweep_aggregator = lambda x: x.mean(dim=1)
        
        # Regression head
        if use_mlp_head:
            self.reg_head = MLPRegressionHead(input_dim=reduced_dim)
        else:
            self.reg_head = SimpleRegressionHead(input_dim=reduced_dim)
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: Tensor of shape (B, S, T, C, H, W) - MUST be multi-sweep!
            return_attention: If True, return attention dictionary
        Returns:
            output: Predicted GA (B, 1)
            attention_info: Either attn_weights or dict with detailed info
        """
        B, S, T, C, H, W = x.shape
        
        # Extract features: (B, S, T, C, H, W) -> (B*S*T, C, H, W)
        x_flat = x.view(B * S * T, C, H, W)
        features_flat = self.feature_extractor(x_flat)
        
        # Reshape to (B, S, T, D)
        if features_flat.dim() == 4:  # (B*S*T, D, 1, 1) for ResNet
            features_flat = features_flat.view(B * S * T, -1)
        features = features_flat.view(B, S, T, self.feature_dim)
        
        # Intra-sweep attention
        sweep_embeddings, frame_attention = self.intra_sweep_attention(features)
        # sweep_embeddings: (B, S, D'), frame_attention: (B, S, T)
        
        # Aggregate sweeps
        study_embedding = self.sweep_aggregator(sweep_embeddings)  # (B, D')
        
        # Regression
        output = self.reg_head(study_embedding)  # (B, 1)
        
        if return_attention:
            attention_dict = {
                'frame_attention': frame_attention,  # (B, S, T)
                'sweep_embeddings': sweep_embeddings,  # (B, S, D')
                'study_embedding': study_embedding  # (B, D')
            }
            return output, attention_dict
        
        return output, frame_attention


# ============================================================================
# CONCRETE MODEL CLASSES - FLAT (ORIGINAL)
# ============================================================================

class NEJMbaselineFlat(BaseFlatModel):
    """ResNet18 with flat attention (original baseline)"""
    def __init__(self, reduced_dim=128, fine_tune_backbone=True, 
                 pretrained=True, use_mlp_head=False):
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        feature_dim = 512
        
        super().__init__(feature_extractor, feature_dim, reduced_dim, 
                        fine_tune_backbone, use_mlp_head)


class ResNet50Flat(BaseFlatModel):
    """ResNet50 with flat attention"""
    def __init__(self, reduced_dim=128, fine_tune_backbone=True,
                 pretrained=True, use_mlp_head=False):
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        feature_dim = 2048
        
        super().__init__(feature_extractor, feature_dim, reduced_dim,
                        fine_tune_backbone, use_mlp_head)


class ConvNeXtTinyFlat(BaseFlatModel):
    """ConvNeXt-Tiny with flat attention"""
    def __init__(self, reduced_dim=128, fine_tune_backbone=True,
                 pretrained=True, use_mlp_head=False):
        if not CONVNEXT_AVAILABLE:
            raise ImportError("ConvNeXt requires torchvision >= 0.13")
        
        convnext = convnext_tiny(
            weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        )
        
        feature_extractor = nn.Sequential(
            convnext.features,
            convnext.avgpool,
            nn.Flatten(1)
        )
        feature_dim = 768
        
        super().__init__(feature_extractor, feature_dim, reduced_dim,
                        fine_tune_backbone, use_mlp_head)


# ============================================================================
# CONCRETE MODEL CLASSES - HIERARCHICAL (NEW)
# ============================================================================

class NEJMbaselineHierarchical(BaseHierarchicalModel):
    """ResNet18 with intra-sweep attention"""
    def __init__(self, reduced_dim=128, num_heads=4, fine_tune_backbone=True,
                 pretrained=True, use_mlp_head=True):
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        feature_dim = 512
        
        super().__init__(feature_extractor, feature_dim, reduced_dim,
                        num_heads, fine_tune_backbone, use_mlp_head)


class ResNet50Hierarchical(BaseHierarchicalModel):
    """ResNet50 with intra-sweep attention"""
    def __init__(self, reduced_dim=128, num_heads=4, fine_tune_backbone=True,
                 pretrained=True, use_mlp_head=True):
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        feature_dim = 2048
        
        super().__init__(feature_extractor, feature_dim, reduced_dim,
                        num_heads, fine_tune_backbone, use_mlp_head)


class ConvNeXtTinyHierarchical(BaseHierarchicalModel):
    """ConvNeXt-Tiny with intra-sweep attention"""
    def __init__(self, reduced_dim=128, num_heads=4, fine_tune_backbone=True,
                 pretrained=True, use_mlp_head=True):
        if not CONVNEXT_AVAILABLE:
            raise ImportError("ConvNeXt requires torchvision >= 0.13")
        
        convnext = convnext_tiny(
            weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        )
        
        feature_extractor = nn.Sequential(
            convnext.features,
            convnext.avgpool,
            nn.Flatten(1)
        )
        feature_dim = 768
        
        super().__init__(feature_extractor, feature_dim, reduced_dim,
                        num_heads, fine_tune_backbone, use_mlp_head)


# ============================================================================
# MODEL FACTORY FUNCTIONS
# ============================================================================

def create_flat_model(model_type='resnet18', **kwargs):
    """Create flat attention model"""
    flat_registry = {
        'resnet18': NEJMbaselineFlat,
        'resnet50': ResNet50Flat,
        'convnext_tiny': ConvNeXtTinyFlat,
    }
    
    if model_type not in flat_registry:
        raise ValueError(f"Unknown flat model type: {model_type}")
    
    return flat_registry[model_type](**kwargs)


def create_hierarchical_model(model_type='resnet18', **kwargs):
    """Create hierarchical model with intra-sweep attention"""
    hierarchical_registry = {
        'resnet18': NEJMbaselineHierarchical,
        'resnet50': ResNet50Hierarchical,
        'convnext_tiny': ConvNeXtTinyHierarchical,
    }
    
    if model_type not in hierarchical_registry:
        raise ValueError(f"Unknown hierarchical model type: {model_type}")
    
    return hierarchical_registry[model_type](**kwargs)


def create_model(model_type='resnet18', architecture='flat', **kwargs):
    """
    General factory function to create any model.
    
    Args:
        model_type: 'resnet18', 'resnet50', or 'convnext_tiny'
        architecture: 'flat' (original) or 'hierarchical' (intra-sweep)
        **kwargs: Model-specific parameters
    
    Example:
        # Original flat models
        model = create_model('resnet18', architecture='flat')
        model = create_model('resnet50', architecture='flat', reduced_dim=256)
        
        # New hierarchical models
        model = create_model('resnet18', architecture='hierarchical')
        model = create_model('convnext_tiny', architecture='hierarchical', num_heads=8)
    """
    if architecture == 'flat':
        return create_flat_model(model_type, **kwargs)
    elif architecture == 'hierarchical':
        return create_hierarchical_model(model_type, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Use 'flat' or 'hierarchical'")


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_all_models():
    """Test all model configurations"""
    batch_size = 2
    sweeps = 2
    frames = 16
    channels = 3
    height = width = 224
    
    print("Testing models...")
    
    # Test configurations
    test_cases = [
        ('resnet18', 'flat'),
        ('resnet50', 'flat'),
        ('convnext_tiny', 'flat'),
        ('resnet18', 'hierarchical'),
        ('resnet50', 'hierarchical'),
        ('convnext_tiny', 'hierarchical'),
    ]
    
    for model_type, architecture in test_cases:
        try:
            print(f"\nTesting {architecture}_{model_type}...")
            
            # Create model
            if architecture == 'flat':
                model = create_flat_model(model_type, pretrained=False)
                # Test with single sweep input (B, T, C, H, W)
                x = torch.randn(batch_size, frames, channels, height, width)
                output, attn = model(x)
                print(f"  Single-sweep input: {x.shape} -> output: {output.shape}, attn: {attn.shape}")
                
                # Test with multi-sweep input (B, S, T, C, H, W)
                x_multi = torch.randn(batch_size, sweeps, frames, channels, height, width)
                output_multi, attn_multi = model(x_multi)
                print(f"  Multi-sweep input: {x_multi.shape} -> output: {output_multi.shape}, attn: {attn_multi.shape}")
                
            else:  # hierarchical
                model = create_hierarchical_model(model_type, pretrained=False)
                # Hierarchical models only accept multi-sweep input
                x = torch.randn(batch_size, sweeps, frames, channels, height, width)
                output, attn = model(x)
                print(f"  Multi-sweep input: {x.shape} -> output: {output.shape}")
                
                if isinstance(attn, dict):
                    print(f"  Attention dict keys: {list(attn.keys())}")
                    for k, v in attn.items():
                        print(f"    {k}: {v.shape}")
                else:
                    print(f"  Attention shape: {attn.shape}")
            
            print(f"  ✅ {architecture}_{model_type} works!")
            
        except Exception as e:
            print(f"  ❌ {architecture}_{model_type} failed: {e}")


if __name__ == "__main__":
    # Run tests
    test_all_models()