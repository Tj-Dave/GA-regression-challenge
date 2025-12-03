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


# ----------------------------
# Attention module (shared by all models)
# ----------------------------
class WeightedAverageAttention(nn.Module):
    """
    Computes a weighted average of temporal features using learned attention scores.
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


# ----------------------------
# Base model class (shared functionality)
# ----------------------------
class BaseGAModel(nn.Module):
    """
    Base class for all GA prediction models.
    Handles the common forward pass logic.
    """
    def __init__(self, feature_extractor, feature_dim, reduced_dim=128, 
                 fine_tune_backbone=True):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_dim = feature_dim
        
        # Optionally freeze backbone
        if not fine_tune_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        self.attention = WeightedAverageAttention(feature_dim=feature_dim, reduced_dim=reduced_dim)
        self.fc = nn.Linear(reduced_dim, 1)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, C, H, W)
        Returns:
            output: Predicted value (B, 1)
            attn_weights: Temporal attention weights (B, T)
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)
        
        # Handle different backbone output shapes
        if features.dim() == 4:  # (B*T, D, 1, 1) for ResNet
            features = features.view(B, T, self.feature_dim)
        elif features.dim() == 2:  # (B*T, D) for ConvNeXt
            features = features.view(B, T, self.feature_dim)
        else:
            features = features.view(B, T, -1)
        
        aggregated, attn_weights = self.attention(features)
        output = self.fc(aggregated)
        return output, attn_weights


# ----------------------------
# ResNet18 Model (Original NEJM baseline)
# ----------------------------
class NEJMbaseline(BaseGAModel):
    """
    ResNet18-based regression model with attention over time (frames).
    """
    def __init__(self, reduced_dim=128, fine_tune_backbone=True, pretrained=True):
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        feature_dim = 512
        
        super().__init__(feature_extractor, feature_dim, reduced_dim, fine_tune_backbone)


# ----------------------------
# ResNet50 Model
# ----------------------------
class ResNet50GA(BaseGAModel):
    """
    ResNet50-based regression model with attention over time (frames).
    More powerful than ResNet18 with 2048-dimensional features.
    """
    def __init__(self, reduced_dim=128, fine_tune_backbone=True, pretrained=True):
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        feature_dim = 2048  # ResNet50 has 2048-dimensional features
        
        super().__init__(feature_extractor, feature_dim, reduced_dim, fine_tune_backbone)


# ----------------------------
# ConvNeXt-Tiny Model
# ----------------------------
class ConvNeXtTinyGA(BaseGAModel):
    """
    ConvNeXt-Tiny-based regression model with attention over time (frames).
    Modern architecture with 768-dimensional features.
    """
    def __init__(self, reduced_dim=128, fine_tune_backbone=True, pretrained=True):
        if not CONVNEXT_AVAILABLE:
            raise ImportError(
                "ConvNeXt requires torchvision >= 0.13. "
                "Install with: pip install torchvision>=0.13 or use Conda"
            )
        
        convnext = convnext_tiny(
            weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        )
        
        # ConvNeXt has different architecture than ResNet
        # We need to extract features properly
        feature_extractor = nn.Sequential(
            convnext.features,  # All convolutional layers
            convnext.avgpool,   # Global average pooling
            nn.Flatten(1),      # Flatten to (B*T, 768)
        )
        feature_dim = 768  # ConvNeXt-Tiny feature dimension
        
        super().__init__(feature_extractor, feature_dim, reduced_dim, fine_tune_backbone)


# ----------------------------
# Model Factory (easy creation)
# ----------------------------
def create_model(model_type='resnet18', **kwargs):
    """
    Factory function to create different model types easily.
    
    Args:
        model_type: 'resnet18', 'resnet50', or 'convnext_tiny'
        **kwargs: Passed to model constructor (reduced_dim, fine_tune_backbone, etc.)
    
    Returns:
        Instantiated model
    
    Example:
        model = create_model('resnet50', reduced_dim=256, fine_tune_backbone=True)
        model = create_model('convnext_tiny', pretrained=True)
    """
    model_registry = {
        'resnet18': NEJMbaseline,
        'resnet50': ResNet50GA,
        'convnext_tiny': ConvNeXtTinyGA,
    }
    
    if model_type not in model_registry:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(model_registry.keys())}"
        )
    
    return model_registry[model_type](**kwargs)


# ----------------------------
# Quick test function
# ----------------------------
def test_models():
    """Test all models with dummy input"""
    batch_size = 2
    frames = 16
    channels = 3
    height = width = 224
    
    # Create dummy input
    x = torch.randn(batch_size, frames, channels, height, width)
    print(f"Input shape: {x.shape}")
    
    # Test each model
    for model_name in ['resnet18', 'resnet50', 'convnext_tiny']:
        try:
            print(f"\nTesting {model_name}...")
            model = create_model(model_name, pretrained=False)  # Use False for faster testing
            output, attention = model(x)
            print(f"  Output shape: {output.shape}")
            print(f"  Attention shape: {attention.shape}")
            print(f"  ✅ {model_name} works!")
        except Exception as e:
            print(f"  ❌ {model_name} failed: {e}")


if __name__ == "__main__":
    # Run tests if file is executed directly
    test_models()