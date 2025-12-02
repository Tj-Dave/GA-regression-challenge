import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ----------------------------
# Attention module
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
# Model definition
# ----------------------------
class NEJMbaseline(nn.Module):
    """
    ResNet18-based regression model with attention over time (frames).
    """
    def __init__(self, reduced_dim=128, fine_tune_backbone=True, pretrained=True):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512

        # Optionally freeze backbone
        if not fine_tune_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.attention = WeightedAverageAttention(feature_dim=self.feature_dim, reduced_dim=reduced_dim)
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
        features = self.feature_extractor(x)        # (B*T, 512, 1, 1)
        features = features.view(B, T, self.feature_dim)
        aggregated, attn_weights = self.attention(features)
        output = self.fc(aggregated)
        return output, attn_weights

