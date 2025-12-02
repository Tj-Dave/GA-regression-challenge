# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# convnext import (torchvision)
try:
    # torchvision > 0.13 style
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
except Exception:
    convnext_tiny = None
    ConvNeXt_Tiny_Weights = None


# ----------------------------
# Intra-Sweep Multi-Head Attention (vectorized)
# ----------------------------
class IntraSweepAttention(nn.Module):
    """
    Multi-head attention summarizing frames within each sweep.
    Input: (B, S, T, D)
    Output:
        sweep_embeddings: (B, S, reduced_dim)
        frame_attentions: (B, S, T)
    Implementation details:
        - Vectorizes across B and S by merging them into one batch dimension when calling nn.MultiheadAttention.
    """
    def __init__(self, feature_dim=512, reduced_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.reduced_dim = reduced_dim
        self.num_heads = num_heads

        # Multi-head attention (expects (batch, seq, embed_dim) when batch_first=True)
        self.mha = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True, dropout=dropout)

        # Simple feedforward after attention to reduce to reduced_dim
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, reduced_dim)
        )

        # Normalization layers
        self.norm_attn = nn.LayerNorm(feature_dim)
        self.norm_ffn = nn.LayerNorm(reduced_dim)

        # Learnable query vector used to summarize the sequence of frames per sweep
        # Shape: (1, 1, feature_dim) so it can be repeated for batch
        self.query = nn.Parameter(torch.randn(1, 1, feature_dim))

    def forward(self, features):
        """
        Args:
            features: (B, S, T, D)  -- D == feature_dim
        Returns:
            sweep_embeddings: (B, S, reduced_dim)
            frame_attentions: (B, S, T)
        """
        B, S, T, D = features.shape
        assert D == self.feature_dim, f"Expected feature_dim={self.feature_dim}, got {D}"

        # Merge B and S for efficient parallel processing
        merged = features.view(B * S, T, D)             # (B*S, T, D)

        # Create query repeated for (B*S)
        q = self.query.repeat(B * S, 1, 1)              # (B*S, 1, D)

        # Multi-head attention: query, key, value
        # attn_out: (B*S, 1, D), attn_weights: (B*S, 1, T)
        attn_out, attn_weights = self.mha(q, merged, merged)  

        # Squeeze and normalize
        attn_out = attn_out.squeeze(1)                  # (B*S, D)
        attn_out = self.norm_attn(attn_out)             # (B*S, D)

        # Feed-forward to reduced_dim
        sweep_emb = self.ffn(attn_out)                  # (B*S, reduced_dim)
        sweep_emb = self.norm_ffn(sweep_emb)            # (B*S, reduced_dim)

        # Reshape back to (B, S, reduced_dim)
        sweep_embeddings = sweep_emb.view(B, S, self.reduced_dim)

        # attn_weights -> (B*S, T) -> reshape to (B, S, T)
        frame_attentions = attn_weights.squeeze(1).view(B, S, T)

        return sweep_embeddings, frame_attentions


# ----------------------------
# MLP Regression Head
# ----------------------------
class MLPRegressionHead(nn.Module):
    """
    Simple MLP head for regression.
    Input: (B, D_in) -> Output: (B, 1)
    """
    def __init__(self, input_dim=128, hidden_dims=(256, 128), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            # Use LayerNorm to be stable across small batches (preferred for representation embeddings)
            layers.append(nn.LayerNorm(h))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# ----------------------------
# Hierarchical (Intra-Sweep) GA Model (Optimized)
# ----------------------------
class HierarchicalGA(nn.Module):
    """
    Hierarchical model using per-frame backbone -> intra-sweep attention -> sweep aggregation -> regression.

    Expected input shape for forward:
        x: (B, S, T, C, H, W)
    Returns:
        output: (B, 1)
        optionally attention dict with:
            'frame_attention': (B, S, T)
            'sweep_embeddings': (B, S, reduced_dim)
    """
    def __init__(self,
                 backbone_type='convnext_tiny',
                 feature_dim=None,
                 reduced_dim=128,
                 num_heads=4,
                 fine_tune_backbone=True,
                 pretrained=True):
        super().__init__()

        # Build backbone
        if backbone_type == 'convnext_tiny':
            if convnext_tiny is None:
                raise RuntimeError("convnext_tiny not available in this torchvision version.")
            backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
            # Create a feature extractor that outputs a flat vector per image: (B, 768)
            self.feature_extractor = nn.Sequential(
                backbone.features,
                backbone.avgpool,
                nn.Flatten(1)
            )
            resolved_feature_dim = 768
        elif backbone_type == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            # Keep everything up to avgpool, then flatten
            self.feature_extractor = nn.Sequential(
                *list(resnet.children())[:-1],  # includes avgpool
                nn.Flatten(1)
            )
            resolved_feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")

        # Allow caller to override feature_dim if needed
        self.feature_dim = feature_dim or resolved_feature_dim

        # Freeze backbone if requested
        if not fine_tune_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        # Intra-sweep attention (vectorized)
        self.intra_attn = IntraSweepAttention(
            feature_dim=self.feature_dim,
            reduced_dim=reduced_dim,
            num_heads=num_heads
        )

        # Regression head
        self.reg_head = MLPRegressionHead(input_dim=reduced_dim, hidden_dims=(256, 128), dropout=0.2)

    def forward(self, x, return_attention=False):
        """
        x: (B, S, T, C, H, W)
        """
        if x.ndim != 6:
            raise ValueError(f"Expected input of shape (B,S,T,C,H,W), got {x.shape}")

        B, S, T, C, H, W = x.shape

        # Flatten frames for backbone: (B*S*T, C, H, W)
        frames = x.view(B * S * T, C, H, W)

        # Extract per-frame features: (B*S*T, feature_dim)
        feat_flat = self.feature_extractor(frames)
        feat_flat = feat_flat.view(B * S * T, -1)  # ensure shape

        # Reshape back to (B, S, T, D)
        features = feat_flat.view(B, S, T, -1)

        # Intra-sweep attention: vectorized
        sweep_embeddings, frame_attention = self.intra_attn(features)  # (B, S, reduced_dim), (B, S, T)

        # Simple inter-sweep aggregation: mean (fast, stable)
        # (You can replace this mean with a learned aggregator or attention later)
        study_embedding = sweep_embeddings.mean(dim=1)  # (B, reduced_dim)

        # Regression
        output = self.reg_head(study_embedding)  # (B, 1)

        if return_attention:
            attn_dict = {
                'frame_attention': frame_attention,   # (B, S, T)
                'sweep_embeddings': sweep_embeddings  # (B, S, reduced_dim)
            }
            return output, attn_dict

        return output, None


# ----------------------------
# Legacy/compatibility classes (optional)
# ----------------------------
class WeightedAverageAttention(nn.Module):
    """Original single-head attention kept for backward compatibility."""
    def __init__(self, feature_dim=512, reduced_dim=128):
        super().__init__()
        self.W = nn.Linear(feature_dim, 64)
        self.V = nn.Linear(64, 1)
        self.Q = nn.Linear(feature_dim, reduced_dim)

    def forward(self, features):
        attn_scores = self.V(torch.tanh(self.W(features)))
        attn_weights = F.softmax(attn_scores, dim=1)
        reduced_features = self.Q(features)
        weighted_sum = torch.sum(attn_weights * reduced_features, dim=1)
        return weighted_sum, attn_weights.squeeze(-1)


class NEJMbaseline(nn.Module):
    """Original ResNet18-based baseline kept for compatibility."""
    def __init__(self, reduced_dim=128, fine_tune_backbone=True, pretrained=True):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten(1))
        self.feature_dim = 512

        if not fine_tune_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.attention = WeightedAverageAttention(feature_dim=self.feature_dim, reduced_dim=reduced_dim)
        self.fc = nn.Linear(reduced_dim, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)
        features = features.view(B, T, self.feature_dim)
        aggregated, attn_weights = self.attention(features)
        output = self.fc(aggregated)
        return output, attn_weights
