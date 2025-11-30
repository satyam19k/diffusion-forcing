"""
Latent Predictor for JEPA-style VAE training.
Predicts future latent representations from current latent + action.
"""

import torch
import torch.nn as nn


class LatentPredictor(nn.Module):
    """
    Predicts next latent from current latent + action using Conv + MLP.
    Preserves spatial structure of the latent representation.
    
    Input:  z_t (B, latent_channels, H, W), a_t (B, action_dim)
    Output: z_t1_pred (B, latent_channels, H, W)
    """
    
    def __init__(
        self,
        latent_channels: int = 4,
        latent_size: int = 32,
        action_dim: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Action embedding: MLP that projects action to spatial feature map
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_channels * latent_size * latent_size),
        )
        
        # Conv layers to combine latent + action and predict residual
        self.conv = nn.Sequential(
            nn.Conv2d(latent_channels * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, latent_channels, kernel_size=3, padding=1),
        )
        
        # Initialize final conv layer with small weights for stable residual learning
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)
    
    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """
        Predict next latent from current latent and action.
        
        Args:
            z_t: Current latent, shape (B, latent_channels, H, W)
            a_t: Current action, shape (B, action_dim)
            
        Returns:
            z_t1_pred: Predicted next latent, shape (B, latent_channels, H, W)
        """
        B, C, H, W = z_t.shape
        
        # Project action to spatial feature map
        a_spatial = self.action_mlp(a_t)  # (B, C * H * W)
        a_spatial = a_spatial.view(B, self.latent_channels, H, W)  # (B, C, H, W)
        
        # Concatenate latent and action features
        combined = torch.cat([z_t, a_spatial], dim=1)  # (B, 2*C, H, W)
        
        # Predict residual and add to current latent
        delta = self.conv(combined)  # (B, C, H, W)
        z_t1_pred = z_t + delta  # Residual prediction
        
        return z_t1_pred

