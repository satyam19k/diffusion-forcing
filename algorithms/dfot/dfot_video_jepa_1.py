"""
Dual-Encoder JEPA for DFoT (BYOL-Style) — ConvPredictor variant.

Changes from dfot_video_jepa.py:
- **ConvPredictor**: Replaces the ViT Predictor with a hybrid conv + attention
  predictor that operates in native (B, D, H, W) image format.  Each block
  combines depthwise separable convolutions for local spatial mixing with
  self-attention for global mixing and optional cross-attention or AdaLN-FiLM
  for action conditioning.
- **NewActionEncoder**: Replaces the simple MLP ActionEncoder with a
  transformer-based encoder that uses AdaLayerNorm conditioning and produces
  per-token embeddings + a CLS summary vector, matching the interface expected
  by ConvPredictor.

Architecture:
- **Target encoder** (frozen): Provides stable latents for DFoT diffusion.
- **Online encoder** (trainable): A copy of the VAE encoder that receives
  JEPA gradients.
- **ConvPredictor**: Operates on *clean* spatial latent states from the online
  encoder and encoded actions; predicts future clean latent states in (B,C,H,W)
  format.
- **EMA sync**: The online encoder is slowly blended into the target encoder.

Gradient paths:
  DFoT loss  -->  diffusion backbone only
  JEPA loss  -->  online encoder + predictor (includes action encoder)
  JEPA decoder loss  -->  online decoder only (latents detached, no grad to encoder/predictor)
  JEPA target: target encoder latents (reused from diffusion, no extra encoding)
  EMA sync: online encoder -> target encoder, online decoder -> target decoder (same schedule)
"""

import logging
import math
from copy import deepcopy
from typing import Optional, Any, Dict, Tuple

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from lightning.pytorch.utilities.types import STEP_OUTPUT
from einops import rearrange, repeat
from transformers import get_scheduler

from algorithms.vae import ImageVAE
from algorithms.vae.common.losses.lpips import LPIPS
from utils.torch_utils import freeze_model
from utils.distributed_utils import rank_zero_print
from utils.print_utils import cyan
from .dfot_video import DFoTVideo

# Import official LeJEPA implementation
try:
    import lejepa
    USE_LEJEPA = True
except ImportError:
    from .sigreg import SigREG
    USE_LEJEPA = False


# =============================================================================
# Helper: modulate (AdaLN utility)
# =============================================================================

def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


# =============================================================================
# AdaLayerNorm
# =============================================================================

class AdaLayerNorm(nn.Module):
    """Adaptive layer norm (AdaLN) — zero-initialized."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.modulation(c).chunk(2, dim=-1)
        return modulate(self.norm(x), shift, scale)


class AdaLayerNormZero(nn.Module):
    """Adaptive layer norm zero (AdaLN-Zero) — returns (normed_x, gate)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True),
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
        shift, scale, gate = self.modulation(c).chunk(3, dim=-1)
        return modulate(self.norm(x), shift, scale), gate


# =============================================================================
# Action Encoders (for ConvPredictor interface)
# =============================================================================

class ActionEncoder(nn.Module):
    """
    Simple action encoder for ConvPredictor.

    Takes a dict with:
        "actions": (B, L, action_dim)
        "mask":    (B, L) bool — True = valid token
    Returns:
        tokens_out: (B, L, d_model)
        cls_out:    (B, d_model)
        key_padding_mask_tokens: (B, L) bool — True = PAD (inverted mask)
    """

    def __init__(
        self,
        action_dim: int,
        d_model: int,
        num_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.0,
        max_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.proj = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Standard transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, actions: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        action_seq = actions["actions"]  # (B, L, action_dim)
        mask = actions.get("mask", None)  # (B, L) bool, True = valid

        B, L, _ = action_seq.shape

        # Project
        tokens = self.proj(action_seq)  # (B, L, d_model)
        tokens = tokens + self.pos_embed[:, :L, :]

        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, L+1, d_model)

        # Build key_padding_mask for transformer: True = ignore
        src_key_padding_mask = None
        if mask is not None:
            cls_valid = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([cls_valid, mask], dim=1)  # (B, L+1)
            src_key_padding_mask = ~full_mask  # True = PAD

        # Encode
        tokens = self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        tokens = self.final_norm(tokens)

        # Split CLS and token outputs
        cls_out = tokens[:, 0]  # (B, d_model)
        tokens_out = tokens[:, 1:]  # (B, L, d_model)

        # Key padding mask for downstream cross-attention (True = PAD)
        kpm = None
        if mask is not None:
            kpm = ~mask  # (B, L)

        return tokens_out, cls_out, kpm


class NewActionEncoder(nn.Module):
    """
    Action encoder with AdaLayerNorm conditioning for ConvPredictor.

    Each transformer layer uses AdaLayerNormZero conditioned on the running
    CLS representation, giving the encoder adaptive modulation.

    Interface identical to ActionEncoder:
        forward(actions: dict) -> (tokens_out, cls_out, key_padding_mask)
    """

    def __init__(
        self,
        action_dim: int,
        d_model: int,
        num_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.0,
        max_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # Input projection
        self.proj = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # AdaLN transformer layers
        self.attn_norms = nn.ModuleList([AdaLayerNormZero(d_model) for _ in range(num_layers)])
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([AdaLayerNormZero(d_model) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, actions: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        action_seq = actions["actions"]  # (B, L, action_dim)
        mask = actions.get("mask", None)  # (B, L) bool, True = valid

        B, L, _ = action_seq.shape

        # Project
        tokens = self.proj(action_seq)  # (B, L, d_model)
        tokens = tokens + self.pos_embed[:, :L, :]

        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, tokens], dim=1)  # (B, L+1, d_model)

        # Build key_padding_mask: True = ignore
        attn_mask = None
        if mask is not None:
            cls_valid = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([cls_valid, mask], dim=1)  # (B, L+1)
            attn_mask = ~full_mask  # True = PAD

        # AdaLN transformer layers — CLS token is the conditioning signal
        for i in range(self.num_layers):
            # Use current CLS as conditioning for AdaLN
            c = x[:, 0:1].expand_as(x)  # (B, L+1, d_model) — broadcast CLS

            # Self-attention with AdaLN-Zero
            x_normed, gate_attn = self.attn_norms[i](x, c)
            attn_out, _ = self.attns[i](
                x_normed, x_normed, x_normed,
                key_padding_mask=attn_mask,
            )
            x = x + gate_attn * attn_out

            # FFN with AdaLN-Zero
            c = x[:, 0:1].expand_as(x)  # refresh conditioning
            x_normed, gate_ffn = self.ffn_norms[i](x, c)
            x = x + gate_ffn * self.ffns[i](x_normed)

        x = self.final_norm(x)

        # Split CLS and token outputs
        cls_out = x[:, 0]  # (B, d_model)
        tokens_out = x[:, 1:]  # (B, L, d_model)

        # Key padding mask for downstream cross-attention (True = PAD)
        kpm = None
        if mask is not None:
            kpm = ~mask  # (B, L)

        return tokens_out, cls_out, kpm


# =============================================================================
# ConvPredictor Components
# =============================================================================

# Import RoPE utilities lazily to avoid hard dependency at module level.
_rope_fns = None


def _get_rope_fns():
    global _rope_fns
    if _rope_fns is None:
        from wm.model import RotaryType, apply_rope_nd
        _rope_fns = (apply_rope_nd, RotaryType)
    return _rope_fns


class ConvPredictorBlock(nn.Module):
    """
    Hybrid conv + attention block operating in (B, D, H, W) image format.

    Sub-layers (all pre-norm with residual connections):
      1. Depthwise separable conv  — local spatial + channel mixing
      2. Self-attention             — global spatial mixing (optional 2D RoPE)
      3. Cross-attention            — action conditioning (cross_attn mode only)
      4. FFN via 1×1 convolutions   — channel mixing

    Conditioning modes:
      - cross_attn:  sub-layer 3 present; norms have affine=True
      - adaln_film:  sub-layer 3 absent; FiLM on sub-layers 2 & 4; norms affine=False
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        conditioning: str,
        dropout: float = 0.0,
        mlp_mult: float = 4.0,
        conv_kernel_size: int = 3,
        norm_num_groups: int = 32,
        use_rope: bool = False,
    ):
        super().__init__()
        assert conditioning in ("cross_attn", "adaln_film")
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.conditioning = conditioning
        self.use_rope = use_rope

        adaln = conditioning == "adaln_film"

        # 1. Depthwise separable conv (always unconditioned)
        self.conv_norm = nn.GroupNorm(norm_num_groups, d_model)
        self.dwconv = nn.Conv2d(
            d_model, d_model, conv_kernel_size,
            padding=conv_kernel_size // 2, groups=d_model,
        )
        self.pwconv = nn.Conv2d(d_model, d_model, 1)

        # 2. Self-attention (custom QKV projections for optional RoPE)
        self.attn_norm = nn.GroupNorm(norm_num_groups, d_model, affine=not adaln)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.attn_out_proj = nn.Linear(d_model, d_model)

        # 3. Action conditioning
        if conditioning == "cross_attn":
            self.cross_norm = nn.GroupNorm(norm_num_groups, d_model)
            self.cross_q_proj = nn.Linear(d_model, d_model)
            self.cross_kv_proj = nn.Linear(d_model, 2 * d_model)
            self.cross_out_proj = nn.Linear(d_model, d_model)
        else:
            self.cond_attn = nn.Linear(d_model, 2 * d_model)
            self.cond_ffn = nn.Linear(d_model, 2 * d_model)
            nn.init.zeros_(self.cond_attn.weight)
            nn.init.zeros_(self.cond_attn.bias)
            nn.init.zeros_(self.cond_ffn.weight)
            nn.init.zeros_(self.cond_ffn.bias)

        # 4. FFN (1×1 convolutions)
        self.ffn_norm = nn.GroupNorm(norm_num_groups, d_model, affine=not adaln)
        inner_dim = int(mlp_mult * d_model)
        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, inner_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(inner_dim, d_model, 1),
            nn.Dropout(dropout),
        )

    # -- helpers --------------------------------------------------------------

    def _adaln(
        self,
        x: Tensor,     # (B, D, H, W)
        cond: Tensor,   # (B, D)
        proj: nn.Linear,
    ) -> Tensor:
        gamma_beta = proj(cond)                     # (B, 2D)
        gamma, beta = gamma_beta.chunk(2, dim=-1)   # (B, D) each
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def _self_attention(
        self, h: Tensor, H: int, W: int,
    ) -> Tensor:
        """h: (B, D, H, W) -> (B, D, H, W)"""
        B, D = h.shape[:2]
        N = H * W

        # Flatten to (B, N, D) and project
        h = h.flatten(2).transpose(1, 2)            # (B, N, D)
        q, k, v = self.qkv_proj(h).chunk(3, dim=-1)

        # Multi-head reshape: (B, heads, N, head_dim)
        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            apply_rope_nd, RotaryType = _get_rope_fns()
            q = q.view(B, self.n_heads, H, W, self.head_dim)
            k = k.view(B, self.n_heads, H, W, self.head_dim)
            q, k = apply_rope_nd(q, k, (H, W), RotaryType.PIXEL)
            q = q.reshape(B, self.n_heads, N, self.head_dim)
            k = k.reshape(B, self.n_heads, N, self.head_dim)

        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).reshape(B, N, D)     # (B, N, D)
        h = self.attn_out_proj(h)
        return h.transpose(1, 2).reshape(B, D, H, W)

    def _cross_attention(
        self,
        h: Tensor,                          # (B, D, H, W)
        action_tokens: Tensor,              # (B, L, D)
        action_kpm: Optional[Tensor],       # (B, L) bool, True = PAD
        action_cls: Optional[Tensor] = None,  # (B, D)
    ) -> Tensor:
        B, D, H, W = h.shape
        N = H * W

        # Prepend CLS to K/V sequence
        if action_cls is not None:
            cls_kv = action_cls.unsqueeze(1)  # (B, 1, D)
            action_tokens = torch.cat([cls_kv, action_tokens], dim=1)
            if action_kpm is not None:
                cls_valid = torch.zeros(B, 1, dtype=torch.bool, device=h.device)
                action_kpm = torch.cat([cls_valid, action_kpm], dim=1)

        L = action_tokens.shape[1]

        h_flat = h.flatten(2).transpose(1, 2)       # (B, N, D)
        q = self.cross_q_proj(h_flat)
        kv = self.cross_kv_proj(action_tokens)       # (B, L, 2D)
        k, v = kv.chunk(2, dim=-1)

        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if action_kpm is not None:
            attn_mask = torch.zeros(B, 1, 1, L, dtype=q.dtype, device=q.device)
            attn_mask.masked_fill_(action_kpm[:, None, None, :], float("-inf"))

        h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        h = h.transpose(1, 2).reshape(B, N, D)
        h = self.cross_out_proj(h)
        return h.transpose(1, 2).reshape(B, D, H, W)

    # -- forward --------------------------------------------------------------

    def forward(
        self,
        x: Tensor,                                      # (B, D, H, W)
        action_tokens: Optional[Tensor] = None,          # (B, L, D)
        action_cls: Optional[Tensor] = None,             # (B, D)
        action_kpm: Optional[Tensor] = None,             # (B, L) bool
    ) -> Tensor:
        B, D, H, W = x.shape

        # 1. Depthwise separable conv
        h = self.conv_norm(x)
        h = self.pwconv(F.silu(self.dwconv(h)))
        x = x + h

        # 2. Self-attention
        h = self.attn_norm(x)
        if self.conditioning == "adaln_film":
            h = self._adaln(h, action_cls, self.cond_attn)
        x = x + self._self_attention(h, H, W)

        # 3. Cross-attention (cross_attn mode only)
        if self.conditioning == "cross_attn":
            h = self.cross_norm(x)
            x = x + self._cross_attention(h, action_tokens, action_kpm, action_cls)

        # 4. FFN
        h = self.ffn_norm(x)
        if self.conditioning == "adaln_film":
            h = self._adaln(h, action_cls, self.cond_ffn)
        x = x + self.ffn(h)

        return x


class ConvPredictor(nn.Module):
    """
    Hybrid conv + attention predictor operating in (B, D, H, W) image format.

    forward(z: (B, C, H, W), actions: dict) -> z_pred: (B, C, H, W)

    Key features:
      - Each block includes a depthwise separable conv for local spatial mixing.
      - Uses GroupNorm (2-D native) instead of LayerNorm.
      - Positional embedding: "rope" (2-D RoPE), "learned" (row+col), or "none".
    """

    def __init__(
        self,
        latent_channels: int,
        action_dim: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.0,
        max_actions: int = 64,
        mlp_mult: float = 4.0,
        conditioning: str = "cross_attn",
        action_encoder_layers: int = 2,
        max_H: int = 32,
        max_W: int = 32,
        identity_init: bool = False,
        pos_emb: str = "rope",
        conv_kernel_size: int = 3,
        norm_num_groups: int = 32,
        action_enc_type: str = "new",
    ):
        super().__init__()
        assert conditioning in ("cross_attn", "adaln_film")
        assert pos_emb in ("rope", "learned", "none")
        assert action_enc_type in ("old", "new")
        self.conditioning = conditioning
        self.d_model = d_model
        self.identity_init = identity_init
        self.pos_emb = pos_emb
        self.max_H = max_H
        self.max_W = max_W

        # Input / output projections
        self.in_proj = nn.Conv2d(latent_channels, d_model, 1)
        self.out_proj = nn.Conv2d(d_model, latent_channels, 1)

        # Positional embeddings (learned mode only; RoPE is inside blocks)
        if pos_emb == "learned":
            self.row_embed = nn.Embedding(max_H, d_model)
            self.col_embed = nn.Embedding(max_W, d_model)

        # Action encoder
        enc_cls = NewActionEncoder if action_enc_type == "new" else ActionEncoder
        self.action_encoder = enc_cls(
            action_dim=action_dim,
            d_model=d_model,
            num_layers=action_encoder_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_len=max_actions,
        )

        # Main body
        use_rope = pos_emb == "rope"
        self.blocks = nn.ModuleList([
            ConvPredictorBlock(
                d_model=d_model,
                n_heads=n_heads,
                conditioning=conditioning,
                dropout=dropout,
                mlp_mult=mlp_mult,
                conv_kernel_size=conv_kernel_size,
                norm_num_groups=norm_num_groups,
                use_rope=use_rope,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.GroupNorm(norm_num_groups, d_model)

        if identity_init:
            nn.init.zeros_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        z: Tensor,  # (B, C, H, W)
        actions: Dict[str, Tensor],
    ) -> Tensor:
        B, C, H, W = z.shape

        # 1. Encode actions
        action_tokens, action_cls, action_kpm = self.action_encoder(actions)

        # 2. Project to d_model
        x = self.in_proj(z)  # (B, D, H, W)

        # 3. Optional learned positional embeddings
        if self.pos_emb == "learned":
            if H > self.max_H or W > self.max_W:
                logging.warning(
                    f"ConvPredictor input H,W=({H},{W}) exceeds "
                    f"max_H,max_W=({self.max_H},{self.max_W})."
                )
            rows = torch.arange(H, device=z.device)
            cols = torch.arange(W, device=z.device)
            pos = self.row_embed(rows)[:, None, :] + self.col_embed(cols)[None, :, :]
            x = x + pos.permute(2, 0, 1).unsqueeze(0)

        # 4. Blocks
        for blk in self.blocks:
            x = blk(
                x,
                action_tokens=action_tokens,
                action_cls=action_cls,
                action_kpm=action_kpm,
            )

        # 5. Post-process
        x = self.final_norm(x)
        z_pred = self.out_proj(x)

        if self.identity_init:
            z_pred = z + z_pred

        return z_pred


# =============================================================================
# Main Algorithm
# =============================================================================

class DFoTVideoJEPA1(DFoTVideo):
    """
    DFoT with dual-encoder JEPA (BYOL-style) — ConvPredictor variant.

    - Target encoder (self.vae) is frozen and provides stable latents for DFoT.
    - Online encoder (self.online_encoder + self.online_quant_conv) is trainable
      and produces clean latents for the JEPA predictor.
    - ConvPredictor operates in native spatial (B, C, H, W) format with hybrid
      conv + attention blocks.
    - EMA periodically blends online -> target so DFoT gradually benefits.
    """

    def __init__(self, cfg: DictConfig):
        self.jepa_cfg = cfg.jepa
        self.jepa_loss_weight = cfg.jepa.loss_weight
        self.jepa_training_mode = cfg.jepa.get("training_mode", "teacher_forcing")

        # SigREG config
        self.prediction_loss_weight = cfg.jepa.get("prediction_loss_weight", 1.0)
        self.sigreg_loss_weight = cfg.jepa.get("sigreg_loss_weight", 1.0)
        self.sigreg_num_slices = cfg.jepa.get("sigreg_num_slices", 1024)
        self.sigreg_proj_dim = cfg.jepa.get("sigreg_proj_dim", 256)

        # Cosine similarity loss weight
        self.lambda_cos = cfg.jepa.get("lambda_cos", 0.5)

        # Decoder JEPA config
        self.decoder_loss_weight = cfg.jepa.get("decoder_loss_weight", 1.0)
        self.decoder_pred_loss_weight = cfg.jepa.get("decoder_pred_loss_weight", 1.0)
        self.decoder_enc_loss_weight = cfg.jepa.get("decoder_enc_loss_weight", 1.0)
        
        # Norm regularization: prevent latent collapse
        self.norm_reg_weight = cfg.jepa.get("norm_reg_weight", 0.1)  # L2 norm penalty on encoder latents
        self.target_norm_reg_weight = cfg.jepa.get("target_norm_reg_weight", 0.05)  # Regularize target encoder norms
        self.target_norm_mean_target = cfg.jepa.get("target_norm_mean_target", 1.0)  # Target magnitude for latents

        # EMA config
        self.ema_decay = cfg.jepa.get("ema_decay", 0.99)
        self.ema_update_every = cfg.jepa.get("ema_update_every", 100)
        print(cyan(f"JEPA EMA decay: {self.ema_decay}, update every: {self.ema_update_every} steps"))

        # Progressive unfreezing
        self.encoder_unfreeze_step = cfg.jepa.get("encoder_unfreeze_step", None)
        self.dit_unfreeze_step = cfg.jepa.get("dit_unfreeze_step", None)
        print(cyan(f"JEPA encoder unfreeze step: {self.encoder_unfreeze_step}, DIT unfreeze step: {self.dit_unfreeze_step}"))
        self._encoder_unfrozen = False
        self._dit_unfrozen = False

        # Fixed probe videos for visualisation
        self._jepa_vis_videos: Optional[Tensor] = None
        self._jepa_vis_actions: Optional[Tensor] = None
        self._jepa_num_vis: int = int(cfg.jepa.get("num_vis_videos", 3))

        # ConvPredictor config
        self._conv_pred_cfg = {
            "d_model": cfg.jepa.get("conv_pred_d_model", 512),
            "n_heads": cfg.jepa.get("conv_pred_n_heads", 8),
            "num_layers": cfg.jepa.get("conv_pred_num_layers", 6),
            "dropout": cfg.jepa.get("conv_pred_dropout", 0.0),
            "mlp_mult": cfg.jepa.get("conv_pred_mlp_mult", 4.0),
            "conditioning": cfg.jepa.get("conv_pred_conditioning", "cross_attn"),
            "action_encoder_layers": cfg.jepa.get("conv_pred_action_enc_layers", 2),
            "identity_init": cfg.jepa.get("conv_pred_identity_init", True),
            "pos_emb": cfg.jepa.get("conv_pred_pos_emb", "learned"),
            "conv_kernel_size": cfg.jepa.get("conv_pred_kernel_size", 3),
            "norm_num_groups": cfg.jepa.get("conv_pred_norm_groups", 32),
            "action_enc_type": cfg.jepa.get("conv_pred_action_enc_type", "new"),
        }

        # Force online latent mode
        assert cfg.latent.enable, "JEPA training requires latent diffusion"
        assert cfg.latent.type == "online", "JEPA training requires online latent processing"

        super().__init__(cfg)

    # -----------------------------------------------------------------
    # Model building
    # -----------------------------------------------------------------

    def _build_model(self):
        """Build DFoT model, then build JEPA components."""
        super()._build_model()
        self._build_jepa_model()

    def _build_jepa_model(self):
        """Build ConvPredictor (includes action encoder internally)."""
        jepa_cfg = self.jepa_cfg

        # Latent spatial dimensions
        latent_channels = self.x_shape[0]
        latent_h = self.x_shape[1]
        latent_w = self.x_shape[2] if len(self.x_shape) > 2 else latent_h
        self.state_dim = latent_channels * latent_h * latent_w

        # ConvPredictor (action encoder is built inside)
        self.predictor = ConvPredictor(
            latent_channels=latent_channels,
            action_dim=self.external_cond_dim,
            d_model=self._conv_pred_cfg["d_model"],
            n_heads=self._conv_pred_cfg["n_heads"],
            num_layers=self._conv_pred_cfg["num_layers"],
            dropout=self._conv_pred_cfg["dropout"],
            max_actions=self.max_tokens + 1,
            mlp_mult=self._conv_pred_cfg["mlp_mult"],
            conditioning=self._conv_pred_cfg["conditioning"],
            action_encoder_layers=self._conv_pred_cfg["action_encoder_layers"],
            max_H=latent_h,
            max_W=latent_w,
            identity_init=self._conv_pred_cfg["identity_init"],
            pos_emb=self._conv_pred_cfg["pos_emb"],
            conv_kernel_size=self._conv_pred_cfg["conv_kernel_size"],
            norm_num_groups=self._conv_pred_cfg["norm_num_groups"],
            action_enc_type=self._conv_pred_cfg["action_enc_type"],
        )

        # SigREG — anti-collapse regularization on online encoder embeddings
        if USE_LEJEPA:
            try:
                univariate_test = lejepa.univariate.EppsPulley()
                self.sigreg = lejepa.multivariate.SlicingUnivariateTest(
                    univariate_test=univariate_test,
                    num_slices=self.sigreg_num_slices,
                )
                rank_zero_print(cyan("Using official LeJEPA SigREG implementation"))
            except Exception as e:
                rank_zero_print(cyan(f"LeJEPA init failed ({e}), using custom SigREG"))
                self.sigreg = SigREG(num_slices=self.sigreg_num_slices)
        else:
            self.sigreg = SigREG(num_slices=self.sigreg_num_slices)
            rank_zero_print(cyan("Using custom SigREG implementation (LeJEPA not installed)"))

        # SigREG projection
        if self.sigreg_proj_dim > 0:
            self.sigreg_proj = nn.Sequential(
                nn.Linear(self.state_dim, self.sigreg_proj_dim),
                nn.LayerNorm(self.sigreg_proj_dim),
                nn.GELU(),
                nn.Linear(self.sigreg_proj_dim, self.sigreg_proj_dim),
            )
            rank_zero_print(cyan(f"JEPA SigREG projection: {self.state_dim} -> {self.sigreg_proj_dim}"))
        else:
            self.sigreg_proj = None

        rank_zero_print(cyan(f"JEPA State dim: {self.state_dim}"))
        rank_zero_print(cyan(f"JEPA ConvPredictor d_model: {self._conv_pred_cfg['d_model']}, "
                             f"layers: {self._conv_pred_cfg['num_layers']}, "
                             f"conditioning: {self._conv_pred_cfg['conditioning']}, "
                             f"action_enc: {self._conv_pred_cfg['action_enc_type']}"))
        rank_zero_print(cyan(f"JEPA Training mode: {self.jepa_training_mode}"))
        rank_zero_print(cyan(f"JEPA EMA decay: {self.ema_decay}, update every: {self.ema_update_every}"))
        rank_zero_print(cyan(f"JEPA SigREG weight: {self.sigreg_loss_weight}, slices: {self.sigreg_num_slices}"))

        # Perceptual loss for decoder reconstruction
        self.perceptual_loss = LPIPS().eval()
        for p in self.perceptual_loss.parameters():
            p.requires_grad = False

    # -----------------------------------------------------------------
    # VAE loading -- dual encoder
    # -----------------------------------------------------------------

    def _load_vae(self) -> None:
        """
        Load VAE and create the dual-encoder setup.
        """
        assert not self.is_latent_video_vae, "JEPA currently only supports ImageVAE"

        self.vae = ImageVAE.from_pretrained(
            path=self.cfg.vae.pretrained_path,
            **self.cfg.vae.pretrained_kwargs,
        ).to(self.device)

        freeze_model(self.vae)
        rank_zero_print(cyan("Target VAE is fully frozen"))

        self.online_encoder = deepcopy(self.vae.encoder)
        self.online_quant_conv = deepcopy(self.vae.quant_conv)

        for p in self.online_encoder.parameters():
            p.requires_grad = True
        for p in self.online_quant_conv.parameters():
            p.requires_grad = True

        n_online_enc_params = (
            sum(p.numel() for p in self.online_encoder.parameters())
            + sum(p.numel() for p in self.online_quant_conv.parameters())
        )
        rank_zero_print(
            cyan(f"Online encoder created: {n_online_enc_params / 1e6:.1f}M trainable params")
        )

        # --- Online decoder (trainable copy for JEPA decoder loss) ---
        self.online_decoder = deepcopy(self.vae.decoder)
        self.online_post_quant_conv = deepcopy(self.vae.post_quant_conv)

        for p in self.online_decoder.parameters():
            p.requires_grad = True
        for p in self.online_post_quant_conv.parameters():
            p.requires_grad = True

        n_online_dec_params = (
            sum(p.numel() for p in self.online_decoder.parameters())
            + sum(p.numel() for p in self.online_post_quant_conv.parameters())
        )
        rank_zero_print(
            cyan(f"Online decoder created: {n_online_dec_params / 1e6:.1f}M trainable params")
        )

    # -----------------------------------------------------------------
    # Encoding helpers
    # -----------------------------------------------------------------

    def _encode_online(self, videos: Tensor) -> Tensor:
        """
        Encode raw videos with the trainable online encoder.

        Args:
            videos: (B, T, 3, H, W) in [0, 1]
        Returns:
            latents: (B, T, C, H, W)
        """
        B, T = videos.shape[:2]
        x_flat = rearrange(videos, "b t c h w -> (b t) c h w")
        x_normalized = 2.0 * x_flat - 1.0

        h = self.online_encoder(x_normalized)
        moments = self.online_quant_conv(h)
        mean, _ = torch.chunk(moments, 2, dim=1)
        latents_flat = mean

        return latents_flat.view(B, T, *latents_flat.shape[1:])

    def _encode_target(self, videos: Tensor) -> Tensor:
        """
        Encode raw videos with the frozen target encoder.

        Args:
            videos: (B, T, 3, H, W) in [0, 1]
        Returns:
            latents: (B, T, C, H, W)
        """
        B, T = videos.shape[:2]
        x_flat = rearrange(videos, "b t c h w -> (b t) c h w")
        x_normalized = 2.0 * x_flat - 1.0

        with torch.no_grad():
            posterior = self.vae.encode(x_normalized)
            latents_flat = posterior.mode()

        return latents_flat.view(B, T, *latents_flat.shape[1:])

    # -----------------------------------------------------------------
    # Online decoder
    # -----------------------------------------------------------------

    def _decode_online(self, latents: Tensor) -> Tensor:
        """
        Decode latents with the trainable online decoder.

        Args:
            latents: (B*T, C, H, W) latent tensors (should be detached from encoder graph)
        Returns:
            images: (B*T, 3, H, W) in [-1, 1]
        """
        z = self.online_post_quant_conv(latents)
        return self.online_decoder(z)

    # -----------------------------------------------------------------
    # EMA
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _ema_update_target_encoder(self) -> None:
        decay = self.ema_decay
        for p_online, p_target in zip(
            self.online_encoder.parameters(), self.vae.encoder.parameters()
        ):
            p_target.data.mul_(decay).add_(p_online.data, alpha=1.0 - decay)

        for p_online, p_target in zip(
            self.online_quant_conv.parameters(), self.vae.quant_conv.parameters()
        ):
            p_target.data.mul_(decay).add_(p_online.data, alpha=1.0 - decay)

    @torch.no_grad()
    def _ema_update_target_decoder(self) -> None:
        """EMA: online decoder -> target decoder (same schedule as encoder)."""
        decay = self.ema_decay
        for p_online, p_target in zip(
            self.online_decoder.parameters(), self.vae.decoder.parameters()
        ):
            p_target.data.mul_(decay).add_(p_online.data, alpha=1.0 - decay)

        for p_online, p_target in zip(
            self.online_post_quant_conv.parameters(), self.vae.post_quant_conv.parameters()
        ):
            p_target.data.mul_(decay).add_(p_online.data, alpha=1.0 - decay)

    # -----------------------------------------------------------------
    # Progressive Unfreezing
    # -----------------------------------------------------------------

    def _update_component_freezing(self) -> None:
        if self.encoder_unfreeze_step is not None and self.global_step == self.encoder_unfreeze_step:
            if not self._encoder_unfrozen:
                for p in self.online_encoder.parameters():
                    p.requires_grad = True
                for p in self.online_quant_conv.parameters():
                    p.requires_grad = True
                self._encoder_unfrozen = True
                rank_zero_print(
                    cyan(f"Unfroze online encoder at step {self.global_step}")
                )

        if self.dit_unfreeze_step is not None and self.global_step == self.dit_unfreeze_step:
            if not self._dit_unfrozen:
                for p in self.diffusion_model.parameters():
                    p.requires_grad = True
                self._dit_unfrozen = True
                rank_zero_print(
                    cyan(f"Unfroze DIT (diffusion model) at step {self.global_step}")
                )

    # -----------------------------------------------------------------
    # Optimizers
    # -----------------------------------------------------------------

    def configure_optimizers(self):
        """Three param groups: diffusion, JEPA predictor, online encoder."""
        params_groups = []

        # 1. Diffusion model
        dit_params = list(self.diffusion_model.parameters())
        if self.dit_unfreeze_step is not None:
            for p in dit_params:
                p.requires_grad = False
            rank_zero_print(cyan(f"Freezing DIT until step {self.dit_unfreeze_step}"))

        params_groups.append({
            "params": dit_params,
            "lr": self.cfg.lr,
            "name": "diffusion",
        })

        # 2. JEPA predictor (includes action encoder) + sigreg projection
        jepa_params = list(self.predictor.parameters())
        if self.sigreg_proj is not None:
            jepa_params += list(self.sigreg_proj.parameters())
        params_groups.append({
            "params": jepa_params,
            "lr": self.jepa_cfg.lr,
            "name": "jepa",
        })

        # 3. Online encoder
        online_params = (
            list(self.online_encoder.parameters())
            + list(self.online_quant_conv.parameters())
        )
        if self.encoder_unfreeze_step is not None:
            for p in online_params:
                p.requires_grad = False
            rank_zero_print(cyan(f"Freezing online encoder until step {self.encoder_unfreeze_step}"))

        params_groups.append({
            "params": online_params,
            "lr": self.jepa_cfg.get("encoder_lr", 5e-5),
            "name": "online_encoder",
        })

        # 4. Online decoder
        online_dec_params = (
            list(self.online_decoder.parameters())
            + list(self.online_post_quant_conv.parameters())
        )
        params_groups.append({
            "params": online_dec_params,
            "lr": self.jepa_cfg.get("decoder_lr", 5e-5),
            "name": "online_decoder",
        })

        optimizer = torch.optim.AdamW(
            params_groups,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.optimizer_beta,
        )

        lr_scheduler_config = {
            "scheduler": get_scheduler(
                optimizer=optimizer,
                **self.cfg.lr_scheduler,
            ),
            "interval": "step",
            "frequency": 1,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

    # -----------------------------------------------------------------
    # Batch preprocessing
    # -----------------------------------------------------------------

    def on_after_batch_transfer(
        self, batch: Dict, dataloader_idx: int
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Returns 5-element tuple:
            xs, conditions, masks, gt_videos, actions_raw
        """
        gt_videos = batch.get("videos", None)
        actions_raw = batch.get("conds", None)

        if self.is_latent_diffusion and self.is_latent_online:
            xs = self._encode(batch["videos"])
        else:
            xs = batch.get("latents", batch["videos"])

        xs = self._normalize_x(xs)

        conditions = batch.get("conds", None)

        if "masks" in batch:
            masks = batch["masks"]
        else:
            masks = torch.ones(*xs.shape[:2], dtype=torch.bool, device=self.device)

        return xs, conditions, masks, gt_videos, actions_raw

    # -----------------------------------------------------------------
    # Helper: build actions dict for ConvPredictor
    # -----------------------------------------------------------------

    def _make_actions_dict(
        self,
        actions: Tensor,          # (B, L, action_dim)
        mask: Optional[Tensor] = None,  # (B, L) bool
    ) -> Dict[str, Tensor]:
        """Build actions dict expected by ConvPredictor."""
        d = {"actions": actions}
        if mask is not None:
            d["mask"] = mask
        else:
            d["mask"] = torch.ones(
                actions.shape[0], actions.shape[1],
                dtype=torch.bool, device=actions.device,
            )
        return d

    # -----------------------------------------------------------------
    # JEPA Loss
    # -----------------------------------------------------------------

    def _compute_jepa_loss(
        self,
        videos: Tensor,
        actions: Tensor,
        masks: Tensor,
        target_latents_precomputed: Tensor = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute JEPA prediction loss on clean latents from the online encoder.

        Pipeline:
        1. Encode all frames with the online encoder -> clean spatial latents
        2. SigREG on flattened online embeddings
        3. For each timestep t, predict z_{t+1} from z_t + action_t via ConvPredictor
        4. Loss = MSE on normalized latents (pred vs target-encoder latents)
        """
        B, T = videos.shape[:2]

        if T < 2:
            return torch.tensor(0.0, device=videos.device), {}

        # 1. Encode with online encoder (gradients flow through)
        online_latents = self._encode_online(videos)  # (B, T, C, H, W)
        C, latent_h, latent_w = online_latents.shape[2:]

        # 2. SigREG on online embeddings
        states_flat = online_latents.reshape(B * T, -1)  # (B*T, state_dim)
        sigreg_input = states_flat
        if self.sigreg_proj is not None:
            sigreg_input = self.sigreg_proj(sigreg_input)
        sigreg_loss = self.sigreg(sigreg_input)

        # 3. Predict next latent for each timestep
        if self.jepa_training_mode == "autoregressive":
            pred_list = []
            current = online_latents[:, 0]  # (B, C, H, W)
            for t in range(T - 1):
                action_t = actions[:, t:t+1]  # (B, 1, action_dim)
                actions_dict = self._make_actions_dict(action_t)
                pred_t = self.predictor(current, actions_dict)  # (B, C, H, W)
                pred_list.append(pred_t)
                current = pred_t.detach() if t < T - 2 else pred_t
            pred_latents = torch.stack(pred_list, dim=1)  # (B, T-1, C, H, W)
        else:
            # Teacher forcing: predict each step independently
            pred_list = []
            for t in range(T - 1):
                z_t = online_latents[:, t]  # (B, C, H, W)
                action_t = actions[:, t:t+1]  # (B, 1, action_dim)
                actions_dict = self._make_actions_dict(action_t)
                pred_t = self.predictor(z_t, actions_dict)  # (B, C, H, W)
                pred_list.append(pred_t)
            pred_latents = torch.stack(pred_list, dim=1)  # (B, T-1, C, H, W)

        # Use target encoder latents (already computed for diffusion) as JEPA target
        if target_latents_precomputed is not None:
            target_latents = target_latents_precomputed[:, 1:].detach()  # (B, T-1, C, H, W)
        else:
            # Fallback: encode with target encoder if not provided
            target_latents = self._encode_target(videos)[:, 1:].detach()  # (B, T-1, C, H, W)

        # 4. SigREG on predicted embeddings
        pred_flat = pred_latents.reshape(-1, self.state_dim)
        sigreg_input_pred = pred_flat
        if self.sigreg_proj is not None:
            sigreg_input_pred = self.sigreg_proj(sigreg_input_pred)
        sigreg_loss_pred = self.sigreg(sigreg_input_pred)

        # 5. Loss (normalize then MSE)
        states_pred = pred_latents.reshape(B, T - 1, -1)  # (B, T-1, state_dim)
        states_target = target_latents.reshape(B, T - 1, -1)  # (B, T-1, state_dim)
        states_input = online_latents[:, :-1].reshape(B, T - 1, -1)  # for copy baseline

        states_pred_norm = F.normalize(states_pred, dim=-1)
        states_target_norm = F.normalize(states_target, dim=-1)

        transition_masks = masks[:, :-1] & masks[:, 1:]

        pred_loss = F.mse_loss(
            states_pred_norm, states_target_norm, reduction="none"
        ).mean(dim=-1)  # (B, T-1)

        if transition_masks.sum() > 0:
            pred_loss = (
                (pred_loss * transition_masks.float()).sum() / transition_masks.sum()
            )
        else:
            pred_loss = pred_loss.mean()

        # Metrics
        with torch.no_grad():
            copy_pred_norm = F.normalize(states_input, dim=-1)
            copy_loss = F.mse_loss(
                copy_pred_norm, states_target_norm, reduction="none"
            ).mean(dim=-1)
            if transition_masks.sum() > 0:
                copy_loss = (copy_loss * transition_masks.float()).sum() / transition_masks.sum()
            else:
                copy_loss = copy_loss.mean()

            pred_std_across_batch = states_pred.std(dim=0).mean()

            if transition_masks.sum() > 0:
                mse = F.mse_loss(
                    states_pred_norm[transition_masks],
                    states_target_norm[transition_masks],
                )
                cos_sim = F.cosine_similarity(
                    states_pred_norm[transition_masks],
                    states_target_norm[transition_masks],
                    dim=-1,
                ).mean()
            else:
                mse = torch.tensor(0.0, device=videos.device)
                cos_sim = torch.tensor(0.0, device=videos.device)

        # ======================================================================
        # Norm Regularization: Prevent latent magnitude collapse
        # ======================================================================
        
        # 1. Regularize online encoder latents to maintain non-zero magnitude
        encoder_norms = states_flat.norm(dim=-1)  # (B*T,)
        encoder_norm_mean = encoder_norms.mean()
        # L2 penalty if norms drift too low
        encoder_norm_reg = torch.relu(self.target_norm_mean_target - encoder_norm_mean).pow(2)
        
        # 2. Predictor norms should also be maintained
        pred_norms = states_pred.norm(dim=-1)  # (B, T-1)
        pred_norm_mean = pred_norms.mean()
        pred_norm_reg = torch.relu(self.target_norm_mean_target - pred_norm_mean).pow(2)

        # ======================================================================
        # Decoder JEPA Loss: decode detached latents, train only online decoder
        # ======================================================================
        decoder_loss = torch.tensor(0.0, device=videos.device)
        decoder_lpips_loss = torch.tensor(0.0, device=videos.device)
        decoder_mse_loss = torch.tensor(0.0, device=videos.device)
        decoder_enc_loss = torch.tensor(0.0, device=videos.device)
        decoder_enc_mse_loss = torch.tensor(0.0, device=videos.device)
        decoder_enc_lpips_loss = torch.tensor(0.0, device=videos.device)
        if self.decoder_loss_weight > 0 and (self.decoder_pred_loss_weight > 0 or self.decoder_enc_loss_weight > 0):
            # GT frames for the predicted timesteps (frames 1..T-1)
            gt_frames_target = videos[:, 1:]  # (B, T-1, 3, H, W)
            gt_flat = rearrange(gt_frames_target, "b t c h w -> (b t) c h w")
            gt_flat_norm = 2.0 * gt_flat - 1.0  # [0,1] -> [-1,1] to match VAE output space

            # Mask valid transitions
            transition_flat = transition_masks.reshape(-1)  # (B*(T-1),)
            if transition_flat.sum() > 0:
                gt_valid = gt_flat_norm[transition_flat]
            else:
                gt_valid = gt_flat_norm

            decoder_lpips_weight = self.jepa_cfg.get("decoder_lpips_weight", 1.0)
            decoder_mse_weight = self.jepa_cfg.get("decoder_mse_weight", 1.0)

            # --- (A) Decode predicted latents (predictor output) ---
            if self.decoder_pred_loss_weight > 0:
                pred_latents_detached = pred_latents.detach()  # (B, T-1, C, H, W)
                pred_flat_for_dec = rearrange(pred_latents_detached, "b t c h w -> (b t) c h w")
                decoded_pred = self._decode_online(pred_flat_for_dec)  # (B*(T-1), 3, H, W)

                if transition_flat.sum() > 0:
                    decoded_pred_valid = decoded_pred[transition_flat]
                else:
                    decoded_pred_valid = decoded_pred

                decoder_mse_loss = F.mse_loss(decoded_pred_valid, gt_valid)
                decoder_lpips_loss = self.perceptual_loss(decoded_pred_valid.contiguous(), gt_valid.contiguous()).mean()
                decoder_loss = self.decoder_pred_loss_weight * (decoder_mse_weight * decoder_mse_loss + decoder_lpips_weight * decoder_lpips_loss)

            # --- (B) Decode encoded GT latents (online encoder output) ---
            if self.decoder_enc_loss_weight > 0:
                enc_latents_detached = online_latents[:, 1:].detach()  # (B, T-1, C, H, W)
                enc_flat_for_dec = rearrange(enc_latents_detached, "b t c h w -> (b t) c h w")
                decoded_enc = self._decode_online(enc_flat_for_dec)  # (B*(T-1), 3, H, W)

                if transition_flat.sum() > 0:
                    decoded_enc_valid = decoded_enc[transition_flat]
                else:
                    decoded_enc_valid = decoded_enc

                decoder_enc_mse_loss = F.mse_loss(decoded_enc_valid, gt_valid)
                decoder_enc_lpips_loss = self.perceptual_loss(decoded_enc_valid.contiguous(), gt_valid.contiguous()).mean()
                decoder_enc_loss = self.decoder_enc_loss_weight * (decoder_mse_weight * decoder_enc_mse_loss + decoder_lpips_weight * decoder_enc_lpips_loss)

            # Combined decoder loss
            decoder_loss = decoder_loss + decoder_enc_loss

        # Combine losses:
        # 1. Prediction loss (directional alignment with normalized targets)
        # 2. SigREG for both encoder and predictor (Gaussianity)
        # 3. Norm regularization to prevent magnitude collapse
        # 4. Decoder reconstruction loss (trains online decoder only)
        sigreg_loss_pred = 0
        total_jepa_loss = (
            self.prediction_loss_weight * pred_loss +
            self.sigreg_loss_weight * (sigreg_loss + sigreg_loss_pred) +
            self.norm_reg_weight * encoder_norm_reg +
            self.norm_reg_weight * pred_norm_reg +
            self.decoder_loss_weight * decoder_loss
        )

        log_dict = {
            "jepa/pred_loss": pred_loss,
            "jepa/copy_baseline_loss": copy_loss,
            "jepa/pred_vs_copy_ratio": pred_loss / (copy_loss + 1e-8),
            "jepa/sigreg_loss_encoder": sigreg_loss,
            "jepa/sigreg_loss_predictor": sigreg_loss_pred,
            "jepa/sigreg_loss": sigreg_loss + sigreg_loss_pred,
            "jepa/encoder_norm_reg": encoder_norm_reg,
            "jepa/pred_norm_reg": pred_norm_reg,
            "jepa/decoder_loss": decoder_loss,
            "jepa/decoder_pred_mse_loss": decoder_mse_loss,
            "jepa/decoder_pred_lpips_loss": decoder_lpips_loss,
            "jepa/decoder_enc_loss": decoder_enc_loss,
            "jepa/decoder_enc_mse_loss": decoder_enc_mse_loss,
            "jepa/decoder_enc_lpips_loss": decoder_enc_lpips_loss,
            "jepa/weighted_decoder_loss": self.decoder_loss_weight * decoder_loss,
            "jepa/mse": mse,
            "jepa/cos_sim": cos_sim,
            "jepa/encoder_norm_mean": encoder_norm_mean,
            "jepa/pred_norm_mean": pred_norm_mean,
            "jepa/target_norm_mean": states_target.norm(dim=-1).mean(),
            "jepa/pred_std_across_batch": pred_std_across_batch,
            "jepa/weighted_pred_loss": self.prediction_loss_weight * pred_loss,
            "jepa/weighted_sigreg_loss": self.sigreg_loss_weight * (sigreg_loss + sigreg_loss_pred),
            "jepa/weighted_norm_reg": self.norm_reg_weight * (encoder_norm_reg + pred_norm_reg),
        }
        return total_jepa_loss, log_dict

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    def training_step(self, batch, batch_idx, namespace="training") -> STEP_OUTPUT:
        """Training step: DFoT loss + JEPA loss, then EMA update."""
        self._update_component_freezing()

        xs, conditions, masks, gt_videos, actions_raw = batch

        # =============== DFoT Loss (uses target encoder latents) ===============
        noise_levels, masks_dfot = self._get_training_noise_levels(xs, masks)
        xs_pred, dfot_loss = self.diffusion_model(
            xs,
            self._process_conditions(conditions),
            k=noise_levels,
        )
        dfot_loss = self._reweight_loss(dfot_loss, masks_dfot)

        # =============== JEPA Loss (uses online encoder latents) ===============
        jepa_loss = torch.tensor(0.0, device=xs.device)
        jepa_log_dict: Dict[str, Tensor] = {}

        if (
            gt_videos is not None
            and actions_raw is not None
            and self.jepa_loss_weight > 0
        ):
            # Reuse target encoder latents already computed for diffusion (unnormalized)
            target_latents_for_jepa = self._unnormalize_x(xs)
            jepa_loss, jepa_log_dict = self._compute_jepa_loss(
                gt_videos, actions_raw, masks,
                target_latents_precomputed=target_latents_for_jepa,
            )

        # =============== Combined Loss ===============
        total_loss = dfot_loss + self.jepa_loss_weight * jepa_loss

        # =============== Logging ===============
        if batch_idx % self.cfg.logging.loss_freq == 0:
            self.log(f"{namespace}/loss", total_loss, on_step=True, sync_dist=True)
            self.log(f"{namespace}/dfot_loss", dfot_loss, on_step=True, sync_dist=True)
            self.log(f"{namespace}/jepa_loss", jepa_loss, on_step=True, sync_dist=True)
            for key, value in jepa_log_dict.items():
                self.log(f"{namespace}/{key}", value, on_step=True, sync_dist=True)

        xs, xs_pred = map(self._unnormalize_x, (xs, xs_pred))

        return {
            "loss": total_loss,
            "dfot_loss": dfot_loss,
            "jepa_loss": jepa_loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        """EMA update of target encoder and decoder after each training step (if due)."""
        super().on_train_batch_end(outputs, batch, batch_idx)
        if (self.global_step + 1) % self.ema_update_every == 0:
            if self.dit_unfreeze_step is None or self._dit_unfrozen:
                self._ema_update_target_encoder()
                self._ema_update_target_decoder()

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        xs, conditions, masks, gt_videos, actions_raw = batch
        parent_batch = (xs, conditions, masks, gt_videos)

        if self.trainer.state.fn == "FIT":
            self._eval_denoising_jepa(parent_batch, batch_idx, namespace=namespace)
            self._log_jepa_embeddings(gt_videos, actions_raw, namespace=namespace)
            self._log_jepa_upsampled_images(gt_videos, actions_raw, namespace=namespace)

        if not (
            self.trainer.sanity_checking and not self.cfg.logging.sanity_generation
        ):
            all_videos = self._sample_all_videos(parent_batch, batch_idx, namespace)
            if all_videos is not None:
                self._update_metrics(all_videos)
                self._log_videos(all_videos, namespace)

            if self.logging.save_embeddings:
                if namespace not in self.validation_embeddings:
                    self.validation_embeddings[namespace] = {}
                entry = {}
                if gt_videos is not None:
                    entry["gt"] = self._encode_online(gt_videos).detach().float().cpu()
                if all_videos is not None and "prediction" in all_videos:
                    entry["prediction"] = self._encode_online(
                        all_videos["prediction"]
                    ).detach().float().cpu()
                self.validation_embeddings[namespace][batch_idx] = entry
                rank_zero_print(
                    cyan(f"Stored JEPA embeddings for batch {batch_idx} "
                         f"keys={list(entry.keys())} (namespace={namespace})")
                )

    def _eval_denoising_jepa(self, batch, batch_idx, namespace="training") -> None:
        xs, conditions, masks, gt_videos = batch

        xs = xs[:, : self.max_tokens]
        if conditions is not None:
            conditions = conditions[:, : self.max_tokens]
        masks = masks[:, : self.max_tokens]
        if gt_videos is not None:
            gt_videos = gt_videos[:, : self.max_frames]

        jepa_batch = (xs, conditions, masks, gt_videos, conditions)
        output = self.training_step(jepa_batch, batch_idx, namespace=namespace)

        gt_videos_vis = gt_videos if self.is_latent_diffusion else output["xs"]
        recons = output["xs_pred"]
        if self.is_latent_diffusion:
            recons = self._decode(recons)

        if recons.shape[1] < gt_videos_vis.shape[1]:
            recons = F.pad(
                recons,
                (0, 0, 0, 0, 0, 0, 0, gt_videos_vis.shape[1] - recons.shape[1], 0, 0),
            )

        gt_videos_vis, recons = self.gather_data((gt_videos_vis, recons))

        from utils.distributed_utils import is_rank_zero
        from utils.logging_utils import log_video

        if not (
            is_rank_zero
            and self.logger
            and self.num_logged_videos < self.logging.max_num_videos
        ):
            return

        num_videos_to_log = min(
            self.logging.max_num_videos - self.num_logged_videos,
            gt_videos_vis.shape[0],
        )
        log_video(
            recons[:num_videos_to_log],
            gt_videos_vis[:num_videos_to_log],
            step=self.global_step,
            namespace="denoising_vis",
            logger=self.logger.experiment,
            indent=self.num_logged_videos,
            captions="denoised | gt",
        )

    @torch.no_grad()
    def _log_jepa_embeddings(
        self,
        gt_videos: Tensor,
        actions_raw: Tensor,
        namespace: str = "validation",
    ) -> None:
        """
        Visualise JEPA latents: first 3 channels of online-encoder latents
        (target) and ConvPredictor outputs (predicted) as W&B images/gifs.
        """
        import numpy as np
        import wandb
        from utils.distributed_utils import is_rank_zero

        if not (is_rank_zero and self.logger):
            return
        if gt_videos is None or actions_raw is None:
            return

        if self._jepa_vis_videos is None:
            n = min(self._jepa_num_vis, gt_videos.shape[0])
            self._jepa_vis_videos = gt_videos[:n].detach().cpu()
            self._jepa_vis_actions = actions_raw[:n].detach().cpu()

        vis_videos = self._jepa_vis_videos.to(device=gt_videos.device, dtype=gt_videos.dtype)
        vis_actions = self._jepa_vis_actions.to(device=actions_raw.device, dtype=actions_raw.dtype)
        N, T = vis_videos.shape[:2]
        if T < 2:
            return

        # 1. Encode with online encoder
        online_latents = self._encode_online(vis_videos)  # (N, T, C, H, W)
        C, latent_h, latent_w = online_latents.shape[2:]

        # 2. Run ConvPredictor in teacher-forcing mode
        pred_list = []
        for t in range(T - 1):
            z_t = online_latents[:, t]  # (N, C, H, W)
            action_t = vis_actions[:, t:t+1]  # (N, 1, action_dim)
            actions_dict = self._make_actions_dict(action_t)
            pred_t = self.predictor(z_t, actions_dict)  # (N, C, H, W)
            pred_list.append(pred_t)
        pred_latents = torch.stack(pred_list, dim=1)  # (N, T-1, C, H, W)

        # 3. Reshape
        target_latents = online_latents[:, 1:]  # (N, T-1, C, H, W)

        # 4. First 3 channels
        vis_ch = min(3, C)
        target_vis = target_latents[:, :, :vis_ch]
        pred_vis = pred_latents[:, :, :vis_ch]

        if vis_ch < 3:
            pad_shape = (*target_vis.shape[:2], 3 - vis_ch, latent_h, latent_w)
            pad = torch.zeros(pad_shape, device=target_vis.device)
            target_vis = torch.cat([target_vis, pad], dim=2)
            pred_vis = torch.cat([pred_vis, pad], dim=2)

        # 5. Shared min-max normalization
        def minmax_norm_shared(target: Tensor, pred: Tensor) -> Tuple[Tensor, Tensor]:
            flat_target = target.reshape(target.shape[0], -1)
            flat_pred = pred.reshape(pred.shape[0], -1)
            mn = torch.minimum(
                flat_target.min(dim=1).values,
                flat_pred.min(dim=1).values
            )[:, None, None, None, None]
            mx = torch.maximum(
                flat_target.max(dim=1).values,
                flat_pred.max(dim=1).values
            )[:, None, None, None, None]
            return (target - mn) / (mx - mn + 1e-8), (pred - mn) / (mx - mn + 1e-8)

        target_vis, pred_vis = minmax_norm_shared(target_vis, pred_vis)

        # 6. Flatten and log
        target_grid = target_vis.reshape(-1, 3, latent_h, latent_w)
        pred_grid = pred_vis.reshape(-1, 3, latent_h, latent_w)

        def to_hwc_uint8(t: Tensor) -> list:
            arr = (t.detach().cpu().float().numpy() * 255).astype(np.uint8)
            return list(np.transpose(arr, (0, 2, 3, 1)))

        def to_video_uint8(t: Tensor, target_size: int = 128) -> np.ndarray:
            upsampled = F.interpolate(t, size=(target_size, target_size), mode="nearest")
            return (upsampled.detach().cpu().float().numpy() * 255).astype(np.uint8)

        captions = [
            f"vid{v} t{t}->{t+1}"
            for v in range(N)
            for t in range(T - 1)
        ]

        log_dict: dict = {"trainer/global_step": self.global_step}

        log_dict[f"jepa_vis/{namespace}/frames/target"] = [
            wandb.Image(img, caption=cap)
            for img, cap in zip(to_hwc_uint8(target_grid), captions)
        ]
        log_dict[f"jepa_vis/{namespace}/frames/predicted"] = [
            wandb.Image(img, caption=cap)
            for img, cap in zip(to_hwc_uint8(pred_grid), captions)
        ]

        for v in range(N):
            log_dict[f"jepa_vis/{namespace}/gifs/target_vid{v}"] = wandb.Video(
                to_video_uint8(target_vis[v]), fps=4, format="gif"
            )
            log_dict[f"jepa_vis/{namespace}/gifs/predicted_vid{v}"] = wandb.Video(
                to_video_uint8(pred_vis[v]), fps=4, format="gif"
            )

        wandb.log(log_dict, step=self.global_step, commit=False)

    @torch.no_grad()
    def _log_jepa_upsampled_images(
        self,
        gt_videos: Tensor,
        actions_raw: Tensor,
        namespace: str = "validation",
    ) -> None:
        """Log 64x64 upsampled JEPA latent frames to W&B."""
        import numpy as np
        import wandb
        import torch.nn.functional as F
        from utils.distributed_utils import is_rank_zero

        if not (is_rank_zero and self.logger):
            return
        if self._jepa_vis_videos is None:
            return

        vis_videos = self._jepa_vis_videos.to(
            device=gt_videos.device, dtype=gt_videos.dtype
        )
        vis_actions = self._jepa_vis_actions.to(
            device=actions_raw.device, dtype=actions_raw.dtype
        )
        N, T = vis_videos.shape[:2]
        if T < 2:
            return

        # 1. Encode & predict
        online_latents = self._encode_online(vis_videos)
        C, latent_h, latent_w = online_latents.shape[2:]

        pred_list = []
        for t in range(T - 1):
            z_t = online_latents[:, t]
            action_t = vis_actions[:, t:t+1]
            actions_dict = self._make_actions_dict(action_t)
            pred_t = self.predictor(z_t, actions_dict)
            pred_list.append(pred_t)
        pred_latents = torch.stack(pred_list, dim=1)

        target_latents = online_latents[:, 1:]

        # 2. First 3 channels
        vis_ch = min(3, C)
        target_vis = target_latents[:, :, :vis_ch]
        pred_vis = pred_latents[:, :, :vis_ch]
        if vis_ch < 3:
            pad_shape = (*target_vis.shape[:2], 3 - vis_ch, latent_h, latent_w)
            pad = torch.zeros(pad_shape, device=target_vis.device)
            target_vis = torch.cat([target_vis, pad], dim=2)
            pred_vis = torch.cat([pred_vis, pad], dim=2)

        # 3. Shared min-max normalization
        def minmax_norm_shared(target: Tensor, pred: Tensor) -> Tuple[Tensor, Tensor]:
            flat_target = target.reshape(target.shape[0], -1)
            flat_pred = pred.reshape(pred.shape[0], -1)
            mn = torch.minimum(
                flat_target.min(dim=1).values,
                flat_pred.min(dim=1).values
            )[:, None, None, None, None]
            mx = torch.maximum(
                flat_target.max(dim=1).values,
                flat_pred.max(dim=1).values
            )[:, None, None, None, None]
            return (target - mn) / (mx - mn + 1e-8), (pred - mn) / (mx - mn + 1e-8)

        target_vis, pred_vis = minmax_norm_shared(target_vis, pred_vis)

        # 4. Upsample to 64x64
        target_grid = target_vis.reshape(-1, 3, latent_h, latent_w)
        pred_grid = pred_vis.reshape(-1, 3, latent_h, latent_w)

        target_64 = F.interpolate(target_grid, size=(64, 64), mode="nearest")
        pred_64 = F.interpolate(pred_grid, size=(64, 64), mode="nearest")

        def to_hwc_uint8(t: Tensor) -> list:
            arr = (t.detach().cpu().float().numpy() * 255).astype(np.uint8)
            return list(np.transpose(arr, (0, 2, 3, 1)))

        captions = [
            f"vid{v} t{t}->{t+1}" for v in range(N) for t in range(T - 1)
        ]

        log_dict: dict = {}
        log_dict[f"jepa_vis_64/{namespace}/frames/target"] = [
            wandb.Image(img, caption=cap)
            for img, cap in zip(to_hwc_uint8(target_64), captions)
        ]
        log_dict[f"jepa_vis_64/{namespace}/frames/predicted"] = [
            wandb.Image(img, caption=cap)
            for img, cap in zip(to_hwc_uint8(pred_64), captions)
        ]

        wandb.log(log_dict, step=self.global_step, commit=False)

    # -----------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------

    def _should_include_in_checkpoint(self, key: str) -> bool:
        base_include = super()._should_include_in_checkpoint(key)
        jepa_include = (
            key.startswith("predictor")
            or key.startswith("online_encoder")
            or key.startswith("online_quant_conv")
            or key.startswith("online_decoder")
            or key.startswith("online_post_quant_conv")
        )
        return base_include or jepa_include

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["jepa_cfg"] = self.jepa_cfg

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)

        expected_param_groups = 4
        ckpt_schedulers = checkpoint.get("lr_schedulers", None)
        if isinstance(ckpt_schedulers, list) and len(ckpt_schedulers) > 0:
            first_sched = ckpt_schedulers[0]
            if isinstance(first_sched, dict):
                sched_state = first_sched.get("state_dict", first_sched)
                if isinstance(sched_state, dict):
                    base_lrs = sched_state.get("base_lrs", None)
                    if (
                        isinstance(base_lrs, list)
                        and len(base_lrs) != expected_param_groups
                    ):
                        rank_zero_print(
                            cyan(
                                "Ignoring incompatible lr_schedulers state from "
                                f"checkpoint (ckpt groups={len(base_lrs)}, "
                                f"expected={expected_param_groups})."
                            )
                        )
                        checkpoint["lr_schedulers"] = []

        loaded_keys = set(checkpoint.get("state_dict", {}).keys())
        jepa_keys = [
            k
            for k in self.state_dict().keys()
            if k.startswith("predictor")
            or k.startswith("online_encoder")
            or k.startswith("online_quant_conv")
            or k.startswith("online_decoder")
            or k.startswith("online_post_quant_conv")
        ]
        loaded_jepa = [k for k in jepa_keys if k in loaded_keys]
        new_jepa = [k for k in jepa_keys if k not in loaded_keys]

        if loaded_jepa:
            rank_zero_print(cyan(f"Loaded JEPA weights: {len(loaded_jepa)} parameters"))
        if new_jepa:
            rank_zero_print(
                cyan(f"Randomly initialized JEPA weights: {len(new_jepa)} parameters")
            )
            rank_zero_print(
                cyan("  (This is expected when finetuning from pre-trained DFoT)")
            )

        rank_zero_print(cyan(f"VAE will be loaded from: {self.cfg.vae.pretrained_path}"))
