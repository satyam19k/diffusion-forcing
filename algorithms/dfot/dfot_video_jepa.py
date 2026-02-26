"""
Dual-Encoder JEPA for DFoT (BYOL-Style).

Architecture:
- **Target encoder** (frozen): Provides stable latents for DFoT diffusion.
  The data_mean / data_std normalization stays valid.
- **Online encoder** (trainable): A copy of the VAE encoder that receives
  JEPA gradients.  Learns to produce latents that are inherently predictive
  of future states.
- **ViT Predictor**: Operates on *clean* latent states from the online
  encoder and encoded actions; predicts future clean latent states.
- **EMA sync**: The online encoder is slowly blended into the target
  encoder (like BYOL / DINO) so DFoT gradually benefits from improved
  representations without sudden distribution shifts.

Gradient paths:
  DFoT loss  -->  diffusion backbone only
  JEPA loss  -->  online encoder + predictor + action encoder
"""

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
from utils.torch_utils import freeze_model
from utils.distributed_utils import rank_zero_print
from utils.print_utils import cyan
from .dfot_video import DFoTVideo

# Import official LeJEPA implementation
try:
    import lejepa
    USE_LEJEPA = True
except ImportError:
    # Fallback to custom standalone implementation
    from .sigreg import SigREG
    USE_LEJEPA = False


# =============================================================================
# JEPA Components
# =============================================================================

class FeedForward(nn.Module):
    """Feed-forward network with pre-norm."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CausalAttention(nn.Module):
    """Multi-head attention with causal masking for temporal sequences."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        dots = dots.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """Transformer block with causal attention."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.):
        super().__init__()
        self.attn = CausalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class ViTPredictor(nn.Module):
    """
    Causal ViT predictor that operates on *clean* latent states and
    encoded actions to predict the next clean latent state.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        depth: int = 4,
        heads: int = 8,
        dim_head: int = 64,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Input projections
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.combine_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )
        self.dropout_layer = nn.Dropout(dropout)

        # Transformer
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # Output projection -> next state
        self.output_proj = nn.Linear(hidden_dim, state_dim)

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        """
        Args:
            states:  (B, T, state_dim)  -- flattened clean latent states
            actions: (B, T, action_dim) -- encoded action embeddings
        Returns:
            pred_next_states: (B, T, state_dim) -- prediction for s_{t+1}
        """
        B, T, _ = states.shape

        state_emb = self.state_proj(states)
        action_emb = self.action_proj(actions)
        combined = torch.cat([state_emb, action_emb], dim=-1)
        x = self.combine_proj(combined)

        x = x + self.pos_embedding[:, :T, :]
        x = self.dropout_layer(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return states + self.output_proj(x)  # residual: predict delta, start from copy


class ActionEncoder(nn.Module):
    """Encodes raw actions into embeddings."""

    def __init__(self, action_dim: int, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, actions: Tensor) -> Tensor:
        return self.net(actions)


# =============================================================================
# Main Algorithm
# =============================================================================

class DFoTVideoJEPA(DFoTVideo):
    """
    DFoT with dual-encoder JEPA (BYOL-style).

    - Target encoder (self.vae) is frozen and provides stable latents for DFoT.
    - Online encoder (self.online_encoder + self.online_quant_conv) is trainable
      and produces clean latents for the JEPA predictor.
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

        # EMA config
        self.ema_decay = cfg.jepa.get("ema_decay", 0.99)
        self.ema_update_every = cfg.jepa.get("ema_update_every", 100)
        print(cyan(f"JEPA EMA decay: {self.ema_decay}, update every: {self.ema_update_every} steps"))

        # Progressive unfreezing: encoder and DIT training start after N steps
        self.encoder_unfreeze_step = cfg.jepa.get("encoder_unfreeze_step", None)  # None = train from start
        self.dit_unfreeze_step = cfg.jepa.get("dit_unfreeze_step", None)  # None = train from start
        print(cyan(f"JEPA encoder unfreeze step: {self.encoder_unfreeze_step}, DIT unfreeze step: {self.dit_unfreeze_step}"))
        self._encoder_unfrozen = False
        self._dit_unfrozen = False

        # Fixed probe videos for visualisation (captured lazily on first validation call)
        self._jepa_vis_videos: Optional[Tensor] = None   # (N, T, 3, H, W) on CPU
        self._jepa_vis_actions: Optional[Tensor] = None  # (N, T, action_dim) on CPU
        self._jepa_num_vis: int = int(cfg.jepa.get("num_vis_videos", 3))

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
        """Build ViT predictor and action encoder."""
        jepa_cfg = self.jepa_cfg

        # State dim = flattened latent
        latent_channels = self.x_shape[0]
        latent_h = self.x_shape[1]
        latent_w = self.x_shape[2] if len(self.x_shape) > 2 else latent_h
        self.state_dim = latent_channels * latent_h * latent_w

        # Action encoder
        self.action_encoder = ActionEncoder(
            action_dim=self.external_cond_dim,
            embed_dim=jepa_cfg.action_embed_dim,
            hidden_dim=jepa_cfg.action_hidden_dim,
        )

        # ViT Predictor
        self.predictor = ViTPredictor(
            state_dim=self.state_dim,
            action_dim=jepa_cfg.action_embed_dim,
            hidden_dim=jepa_cfg.predictor_hidden_dim,
            depth=jepa_cfg.predictor_depth,
            heads=jepa_cfg.predictor_heads,
            dim_head=jepa_cfg.get("predictor_dim_head", 64),
            mlp_ratio=jepa_cfg.get("predictor_mlp_ratio", 4.0),
            max_seq_len=self.max_tokens + 1,
            dropout=jepa_cfg.get("predictor_dropout", 0.1),
        )

        # SigREG — anti-collapse regularization on online encoder embeddings
        if USE_LEJEPA:
            # Use official LeJEPA implementation
            try:
                # Try the proper LeJEPA API
                univariate_test = lejepa.univariate.EppsPulley()
                self.sigreg = lejepa.multivariate.SlicingUnivariateTest(
                    univariate_test=univariate_test,
                    num_slices=self.sigreg_num_slices,
                )
                rank_zero_print(cyan("✓ Using official LeJEPA SigREG implementation"))
            except Exception as e:
                # Fallback if LeJEPA API is different
                rank_zero_print(cyan(f"⚠ LeJEPA init failed ({e}), using custom SigREG"))
                self.sigreg = SigREG(num_slices=self.sigreg_num_slices)
        else:
            # Fallback to custom standalone implementation
            self.sigreg = SigREG(num_slices=self.sigreg_num_slices)
            rank_zero_print(cyan("⚠ Using custom SigREG implementation (LeJEPA not installed)"))

        # SigREG projection: project from state_dim to lower dim before SigREG
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
        rank_zero_print(cyan(f"JEPA Predictor hidden dim: {jepa_cfg.predictor_hidden_dim}"))
        rank_zero_print(cyan(f"JEPA Training mode: {self.jepa_training_mode}"))
        rank_zero_print(cyan(f"JEPA EMA decay: {self.ema_decay}, update every: {self.ema_update_every}"))
        rank_zero_print(cyan(f"JEPA SigREG weight: {self.sigreg_loss_weight}, slices: {self.sigreg_num_slices}"))

    # -----------------------------------------------------------------
    # VAE loading -- dual encoder
    # -----------------------------------------------------------------

    def _load_vae(self) -> None:
        """
        Load VAE and create the dual-encoder setup:
        - self.vae = full VAE (target encoder -- frozen)
        - self.online_encoder = trainable copy of encoder
        - self.online_quant_conv = trainable copy of quant_conv
        """
        assert not self.is_latent_video_vae, "JEPA currently only supports ImageVAE"

        self.vae = ImageVAE.from_pretrained(
            path=self.cfg.vae.pretrained_path,
            **self.cfg.vae.pretrained_kwargs,
        ).to(self.device)

        # Freeze entire target VAE
        freeze_model(self.vae)
        rank_zero_print(cyan("Target VAE is fully frozen"))

        # Create trainable online encoder (deep copy of encoder + quant_conv)
        self.online_encoder = deepcopy(self.vae.encoder)
        self.online_quant_conv = deepcopy(self.vae.quant_conv)

        # Ensure online copies are trainable
        for p in self.online_encoder.parameters():
            p.requires_grad = True
        for p in self.online_quant_conv.parameters():
            p.requires_grad = True

        n_online_params = (
            sum(p.numel() for p in self.online_encoder.parameters())
            + sum(p.numel() for p in self.online_quant_conv.parameters())
        )
        rank_zero_print(
            cyan(f"Online encoder created: {n_online_params / 1e6:.1f}M trainable params")
        )

    # -----------------------------------------------------------------
    # Encoding helpers
    # -----------------------------------------------------------------

    def _encode_online(self, videos: Tensor) -> Tensor:
        """
        Encode raw videos with the trainable online encoder.
        Uses mode() for deterministic, clean latents.

        Args:
            videos: (B, T, 3, H, W) in [0, 1]
        Returns:
            latents: (B, T, C, H, W)
        """
        B, T = videos.shape[:2]
        x_flat = rearrange(videos, "b t c h w -> (b t) c h w")
        x_normalized = 2.0 * x_flat - 1.0

        # Online encoder forward
        h = self.online_encoder(x_normalized)
        moments = self.online_quant_conv(h)
        # Split into mean and logvar, take mode (= mean)
        mean, _ = torch.chunk(moments, 2, dim=1)
        latents_flat = mean  # mode() of DiagonalGaussian = mean

        return latents_flat.view(B, T, *latents_flat.shape[1:])

    def _encode_target(self, videos: Tensor) -> Tensor:
        """
        Encode raw videos with the frozen target encoder.
        Uses mode() for deterministic, clean latents.

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
    # EMA
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _ema_update_target_encoder(self) -> None:
        """
        Exponential moving average update: blend online encoder weights
        into the target encoder.  Called every ``ema_update_every`` steps.
        """
        decay = self.ema_decay
        for p_online, p_target in zip(
            self.online_encoder.parameters(), self.vae.encoder.parameters()
        ):
            p_target.data.mul_(decay).add_(p_online.data, alpha=1.0 - decay)

        for p_online, p_target in zip(
            self.online_quant_conv.parameters(), self.vae.quant_conv.parameters()
        ):
            p_target.data.mul_(decay).add_(p_online.data, alpha=1.0 - decay)

    # -----------------------------------------------------------------
    # Progressive Unfreezing
    # -----------------------------------------------------------------

    def _update_component_freezing(self) -> None:
        """
        Progressive unfreezing: gradually enable training of encoder and DIT
        at specified steps for curriculum learning.
        """
        # Unfreeze encoder at specified step
        if self.encoder_unfreeze_step is not None and self.global_step == self.encoder_unfreeze_step:
            if not self._encoder_unfrozen:
                for p in self.online_encoder.parameters():
                    p.requires_grad = True
                for p in self.online_quant_conv.parameters():
                    p.requires_grad = True
                self._encoder_unfrozen = True
                rank_zero_print(
                    cyan(f"✓ Unfroze online encoder at step {self.global_step}")
                )

        # Unfreeze DIT at specified step
        if self.dit_unfreeze_step is not None and self.global_step == self.dit_unfreeze_step:
            if not self._dit_unfrozen:
                for p in self.diffusion_model.parameters():
                    p.requires_grad = True
                self._dit_unfrozen = True
                rank_zero_print(
                    cyan(f"✓ Unfroze DIT (diffusion model) at step {self.global_step}")
                )

    # -----------------------------------------------------------------
    # Optimizers
    # -----------------------------------------------------------------

    def configure_optimizers(self):
        """Three param groups: diffusion, JEPA head, online encoder."""
        params_groups = []

        # 1. Diffusion model (backbone -- trained by DFoT loss only)
        # Initially frozen if dit_unfreeze_step is set
        dit_params = list(self.diffusion_model.parameters())
        if self.dit_unfreeze_step is not None:
            for p in dit_params:
                p.requires_grad = False
            rank_zero_print(cyan(f"Freezing DIT until step {self.dit_unfreeze_step}"))
        
        params_groups.append({
            "params": dit_params,
            "lr":  self.cfg.lr,  # Lower LR for stability when training with JEPA
            "name": "diffusion",
        })

        # 2. JEPA predictor + action encoder + sigreg projection
        jepa_params = (
            list(self.predictor.parameters()) +
            list(self.action_encoder.parameters()) +
            (list(self.sigreg_proj.parameters()) if self.sigreg_proj is not None else [])
        )
        params_groups.append({
            "params": jepa_params,
            "lr": self.jepa_cfg.lr,
            "name": "jepa",
        })

        # 3. Online encoder (lower LR for stability)
        # Initially frozen if encoder_unfreeze_step is set
        online_params = (
            list(self.online_encoder.parameters()) +
            list(self.online_quant_conv.parameters())
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
            xs:          (B, T, C, H, W) normalized latents from *target* encoder for DFoT
            conditions:  (B, T, cond_dim)
            masks:       (B, T)
            gt_videos:   (B, T, 3, H, W) raw videos for JEPA
            actions_raw: (B, T, action_dim) raw actions for JEPA
        """
        gt_videos = batch.get("videos", None)
        actions_raw = batch.get("conds", None)

        # DFoT path: encode with frozen target VAE (via parent's _encode)
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
    # JEPA Loss
    # -----------------------------------------------------------------

    def _compute_jepa_loss(
        self,
        videos: Tensor,
        actions: Tensor,
        masks: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute JEPA prediction loss on clean latents from the online encoder.

        Pipeline:
        1. Encode all frames with the online encoder -> clean latents
        2. Flatten to state vectors
        3. Encode actions
        4. Predictor: (states[:-1], actions[:-1]) -> pred_states
        5. Loss = smooth_l1(pred_states, states[1:].detach())
        """
        B, T = videos.shape[:2]

        if T < 2:
            return torch.tensor(0.0, device=videos.device), {}

        # 1. Encode with online encoder (gradients flow through)
        online_latents = self._encode_online(videos)  # (B, T, C, H, W)

        # 2. Flatten
        states = online_latents.reshape(B, T, -1)  # (B, T, state_dim)

        # 2b. SigREG on online embeddings (pool across time → one vector per sample)
        sigreg_input = states.reshape(B * T, -1)  # (B*T, state_dim)
        if self.sigreg_proj is not None:
            sigreg_input = self.sigreg_proj(sigreg_input)  # (B*T, sigreg_proj_dim)
        sigreg_loss = self.sigreg(sigreg_input)

        # 3. Encode actions
        action_embeds = self.action_encoder(actions)  # (B, T, action_embed_dim)

        # 4. Inputs and targets
        states_input = states[:, :-1]             # (B, T-1, state_dim)
        action_embeds_input = action_embeds[:, :-1]  # (B, T-1, action_embed_dim)
        states_target = states[:, 1:].detach()    # (B, T-1, state_dim) -- detached!

        # 5. Predict
        if self.jepa_training_mode == "autoregressive":
            pred_list = []
            current = states[:, 0:1]
            for t in range(T - 1):
                act_t = action_embeds_input[:, t : t + 1]
                pred_t = self.predictor(current, act_t)
                pred_list.append(pred_t)
                # Detach to prevent long backprop chains
                current = pred_t.detach() if t < T - 2 else pred_t
            states_pred = torch.cat(pred_list, dim=1)
        else:
            # Teacher forcing (default)
            states_pred = self.predictor(states_input, action_embeds_input)

        # 2c. SigREG on predicted embeddings (prevent predictor collapse)
        sigreg_input_pred = states_pred.reshape(-1, self.state_dim)  # (B*(T-1), state_dim)
        if self.sigreg_proj is not None:
            sigreg_input_pred = self.sigreg_proj(sigreg_input_pred)  # (B*(T-1), sigreg_proj_dim)
        sigreg_loss_pred = self.sigreg(sigreg_input_pred)

        # 6. Loss
        transition_masks = masks[:, :-1] & masks[:, 1:]

        # Normalize latents for direction-focused losses
        states_pred_norm = F.normalize(states_pred, dim=-1)
        states_target_norm = F.normalize(states_target, dim=-1)

        pred_loss = F.mse_loss(
            states_pred_norm, states_target_norm, reduction="none"
        ).mean(dim=-1)  # (B, T-1)

        if transition_masks.sum() > 0:
            pred_loss = (
                (pred_loss * transition_masks.float()).sum() / transition_masks.sum()
            )
        else:
            pred_loss = pred_loss.mean()

        # Cosine similarity loss (directional alignment)
        # cos_sim_pred = F.cosine_similarity(states_pred_norm, states_target_norm, dim=-1)  # (B, T-1)
        # cos_sim_loss = 1.0 - cos_sim_pred  # Loss in [0, 2], minimized when cos_sim = 1

        # if transition_masks.sum() > 0:
        #     cos_sim_loss = (cos_sim_loss * transition_masks.float()).sum() / transition_masks.sum()
        # else:
        #     cos_sim_loss = cos_sim_loss.mean()

        # Metrics
        with torch.no_grad():
            # Copy baseline: how good is s_t as a prediction of s_{t+1}?
            copy_pred_norm = F.normalize(states_input, dim=-1)
            copy_loss = F.mse_loss(
                copy_pred_norm, states_target_norm, reduction="none"
            ).mean(dim=-1)  # (B, T-1)
            if transition_masks.sum() > 0:
                copy_loss = (copy_loss * transition_masks.float()).sum() / transition_masks.sum()
            else:
                copy_loss = copy_loss.mean()

            # Predictor output stats
            pred_norm_mean = states_pred.norm(dim=-1).mean()
            target_norm_mean = states_target.norm(dim=-1).mean()
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

        # Combine prediction loss + cosine similarity loss + SigREG (encoder + predictor)
        sigreg_loss_pred = 0
        total_jepa_loss = (
            self.prediction_loss_weight * pred_loss + 
            #self.lambda_cos * cos_sim_loss +
            self.sigreg_loss_weight * (sigreg_loss + sigreg_loss_pred)
        )

        log_dict = {
            "jepa/pred_loss": pred_loss,
            "jepa/copy_baseline_loss": copy_loss,
            "jepa/pred_vs_copy_ratio": pred_loss / (copy_loss + 1e-8),
            #"jepa/cos_sim_loss": cos_sim_loss,
            "jepa/sigreg_loss_encoder": sigreg_loss,
            "jepa/sigreg_loss_predictor": sigreg_loss_pred,
            "jepa/sigreg_loss": sigreg_loss + sigreg_loss_pred,
            "jepa/mse": mse,
            "jepa/cos_sim": cos_sim,
            "jepa/pred_norm_mean": pred_norm_mean,
            "jepa/target_norm_mean": target_norm_mean,
            "jepa/pred_std_across_batch": pred_std_across_batch,
            "jepa/weighted_pred_loss": self.prediction_loss_weight * pred_loss,
            #"jepa/weighted_cos_sim_loss": self.lambda_cos * cos_sim_loss,
            "jepa/weighted_sigreg_loss": self.sigreg_loss_weight * (sigreg_loss + sigreg_loss_pred),
        }
        return total_jepa_loss, log_dict

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    def training_step(self, batch, batch_idx, namespace="training") -> STEP_OUTPUT:
        """Training step: DFoT loss + JEPA loss, then EMA update."""
        # Progressive unfreezing: check if we should unfreeze components
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
            jepa_loss, jepa_log_dict = self._compute_jepa_loss(
                gt_videos, actions_raw, masks,
            )

        # =============== Combined Loss ===============
        total_loss =  dfot_loss + self.jepa_loss_weight * jepa_loss #+ self.sigreg_loss_weight * jepa_log_dict.get("jepa/sigreg_loss", 0.0)
       
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
        """EMA update of target encoder after each training step (if due)."""
        super().on_train_batch_end(outputs, batch, batch_idx)
        if (self.global_step + 1) % self.ema_update_every == 0:
            # Only update EMA if DIT is unfrozen (or if no freezing configured)
            if self.dit_unfreeze_step is None or self._dit_unfrozen:
                self._ema_update_target_encoder()

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        """Validation step -- convert 5-element batch for parent methods."""
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

            # Save JEPA embeddings (shape: B, T, C, H, W in latent space)
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
                    cyan(f"✓ Stored JEPA embeddings for batch {batch_idx} "
                         f"keys={list(entry.keys())} (namespace={namespace})")
                )

    def _eval_denoising_jepa(self, batch, batch_idx, namespace="training") -> None:
        """Evaluate denoising -- adapted for the 5-element batch."""
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
        Visualise JEPA latents at validation time by logging the first 3 channels
        of online-encoder latents (target) and predictor outputs (predicted) as
        W&B images.  Useful for diagnosing representational collapse and predictor
        mistakes.

        Logs two image grids to W&B:
          jepa_vis/{namespace}/target    – first 3 ch of online latents[t+1]
          jepa_vis/{namespace}/predicted – first 3 ch of predictor output[t]
        Each image in the grid: one batch item × one timestep transition.
        """
        import numpy as np
        import wandb
        from utils.distributed_utils import is_rank_zero
        # import pdb; pdb.set_trace()

        if not (is_rank_zero and self.logger):
            return
        if gt_videos is None or actions_raw is None:
            return

        # Lazy init: capture fixed probe videos from the first validation batch.
        # These same videos will be encoded and logged at every subsequent validation step,
        # making it easy to track how their representations evolve over training.
        if self._jepa_vis_videos is None:
            n = min(self._jepa_num_vis, gt_videos.shape[0])
            self._jepa_vis_videos  = gt_videos[:n].detach().cpu()
            self._jepa_vis_actions = actions_raw[:n].detach().cpu()

        # Move probe set to the current device / dtype
        vis_videos  = self._jepa_vis_videos.to(device=gt_videos.device, dtype=gt_videos.dtype)
        vis_actions = self._jepa_vis_actions.to(device=actions_raw.device, dtype=actions_raw.dtype)
        N, T = vis_videos.shape[:2]
        if T < 2:
            return

        # 1. Encode with online encoder -> (N, T, C, H, W)
        online_latents = self._encode_online(vis_videos)
        C, latent_h, latent_w = online_latents.shape[2:]

        # 2. Run predictor in teacher-forcing mode -> (N, T-1, state_dim)
        states = online_latents.reshape(N, T, -1)
        action_embeds = self.action_encoder(vis_actions)          # (N, T, A)
        states_pred = self.predictor(
            states[:, :-1], action_embeds[:, :-1]
        )  # (N, T-1, state_dim)

        # 3. Reshape predictions back to spatial latent dims
        target_latents = online_latents[:, 1:]                    # (N, T-1, C, H, W)
        pred_latents   = states_pred.reshape(N, T - 1, C, latent_h, latent_w)

        # 4. Take first 3 channels (or fewer if C < 3)
        vis_ch = min(3, C)
        target_vis = target_latents[:, :, :vis_ch]                # (N, T-1, vis_ch, H, W)
        pred_vis   = pred_latents[:, :, :vis_ch]

        # Pad to exactly 3 channels so W&B renders as RGB
        if vis_ch < 3:
            pad_shape = (*target_vis.shape[:3], 3 - vis_ch, latent_h, latent_w)
            pad = torch.zeros(pad_shape, device=target_vis.device)
            target_vis = torch.cat([target_vis, pad], dim=2)
            pred_vis   = torch.cat([pred_vis,   pad], dim=2)

        # 5. Per-sample min-max normalise to [0, 1] using SHARED min/max
        # Collapse shows as uniform grey; healthy latents show spatial structure.
        def minmax_norm_shared(target: Tensor, pred: Tensor) -> Tuple[Tensor, Tensor]:
            # Compute min/max across both target and pred per sample
            # x: (N, T-1, 3, H, W) — normalise per probe video using same scale
            flat_target = target.reshape(target.shape[0], -1)
            flat_pred = pred.reshape(pred.shape[0], -1)
            
            # Global min/max per sample across both target and pred
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

        # 6. Flatten to (N*(T-1), 3, H, W) for logging
        target_grid = target_vis.reshape(-1, 3, latent_h, latent_w)
        pred_grid   = pred_vis.reshape(-1, 3, latent_h, latent_w)

        def to_hwc_uint8(t: Tensor) -> list:
            arr = (t.detach().cpu().float().numpy() * 255).astype(np.uint8)
            return list(np.transpose(arr, (0, 2, 3, 1)))  # (N*(T-1), H, W, 3)

        def to_video_uint8(t: Tensor, target_size: int = 128) -> np.ndarray:
            # Upsample to target_size with nearest-neighbor to preserve pixelated look.
            # wandb.Video expects (T, C, H, W) uint8 — no transpose needed.
            import torch.nn.functional as F
            upsampled = F.interpolate(
                t, size=(target_size, target_size), mode="nearest"
            )
            return (upsampled.detach().cpu().float().numpy() * 255).astype(np.uint8)

        captions = [
            f"vid{v} t{t}->{t+1}"
            for v in range(N)
            for t in range(T - 1)
        ]

        log_dict: dict = {"trainer/global_step": self.global_step}

        # --- Per-frame image grids (jepa_vis/<namespace>/frames/) ---
        log_dict[f"jepa_vis/{namespace}/frames/target"] = [
            wandb.Image(img, caption=cap)
            for img, cap in zip(to_hwc_uint8(target_grid), captions)
        ]
        log_dict[f"jepa_vis/{namespace}/frames/predicted"] = [
            wandb.Image(img, caption=cap)
            for img, cap in zip(to_hwc_uint8(pred_grid), captions)
        ]

        # --- Per-video GIFs (jepa_vis/<namespace>/gifs/) ---
        # Frames upsampled with nearest-neighbor to preserve the pixelated patch look
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
        """
        Log 64x64 upsampled JEPA latent frames to W&B.

        Reuses the same probe videos captured by ``_log_jepa_embeddings``.
        Logged under ``jepa_vis_64/{namespace}/frames/target`` and
        ``jepa_vis_64/{namespace}/frames/predicted``.
        """
        import numpy as np
        import wandb
        import torch.nn.functional as F
        from utils.distributed_utils import is_rank_zero

        if not (is_rank_zero and self.logger):
            return
        if self._jepa_vis_videos is None:
            return  # probe videos not yet captured

        vis_videos = self._jepa_vis_videos.to(
            device=gt_videos.device, dtype=gt_videos.dtype
        )
        vis_actions = self._jepa_vis_actions.to(
            device=actions_raw.device, dtype=actions_raw.dtype
        )
        N, T = vis_videos.shape[:2]
        if T < 2:
            return

        # 1. Encode & predict (same as _log_jepa_embeddings)
        online_latents = self._encode_online(vis_videos)  # (N, T, C, H, W)
        C, latent_h, latent_w = online_latents.shape[2:]

        states = online_latents.reshape(N, T, -1)
        action_embeds = self.action_encoder(vis_actions)
        states_pred = self.predictor(states[:, :-1], action_embeds[:, :-1])

        target_latents = online_latents[:, 1:]  # (N, T-1, C, H, W)
        pred_latents = states_pred.reshape(N, T - 1, C, latent_h, latent_w)

        # 2. First 3 channels, pad to 3 if needed
        vis_ch = min(3, C)
        target_vis = target_latents[:, :, :vis_ch]
        pred_vis = pred_latents[:, :, :vis_ch]
        if vis_ch < 3:
            pad_shape = (*target_vis.shape[:2], 3 - vis_ch, latent_h, latent_w)
            pad = torch.zeros(pad_shape, device=target_vis.device)
            target_vis = torch.cat([target_vis, pad], dim=2)
            pred_vis = torch.cat([pred_vis, pad], dim=2)

        # 3. Per-sample min-max normalise to [0, 1] using SHARED min/max
        def minmax_norm_shared(target: Tensor, pred: Tensor) -> Tuple[Tensor, Tensor]:
            # Compute min/max across both target and pred per sample
            flat_target = target.reshape(target.shape[0], -1)
            flat_pred = pred.reshape(pred.shape[0], -1)
            
            # Global min/max per sample across both target and pred
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

        # 4. Flatten to (N*(T-1), 3, H, W) then upsample 32x32 -> 64x64
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
        """Include JEPA components and online encoder in checkpoint."""
        base_include = super()._should_include_in_checkpoint(key)
        jepa_include = (
            key.startswith("action_encoder")
            or key.startswith("predictor")
            or key.startswith("online_encoder")
            or key.startswith("online_quant_conv")
        )
        return base_include or jepa_include

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["jepa_cfg"] = self.jepa_cfg

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)

        # Guard against scheduler state incompatibility (DFoT ckpt has 1
        # group, JEPA has 3).
        expected_param_groups = 3
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

        # Report which JEPA keys were loaded vs randomly initialized
        loaded_keys = set(checkpoint.get("state_dict", {}).keys())
        jepa_keys = [
            k
            for k in self.state_dict().keys()
            if k.startswith("action_encoder")
            or k.startswith("predictor")
            or k.startswith("online_encoder")
            or k.startswith("online_quant_conv")
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
