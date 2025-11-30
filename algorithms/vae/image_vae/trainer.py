"""
Adapted from CompVis/latent-diffusion
https://github.com/CompVis/stable-diffusion
"""

import types
from typing import Tuple, Callable, Optional, Dict, Any
from functools import partial
from omegaconf import OmegaConf, DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from einops import rearrange
from diffusers import AutoencoderKL as DiffuserImageVAE
from torchmetrics.image import FrechetInceptionDistance
from utils.logging_utils import log_video
from utils.logging_utils import get_validation_metrics_for_videos
from utils.ckpt_utils import (
    is_wandb_run_path,
    is_hf_path,
    wandb_to_local_path,
    download_pretrained as hf_to_local_path,
)
from ..common.distribution import DiagonalGaussianDistribution
from ..common.base_vae import VAE
from ..common.losses import LPIPSWithDiscriminator, warmup
from .model import Encoder, Decoder
from .predictor import LatentPredictor


class ImageVAETrainer(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.learning_rate = cfg.lr
        self.automatic_optimization = False
        ddconfig, lossconfig = cfg.ddconfig, cfg.lossconfig
        self.embed_dim = cfg.embed_dim
        self.warmup_steps = cfg.warmup_steps
        self.gradient_clip_val = cfg.gradient_clip_val
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = LPIPSWithDiscriminator(**lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(
            2 * ddconfig["z_channels"], 2 * self.embed_dim, 1
        )
        self.post_quant_conv = torch.nn.Conv2d(
            self.embed_dim, ddconfig["z_channels"], 1
        )
        if cfg.ckpt_path is not None:
            self.init_from_ckpt(cfg.ckpt_path)
        self.fid_model = None

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def on_save_checkpoint(self, checkpoint):
        """
        save cfgs together to enable easily loading the pretrained model
        """
        checkpoint["cfg"] = self.cfg
        return checkpoint

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def on_after_batch_transfer(
        self, batch: tuple, dataloader_idx: int = 0
    ) -> torch.Tensor:
        x = batch["videos"]
        x = 2.0 * x - 1.0  # normalize to [-1, 1]
        return x

    def training_step(self, batch, batch_idx):
        # pylint: disable=unpacking-non-sequence
        opt_ae, opt_disc = self.optimizers()

        batch = rearrange(batch, "b t c h w -> (b t) c h w")
        reconstructions, posterior = self(batch)

        log_loss = partial(
            self.log,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )

        log_loss_dict = partial(
            self.log_dict,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        # Warm-up: at the beginning of training / after GAN loss starts being used
        # compute lr_scale
        should_warmup, lr_scale = False, 1.0
        if self.global_step < self.warmup_steps:
            should_warmup = True
            lr_scale = float(self.global_step + 1) / self.warmup_steps
        elif (
            self.global_step >= self.cfg.lossconfig.disc_start - 1
            and self.global_step < self.cfg.lossconfig.disc_start + self.warmup_steps
        ):
            should_warmup = True
            lr_scale = (
                float(self.global_step - self.cfg.lossconfig.disc_start + 1)
                / self.warmup_steps
            )
        lr_scale = min(1.0, lr_scale)

        # Optimize the autoencoder

        aeloss, log_dict_ae = self.loss(
            batch,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        self.clip_gradients(opt_ae, gradient_clip_val=self.gradient_clip_val)
        if should_warmup:
            opt_ae = warmup(opt_ae, self.learning_rate, lr_scale)
        opt_ae.step()

        log_loss(
            "aeloss",
            aeloss,
        )

        log_loss_dict(log_dict_ae)

        # Optimize the discriminator
        discloss, log_dict_disc = self.loss(
            batch,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )

        opt_disc.zero_grad()
        self.manual_backward(discloss)
        self.clip_gradients(opt_disc, gradient_clip_val=self.gradient_clip_val)
        if should_warmup:
            opt_disc = warmup(opt_disc, self.learning_rate, lr_scale)
        opt_disc.step()

        log_loss(
            "discloss",
            discloss,
        )

        log_loss_dict(log_dict_disc)

    def on_validation_epoch_start(self):
        self.fid_model = FrechetInceptionDistance(feature=64).to(self.device)

    def on_validation_epoch_end(self):
        self.fid_model = None

    def validation_step(self, batch, batch_idx):
        batch_size = batch.size(0)
        batch = rearrange(batch, "b t c h w -> (b t) c h w")
        reconstructions, posterior = self(batch)
        aeloss, log_dict_ae = self.loss(
            batch,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        discloss, log_dict_disc = self.loss(
            batch,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"], sync_dist=True)
        self.log_dict(log_dict_ae, sync_dist=True)
        self.log_dict(log_dict_disc, sync_dist=True)

        validation_metrics = get_validation_metrics_for_videos(
            *map(
                lambda x: rearrange(x, "(b t) c h w -> t b c h w", b=batch_size)
                .contiguous()
                .detach(),
                (batch, reconstructions),
            ),
            fid_model=self.fid_model,
        )

        self.log_dict(
            {f"val/{k}": v for k, v in validation_metrics.items()},
            prog_bar=True,
            sync_dist=True,
        )

        if batch_idx == 0:  # log visualizations
            batch, reconstructions = (
                self._rearrange_and_unnormalize(x, batch_size).detach().cpu()
                for x in (batch, reconstructions)
            )
            if self.logger is not None:
                log_video(
                    reconstructions,
                    batch,
                    step=self.global_step,
                    namespace="reconstruction_vis",
                    logger=self.logger.experiment,
                )

    def _rearrange_and_unnormalize(
        self, batch: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        batch = rearrange(batch, "(b t) c h w -> t b c h w", b=batch_size)
        batch = 0.5 * batch + 0.5
        return batch

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight


class ImageVAEPredictiveTrainer(ImageVAETrainer):
    """
    ImageVAE Trainer with JEPA-style predictive loss.
    Jointly trains the VAE with a predictor that predicts future latents from current latents + actions.
    
    L_total = L_VAE + λ * L_pred
    where L_pred = MSE(predictor(μ_t, a_t), stopgrad(μ_{t+1}))
    """
    
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        
        # Predictor configuration
        predictor_cfg = cfg.get("predictor", {})
        self.predictor_enabled = predictor_cfg.get("enabled", True)
        self.lambda_pred = predictor_cfg.get("lambda_pred", 1.0)
        
        if self.predictor_enabled:
            # Get latent dimensions from ddconfig
            latent_size = cfg.ddconfig.resolution // 8  # VAE downsamples by 8x
            latent_channels = cfg.embed_dim
            action_dim = cfg.get("action_dim", 4)  # Minecraft has 4 actions
            hidden_dim = predictor_cfg.get("hidden_dim", 64)
            
            self.predictor = LatentPredictor(
                latent_channels=latent_channels,
                latent_size=latent_size,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
            )
        
        # Load pretrained VAE weights if specified
        pretrained_vae_path = cfg.get("pretrained_vae_path", None)
        if pretrained_vae_path is not None:
            self._load_pretrained_vae(pretrained_vae_path)
    
    def _load_pretrained_vae(self, path: str):
        """Load pretrained VAE weights (encoder, decoder, quant_conv, post_quant_conv, loss/discriminator)."""
        if is_hf_path(path):
            path = hf_to_local_path(path)
        elif is_wandb_run_path(path):
            path = wandb_to_local_path(path)
        
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        # Load ALL weights from pretrained checkpoint
        # Only exclude keys that don't exist in pretrained (predictor is new)
        exclude_keys = ["predictor"]
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not any(k.startswith(prefix) for prefix in exclude_keys)
        }
        
        # Load with strict=False to allow missing predictor keys
        missing, unexpected = self.load_state_dict(filtered_state_dict, strict=False)
        
        # Categorize what was loaded
        loaded_components = set()
        for k in filtered_state_dict.keys():
            component = k.split('.')[0]
            if component == 'loss':
                component = k.split('.')[1] if len(k.split('.')) > 1 else 'loss'
            loaded_components.add(component)
        
        print(f"Loaded pretrained checkpoint from {path}")
        print(f"  Loaded components: {sorted(loaded_components)}")
        print(f"  Missing keys (expected for predictor): {[k for k in missing if 'predictor' in k]}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
    
    def on_after_batch_transfer(
        self, batch: Dict[str, torch.Tensor], dataloader_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Process batch after transfer to device.
        Handles both video-only batches and video+action batches.
        """
        if isinstance(batch, dict):
            videos = batch["videos"]
            videos = 2.0 * videos - 1.0  # normalize to [-1, 1]
            result = {"videos": videos}
            
            # Include actions if present (dataset returns "conds")
            if "conds" in batch:
                result["actions"] = batch["conds"]
            
            return result
        else:
            # Fallback for tensor-only batches
            return 2.0 * batch - 1.0
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        Training step with joint VAE + predictive loss.
        """
        # pylint: disable=unpacking-non-sequence
        opt_ae, opt_disc = self.optimizers()
        
        videos = batch["videos"]  # (B, T, C, H, W)
        actions = batch.get("actions", None)  # (B, T, action_dim) or None
        
        batch_size, n_frames = videos.shape[:2]
        
        # Flatten for VAE processing
        videos_flat = rearrange(videos, "b t c h w -> (b t) c h w")
        
        # Forward pass through VAE
        reconstructions, posterior = self(videos_flat)
        
        log_loss = partial(
            self.log,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        
        log_loss_dict = partial(
            self.log_dict,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        
        # Warm-up logic (same as parent)
        should_warmup, lr_scale = False, 1.0
        if self.global_step < self.warmup_steps:
            should_warmup = True
            lr_scale = float(self.global_step + 1) / self.warmup_steps
        elif (
            self.global_step >= self.cfg.lossconfig.disc_start - 1
            and self.global_step < self.cfg.lossconfig.disc_start + self.warmup_steps
        ):
            should_warmup = True
            lr_scale = (
                float(self.global_step - self.cfg.lossconfig.disc_start + 1)
                / self.warmup_steps
            )
        lr_scale = min(1.0, lr_scale)
        
        # Compute VAE loss
        aeloss, log_dict_ae = self.loss(
            videos_flat,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        
        # Compute predictive loss if enabled and actions are available
        pred_loss = torch.tensor(0.0, device=self.device)
        if self.predictor_enabled and actions is not None and n_frames > 1:
            # Get latent means: (B*T, C, H, W) -> (B, T, C, H, W)
            latent_means = posterior.mean
            latent_means = rearrange(
                latent_means, "(b t) c h w -> b t c h w", b=batch_size, t=n_frames
            )
            
            # Current and next latents
            mu_t = latent_means[:, :-1]  # (B, T-1, C, H, W)
            mu_t1_target = latent_means[:, 1:].detach()  # (B, T-1, C, H, W) - stopgrad
            
            # Current actions
            a_t = actions[:, :-1]  # (B, T-1, action_dim)
            
            # Flatten for predictor
            mu_t_flat = rearrange(mu_t, "b t c h w -> (b t) c h w")
            a_t_flat = rearrange(a_t, "b t d -> (b t) d")
            mu_t1_target_flat = rearrange(mu_t1_target, "b t c h w -> (b t) c h w")
            
            # Predict next latent
            mu_t1_pred = self.predictor(mu_t_flat, a_t_flat)
            
            # MSE loss between prediction and target
            pred_loss = F.mse_loss(mu_t1_pred, mu_t1_target_flat)
            
            log_loss("train/pred_loss", pred_loss)
        
        # Total autoencoder loss
        total_aeloss = aeloss + self.lambda_pred * pred_loss
        
        # Optimize autoencoder + predictor
        opt_ae.zero_grad()
        self.manual_backward(total_aeloss)
        self.clip_gradients(opt_ae, gradient_clip_val=self.gradient_clip_val)
        if should_warmup:
            opt_ae = warmup(opt_ae, self.learning_rate, lr_scale)
        opt_ae.step()
        
        log_loss("aeloss", aeloss)
        log_loss("total_aeloss", total_aeloss)
        log_loss_dict(log_dict_ae)
        
        # Optimize discriminator (same as parent)
        discloss, log_dict_disc = self.loss(
            videos_flat,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        
        opt_disc.zero_grad()
        self.manual_backward(discloss)
        self.clip_gradients(opt_disc, gradient_clip_val=self.gradient_clip_val)
        if should_warmup:
            opt_disc = warmup(opt_disc, self.learning_rate, lr_scale)
        opt_disc.step()
        
        log_loss("discloss", discloss)
        log_loss_dict(log_dict_disc)
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        Validation step with predictive loss logging.
        """
        videos = batch["videos"]  # (B, T, C, H, W)
        actions = batch.get("actions", None)
        
        batch_size, n_frames = videos.shape[:2]
        videos_flat = rearrange(videos, "b t c h w -> (b t) c h w")
        
        reconstructions, posterior = self(videos_flat)
        
        # VAE losses
        aeloss, log_dict_ae = self.loss(
            videos_flat,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )
        
        discloss, log_dict_disc = self.loss(
            videos_flat,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )
        
        self.log("val/rec_loss", log_dict_ae["val/rec_loss"], sync_dist=True)
        self.log_dict(log_dict_ae, sync_dist=True)
        self.log_dict(log_dict_disc, sync_dist=True)
        
        # Predictive loss
        if self.predictor_enabled and actions is not None and n_frames > 1:
            latent_means = posterior.mean
            latent_means = rearrange(
                latent_means, "(b t) c h w -> b t c h w", b=batch_size, t=n_frames
            )
            
            mu_t = latent_means[:, :-1]
            mu_t1_target = latent_means[:, 1:].detach()
            a_t = actions[:, :-1]
            
            mu_t_flat = rearrange(mu_t, "b t c h w -> (b t) c h w")
            a_t_flat = rearrange(a_t, "b t d -> (b t) d")
            mu_t1_target_flat = rearrange(mu_t1_target, "b t c h w -> (b t) c h w")
            
            mu_t1_pred = self.predictor(mu_t_flat, a_t_flat)
            pred_loss = F.mse_loss(mu_t1_pred, mu_t1_target_flat)
            
            self.log("val/pred_loss", pred_loss, sync_dist=True)
        
        # Validation metrics
        validation_metrics = get_validation_metrics_for_videos(
            *map(
                lambda x: rearrange(x, "(b t) c h w -> t b c h w", b=batch_size)
                .contiguous()
                .detach(),
                (videos_flat, reconstructions),
            ),
            fid_model=self.fid_model,
        )
        
        self.log_dict(
            {f"val/{k}": v for k, v in validation_metrics.items()},
            prog_bar=True,
            sync_dist=True,
        )
        
        if batch_idx == 0:  # log visualizations
            videos_vis, reconstructions_vis = (
                self._rearrange_and_unnormalize(x, batch_size).detach().cpu()
                for x in (videos_flat, reconstructions)
            )
            if self.logger is not None:
                log_video(
                    reconstructions_vis,
                    videos_vis,
                    step=self.global_step,
                    namespace="reconstruction_vis",
                    logger=self.logger.experiment,
                )
    
    def configure_optimizers(self):
        """Configure optimizers including predictor parameters."""
        lr = self.learning_rate
        
        # Autoencoder parameters
        ae_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters())
        )
        
        # Add predictor parameters if enabled
        if self.predictor_enabled:
            ae_params += list(self.predictor.parameters())
        
        opt_ae = torch.optim.Adam(
            ae_params,
            lr=lr,
            betas=(0.5, 0.9),
        )
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], []


class ImageVAE(VAE):
    """
    Pretrained ImageVAE model that can be used to encode and decode images.
    Can be used to load pretrained models from custom checkpoints or huggingface repository.
    """

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        ddconfig, embed_dim = cfg.ddconfig, cfg.embed_dim
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> VAE:
        if path.startswith("diffuser:"):
            # e.g. diffuser:madebyollin/sdxl-vae-fp16-fix (from HuggingFace)
            path = path.replace("diffuser:", "")
            return cls._from_pretrained_diffuser(path, **kwargs)
        return cls._from_pretrained_custom(path)

    @classmethod
    def _from_pretrained_custom(cls, path: str) -> "ImageVAE":
        if is_wandb_run_path(path):
            path = wandb_to_local_path(path)
        elif is_hf_path(path):
            path = hf_to_local_path(path)
        checkpoint = torch.load(path, map_location="cpu")
        # FIXME: temporary fix for vaes trained with older versions of the code (for minecraft VAE)
        if "cfg" not in checkpoint:
            checkpoint["cfg"] = OmegaConf.load(
                "configurations/algorithm/image_vae.yaml"
            )
            checkpoint["cfg"].ddconfig.resolution = 256
        cfg = checkpoint["cfg"]
        model = cls(cfg)
        # filter out checkpoint state_dict
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            if k.startswith("loss"):
                del state_dict[k]
        model.load_state_dict(state_dict)
        return model

    @classmethod
    def _from_pretrained_diffuser(cls, path: str, **kwargs) -> VAE:
        vae = DiffuserImageVAE.from_pretrained(path, **kwargs)
        return diffuser_to_custom(vae)

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec


def diffuser_to_custom(vae: DiffuserImageVAE) -> VAE:
    """
    Modify DiffuserImageVAE to be compatible with VAE abstract class
    """

    def wrap_encode(encode: Callable) -> Callable:
        def wrapped_encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
            return encode(x).latent_dist

        return wrapped_encode

    def wrap_decode(decode: Callable) -> Callable:
        def wrapped_decode(self, z: torch.Tensor) -> torch.Tensor:
            return decode(z).sample

        return wrapped_decode

    def wrapped_forward(
        self, sample: torch.Tensor, sample_posterior: bool = True
    ) -> Tuple[torch.Tensor, DiagonalGaussianDistribution]:
        posterior = self.encode(sample)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    vae.encode = types.MethodType(wrap_encode(vae.encode), vae)
    vae.decode = types.MethodType(wrap_decode(vae.decode), vae)
    vae.forward = types.MethodType(wrapped_forward, vae)

    return vae
