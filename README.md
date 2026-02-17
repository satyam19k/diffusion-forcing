<h1 align="center">Diffusion-Forcing-Aware JEPA-VAE for Minecraft Video Generation</h1>

This repository extends the [**Diffusion Forcing Transformer (DFoT)**](https://arxiv.org/abs/2502.06764) framework with **Diffusion-Forcing-Aware JEPA-VAE**, a representation learning approach that shapes VAE latents to be action-conditioned and future-predictive while remaining usable for latent video diffusion. Our method integrates a JEPA-style predictive objective applied to VAE latents, trained with teacher forcing to avoid compounding errors, and jointly optimizes diffusion loss, JEPA latent prediction loss, and reconstruction regularization.


## 🎯 Key Contributions

- **JEPA-Style Predictive Objective**: We propose a JEPA-style predictive objective applied to VAE latents to learn action-conditioned, future-predictive representations, addressing representation aliasing concerns for downstream control.
- **Joint Training Paradigm**: We integrate JEPA-VAE training into DFoT-style latent video diffusion, leveraging Diffusion Forcing's per-token noising paradigm for robust long-horizon continuous rollouts.
- **Diffusion-Aware JEPA**: We introduce a diffusion-aware JEPA variant that trains predictiveness under partially noised history, matching DFoT's masking-by-noising regime.
- **Training Protocols**: We present both offline and joint-DFoT training protocols with structured ablations (pairwise vs. trajectory training, teacher forcing vs. autoregressive latent rollouts, noised-context on/off, and freezing vs. finetuning VAE components).

## 🏗️ Architecture Overview

### JEPA-VAE Architecture

The JEPA-VAE architecture consists of three main components:

1. **VAE Encoder (Eψ)**: Encodes each frame `xt` into a latent state `st = vec(μψ(xt))`
2. **Action Encoder (gη)**: Embeds actions `at` into a learned representation `ut = gη(at)`
3. **Latent Predictor (fθ)**: A causal Transformer that predicts the next latent state from history:
   ```
   ŝt+1 = fθ({s0 ∥ u0, ..., st ∥ ut})
   ```

The JEPA objective matches predicted latents to stop-gradient encoder targets:
```
LJEPA = Σt d(ŝt+1, sg(st+1))
```

### Training Regimes

#### 1. Offline JEPA-VAE Fine-tuning

In the **offline setting**, we fine-tune the VAE encoder `Eψ` (decoder frozen) and train the JEPA predictor `fθ` and action encoder `gη` **without updating DFoT**. This isolates whether "better latents" transfer to a fixed diffusion model.

**Offline Training Strategies:**
- **Pairwise-TF**: One-step JEPA from tuples `(xt, at, xt+1)`
- **Rollout-TF**: Trajectory JEPA with teacher forcing (conditions on ground-truth latent history)
- **Rollout-AR**: Trajectory JEPA with autoregressive predictor unrolling

**Key Finding**: Offline JEPA-VAE fine-tuning alone does not transfer cleanly to diffusion-based generation due to latent distribution/geometry mismatch when swapping an updated encoder into a fixed pretrained DFoT denoiser.

#### 2. Joint DFoT + JEPA Training

In the **joint training setting**, we simultaneously optimize the diffusion model and the JEPA-shaped representation so that the latent space is simultaneously good for denoising-based video generation and for action-conditioned predictiveness:

```
min LDFoT + λJ LJEPA + λR Lrec
```

**Why Joint Training Works:**
- DFoT co-adapts to the evolving latent space, avoiding distribution mismatch
- The representation is explicitly shaped for action-conditioned predictiveness
- Teacher forcing provides stable gradients without compounding error artifacts

**Training Modes:**
- **JEPA-0R**: `LDFoT + λJ LJEPA` (no explicit reconstruction)
- **JEPA+R**: `LDFoT + λJ LJEPA + λR Lrec` (with reconstruction regularization)
- **JEPA-AR**: Autoregressive JEPA predictor (ablation, typically performs worse)

## 🚀 Quick Start

### Setup

#### 1. Create a conda environment and install dependencies:
```bash
conda create python=3.10 -n dfot
conda activate dfot
pip install -r requirements.txt
```
### Joint DFoT + JEPA Training

Train the model with joint diffusion-aware JEPA training. This command jointly optimizes DFoT, JEPA predictor, and VAE encoder:

```bash
python main.py \
  +name=jepa_minecraft_2xa100_b \
  experiment=video_generation \
  dataset=minecraft \
  algorithm=dfot_video_jepa \
  dataset_experiment=minecraft_video_generation_jepa \
  wandb.entity=sk12075-new-york-university \
  wandb.project=dfot \
  wandb.mode=online \
  dataset.max_frames=8 \
  dataset.context_length=4 \
  load=pretrained:DFoT_MCRAFT.ckpt \
  algorithm.checkpoint.strict=false \
  algorithm.checkpoint.reset_optimizer=true \
  @DiT/B \
  @diffusion/continuous \
  experiment.training.batch_size=1 \
  experiment.validation.batch_size=1 \
  algorithm.vae.batch_size=1 \
  experiment.training.max_epochs=15 \
  experiment.training.checkpointing.every_n_train_steps=100 \
  experiment.training.checkpointing.every_n_epochs=null \
  dataset.subdataset_size=null \
  experiment.validation.limit_batch=0 \
  algorithm.jepa.loss_weight=0 \
  algorithm.jepa.recon_loss_weight=0.5 \
  experiment.find_unused_parameters=true \
  +experiment.training.checkpointing.save_top_k=-1 \
  algorithm.jepa.training_mode=teacher_forcing
```

```bash
python main.py \
  +name=jepa_minecraft_tf \
  experiment=video_generation \
  dataset=minecraft \
  algorithm=dfot_video_jepa \
  dataset_experiment=minecraft_video_generation_jepa \
  wandb.entity=sk12075-new-york-university \
  wandb.project=debug \
  wandb.mode=online \
  dataset.max_frames=16 \
  dataset.context_length=8 \
  load=pretrained:DFoT_MCRAFT.ckpt \
  algorithm.checkpoint.strict=false \
  algorithm.checkpoint.reset_optimizer=true \
  @DiT/B \
  @diffusion/continuous \
  experiment.training.batch_size=1 \
  experiment.validation.batch_size=1 \
  algorithm.vae.batch_size=1 \
  experiment.training.max_epochs=15 \
  experiment.training.checkpointing.every_n_train_steps=2000 \
  experiment.training.checkpointing.every_n_epochs=null \
  dataset.subdataset_size=null \
  experiment.validation.limit_batch=0 \
  algorithm.jepa.loss_weight=0.5 \
  algorithm.jepa.recon_loss_weight=0.1 \
  experiment.find_unused_parameters=true \
  +experiment.training.checkpointing.save_top_k=-1 \
  algorithm.jepa.training_mode=teacher_forcing \
  experiment.validation.limit_batch=10 \
  experiment.validation.batch_size=5 \
  experiment.validation.val_every_n_step=2000 \
  experiment.validation.val_every_n_epoch=null


```


**Key Parameters:**
- `algorithm.jepa.loss_weight`: Weight for JEPA loss (set to 0 to disable JEPA, >0 to enable)
- `algorithm.jepa.recon_loss_weight`: Weight for reconstruction regularization (0.5 recommended)
- `algorithm.jepa.training_mode`: Set to `teacher_forcing` (recommended) or `autoregressive` (ablation)
- `load=pretrained:DFoT_MCRAFT.ckpt`: Loads pretrained DFoT checkpoint (set to your checkpoint path)

### Validation/Inference

Run validation/inference with a trained JEPA model:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python main.py \
  +name=jepa_inference \
  experiment=video_generation \
  dataset=minecraft \
  algorithm=dfot_video_jepa \
  dataset_experiment=minecraft_video_generation_jepa \
  @DiT/B \
  @diffusion/continuous \
  wandb.entity=local \
  wandb.mode=disabled \
  load='/scratch/sk12075/diffusion-forcing/outputs/2026-02-09/17-41-45/checkpoints/epoch0-step100.ckpt' \
  algorithm.checkpoint.strict=false \
  'experiment.tasks=[validation]' \
  experiment.validation.batch_size=1 \
  dataset.num_eval_videos=20 \
  dataset.max_frames=8 \
  dataset.context_length=4 \
  dataset.n_frames=8 \
  experiment.find_unused_parameters=true \
  experiment.ema.enable=false
```

TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python main.py \
  +name=jepa_inference \
  experiment=video_generation \
  dataset=minecraft \
  algorithm=dfot_video_jepa \
  dataset_experiment=minecraft_video_generation_jepa \
  @DiT/B \
  @diffusion/continuous \
  wandb.entity=local \
  wandb.mode=disabled \
  load='/scratch/sk12075/diffusion-forcing/huggingface/models--kiwhansong--DFoT/snapshots/0959defb4c4fe010f84791d732cc978ad7d49fef/pretrained_models/DFoT_MCRAFT.ckpt' \
  algorithm.checkpoint.strict=false \
  'experiment.tasks=[validation]' \
  experiment.validation.batch_size=8 \
  dataset.num_eval_videos=100 \
  dataset.max_frames=32 \
  dataset.context_length=16 \
  dataset.n_frames=64 \
  experiment.find_unused_parameters=true \
  experiment.ema.enable=false

TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python main.py \
  +name=jepa_inference \
  experiment=video_generation \
  dataset=minecraft \
  algorithm=dfot_video_jepa \
  dataset_experiment=minecraft_video_generation_jepa \
  @DiT/B \
  @diffusion/continuous \
  wandb.entity=local \
  wandb.mode=disabled \
  load='/scratch/sk12075/diffusion-forcing/huggingface/models--kiwhansong--DFoT/snapshots/0959defb4c4fe010f84791d732cc978ad7d49fef/pretrained_models/DFoT_MCRAFT.ckpt' \
  algorithm.checkpoint.strict=false \
  'experiment.tasks=[validation]' \
  experiment.validation.batch_size=8 \
  dataset.num_eval_videos=50 \
  dataset.max_frames=16 \
  dataset.context_length=8 \
  dataset.n_frames=16 \
  experiment.find_unused_parameters=true \
  experiment.ema.enable=false

TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m main \
  @DiT/B \
  @diffusion/continuous \
  +name=jepa_minecraft_tf_training \
  experiment=video_generation \
  dataset=minecraft \
  algorithm=dfot_video_jepa \
  dataset_experiment=minecraft_video_generation_jepa \
  wandb.entity=local \
  wandb.mode=disabled \
  load='/scratch/sk12075/diffusion-forcing/huggingface/models--kiwhansong--DFoT/snapshots/0959defb4c4fe010f84791d732cc978ad7d49fef/pretrained_models/DFoT_MCRAFT.ckpt' \
  algorithm.checkpoint.strict=false \
  "experiment.tasks=[validation]" \
  experiment.validation.batch_size=1 \
  dataset.num_eval_videos=100 \
  dataset.max_frames=16 \
  dataset.context_length=8 \
  dataset.n_frames=16 \
  experiment.find_unused_parameters=true \
  experiment.ema.enable=false

TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m main \
  @DiT/B \
  @diffusion/continuous \
  +name=jepa_minecraft_tf_training \
  experiment=video_generation \
  dataset=minecraft \
  algorithm=dfot_video_jepa \
  dataset_experiment=minecraft_video_generation_jepa \
  wandb.entity=local \
  wandb.mode=disabled \
  load='/scratch/sk12075/diffusion-forcing/outputs/2026-02-12/01-20-07/checkpoints/epoch0-step8000.ckpt' \
  algorithm.checkpoint.strict=false \
  "experiment.tasks=[validation]" \
  experiment.validation.batch_size=1 \
  dataset.num_eval_videos=100 \
  dataset.max_frames=16 \
  dataset.context_length=8 \
  dataset.n_frames=16 \
  experiment.find_unused_parameters=true \
  experiment.ema.enable=false

/scratch/sk12075/diffusion-forcing/outputs/2026-02-12/00-15-02/checkpoints/epoch0-step300.ckpt
/scratch/sk12075/diffusion-forcing/outputs/2026-02-09/21-33-13/checkpoints/epoch0-step700.ckpt

python main.py   +name=jepa_inference   experiment=video_generation   dataset=minecraft   algorithm=dfot_video_jepa   dataset_experiment=minecraft_video_generation_jepa   @DiT/B   @diffusion/continuous   wandb.entity=local   wandb.mode=disabled   load=/scratch/yb2510/RL_Jayesh/diffusion-forcing-jepa/outputs/2025-12-13/02-05-07/checkpoints/epoch_10_step_251000_v2.ckpt   algorithm.checkpoint.strict=false   'experiment.tasks=[validation]'   experiment.validation.batch_size=1   dataset.num_eval_videos=20   dataset.max_frames=8   dataset.context_length=4   dataset.n_frames=8   experiment.find_unused_parameters=true   experiment.ema.enable=false

1python -m main +name=DFoT dataset=minecraft algorithm=dfot_video experiment=video_generation @diffusion/continuous @DiT/B load='/scratch/sk12075/diffusion-forcing/huggingface/models--kiwhansong--DFoT/snapshots/0959defb4c4fe010f84791d732cc978ad7d49fef/pretrained_models/DFoT_MCRAFT.ckpt' 'experiment.tasks=[validation]' 'algorithm.logging.metrics=[fvd]' dataset.n_frames=150 experiment.validation.batch_size=1 dataset.filter_min_len=0 wandb.entity=sk12075-new-york-university wandb.project=inference wandb.mode=online

wandb.entity=sk12075-new-york-university \
  wandb.project=dfot \
  wandb.mode=online \
**Note**: Replace `<path_to_your_checkpoint>` with the path to your trained checkpoint.


Offline VAE training command 

```bash
python -m main \
  +name=predictive_vae_training \
  algorithm=image_vae_predictive \
  experiment=video_latent_learning \
  dataset=minecraft \
  dataset_experiment=minecraft_video_latent_learning_predictive \
  dataset.max_frames=8 \
  dataset.frame_skip=2 \
  dataset.external_cond_dim=4 \
  dataset.context_length=0 \
  dataset.external_cond_stack=true \
  algorithm.pretrained_vae_path=pretrained:ImageVAE_MCRAFT.ckpt \
  experiment.training.batch_size=1 \
  experiment.training.max_steps=50000 \
  experiment.training.checkpointing.every_n_train_steps=500 \
  experiment.validation.val_every_n_step=500 \
  experiment.validation.batch_size=1 \
  wandb.entity=local \
  wandb.mode=offline \
  algorithm.predictor.lambda_pred=1000.0 \
  experiment.training.lr=1e-6 \
  +experiment.training.checkpointing.save_top_k=-1
```

Command to generate latents :- 

```bash
python -m main \
  +name=latent_generation \
  algorithm=image_vae_preprocessor \
  experiment=video_latent_preprocessing \
  dataset=minecraft \
  dataset_experiment=minecraft_video_latent_preprocessing \
  dataset.latent.suffix={Add suffix} \
  experiment.validation.dataset_splits=[validation] \
  algorithm.pretrained_path={Add model path} \
  wandb.entity=local \
  wandb.mode=offline

```

Inference command:- 
```bash
python main.py \
  +name=baseline_inference \
  experiment=video_generation \
  dataset=minecraft \
  algorithm=dfot_video \
  dataset_experiment=minecraft_video_generation \
  @DiT/B \
  @diffusion/continuous \
  wandb.entity=local \
  wandb.mode=disabled \
  load=pretrained:DFoT_MCRAFT.ckpt \
  'experiment.tasks=[validation]' \
  experiment.validation.batch_size=1 \
  dataset.num_eval_videos=50 \
  dataset.max_frames=8 \
  dataset.context_length=4 \
  dataset.n_frames=8 \
  experiment.ema.enable=false \
  dataset.latent.suffix={Add latent suffix} \
  'algorithm.vae.pretrained_path={Add model path}' \
  algorithm.logging.deterministic=1
```


### Model Weights

Pretrained model weights are available for download:

- [Offline Model Weights (Folder 1)](https://drive.google.com/drive/folders/1ZsbWxlDTVMPQ0hKkyEwccg52-NiPbXC-?usp=sharing)
- [Joint JEPA Model Weights (Folder 2)](https://drive.google.com/drive/folders/1DlesLbtgslPcmyGSjFQINlTNjqGbOUXB?usp=sharing)

## 📊 Experimental Results

### Joint Training Results

Our joint diffusion-aware JEPA training achieves:

| Method | FID ↓ | FVD ↓ | IS ↑ | LPIPS ↓ | MSE ↓ | PSNR ↑ | SSIM ↑ |
|--------|-------|-------|------|---------|-------|--------|--------|
| Recon-FT (baseline) | 83.63 | 215.25 | 3.25 | 0.408 | 0.0175 | 17.56 | 0.499 |
| **JEPA-0R (ours)** | **78.25** | **193.25** | **3.41** | **0.389** | 0.0162 | 17.90 | **0.516** |
| JEPA+R (ours) | 80.63 | 212.63 | 3.26 | 0.402 | **0.0161** | **17.94** | 0.501 |
| JEPA-AR (ablation) | 88.56 | 264.25 | 3.21 | 0.420 | 0.0217 | 16.64 | 0.484 |

**Key Findings:**
- **JEPA-0R** achieves the best temporal quality (FVD) and realism (FID)
- Joint training resolves latent distribution mismatch seen in offline fine-tuning
- Teacher forcing provides cleaner supervision than autoregressive rollouts
- Reconstruction regularization improves pixel metrics but can constrain dynamics shaping

### Offline Training Results

Offline JEPA-VAE fine-tuning with a fixed DFoT denoiser:

| Method | FID ↓ | FVD ↓ | IS ↑ | LPIPS ↓ | MSE ↓ | PSNR ↑ | SSIM ↑ |
|--------|-------|-------|------|---------|-------|--------|--------|
| Vanilla (no fine-tuning) | 82.44 | 196.75 | 3.49 | 0.372 | 0.0149 | 18.26 | 0.554 |
| Pairwise-TF | 135.63 | 329.75 | 3.45 | 0.531 | 0.0245 | 16.10 | 0.500 |
| Rollout-TF | 88.75 | 263.75 | 3.37 | 0.405 | 0.0179 | 17.45 | 0.507 |
| Rollout-AR | 121.56 | 439.75 | 3.61 | 0.535 | 0.0248 | 16.05 | 0.490 |

**Key Finding**: Offline updating the encoder hurts generation quality relative to vanilla pretrained DFoT+VAE pipeline, consistent with latent distribution/geometry mismatch.

## 🔬 Method Details

### Why Teacher Forcing?

Our goal is to learn better latents, not to deploy `fθ` as the rollout model. If we train `fθ` autoregressively (feeding back `ŝt+1`), early prediction errors compound, and gradients to the encoder become dominated by "error amplification" artifacts rather than true representational deficiency. Teacher forcing instead conditions each prediction on the ground-truth latent history `{s≤t}`, yielding:
1. A well-conditioned supervised signal for "is `st` informative enough to predict `st+1` under action `at`?"
2. Stable gradients to `Eψ` that encourage preserving transition-relevant details
3. A clean ablation axis: we still evaluate an autoregressive variant to quantify the gap

### Diffusion-Forcing Awareness

A key mismatch can arise if JEPA trains predictiveness only from clean histories, while DFoT sampling may present partially corrupted histories due to per-token noising. To align the representation objective with DFoT conditions, we optionally noise the JEPA predictor's context states using the same `(α, σ)` schedule:

```
s̃t = α(kt) st + σ(kt) ξt,  ξt ~ N(0, I)
```

and predict clean next-state targets:
```
ŝt+1 = fθ({s̃0 ∥ u0, ..., s̃t ∥ ut})
```

This trains latents to remain predictive even when history is partially "masked" by noise, mirroring DFoT's training/sampling regime.



## 📝 Acknowledgements

This repo extends [Boyuan Chen](https://boyuan.space/)'s research template [repo](https://github.com/buoyancy99/research-template) and the original [Diffusion Forcing Transformer](https://github.com/kwsong0113/diffusion-forcing-transformer) implementation. By its license, we ask you to keep the above sentences and links in `README.md` and the `LICENSE` file to credit the authors.

## 📌 Citation

If our work is useful for your research, please consider giving us a star and citing our paper:

```bibtex
@misc{song2025historyguidedvideodiffusion,
  title={History-Guided Video Diffusion}, 
  author={Kiwhan Song and Boyuan Chen and Max Simchowitz and Yilun Du and Russ Tedrake and Vincent Sitzmann},
  year={2025},
  eprint={2502.06764},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2502.06764}, 
}
```

For the JEPA-VAE extension:

```bibtex
@misc{kumar2025diffusionforcingawarejepavae,
  title={Diffusion-Forcing-Aware JEPA-VAE for Minecraft Video Generation},
  author={Satyam Kumar and Jayesh Chaudhari},
  year={2025},
  note={Extension of History-Guided Video Diffusion}
}
```
