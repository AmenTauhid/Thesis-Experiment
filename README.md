# Conditional Generative Models for Strabismus Classification

This project implements and compares two conditional generative models for medical image synthesis on a strabismus dataset:
1. **Conditional DDPM** (Denoising Diffusion Probabilistic Model)
2. **Conditional VAE** (Variational Autoencoder)

Both models are trained on the same preprocessed dataset and evaluated using identical metrics for fair comparison.

## Project Structure

```
Thesis Experiment/
├── ddpm_experiment/                    # Diffusion model experiment
│   ├── conditional_ddpm_strabismus.ipynb   # Main DDPM notebook
│   ├── requirements_ddpm.txt               # DDPM dependencies
│   ├── README_DDPM.md                      # DDPM documentation
│   ├── outputs/                            # Generated samples and checkpoints
│   ├── samples_run_1/                      # Training run 1 samples
│   ├── samples_run_2/                      # Training run 2 samples
│   └── wandb/                              # Weights & Biases logs
│
├── vae_experiment/                     # VAE model experiment
│   ├── conditional_vae_strabismus.ipynb    # Main VAE notebook
│   ├── requirements_vae.txt                # VAE dependencies
│   └── README_VAE.md                       # VAE documentation
│
├── data/                               # Shared dataset (314 images)
│   ├── STRABISMUS/                         # 209 strabismus images
│   └── NORMAL/                             # 105 normal images
│
└── preprocess_dataset.py               # Dataset preprocessing script
```

## Dataset

- **Total Images**: 314
  - Strabismus: 209 (66.6%)
  - Normal: 105 (33.4%)
- **Size**: 256x256 RGB
- **Normalization**: [-1, 1]
- **Location**: `data/`

## Quick Start

### 1. DDPM Experiment

```bash
cd ddpm_experiment
pip install -r requirements_ddpm.txt
jupyter notebook conditional_ddpm_strabismus.ipynb
```

**Expected Training Time**: 20-30 hours (1000 epochs)
**Expected Results**: FID ~20-40

### 2. VAE Experiment

```bash
cd vae_experiment
pip install -r requirements_vae.txt
jupyter notebook conditional_vae_strabismus.ipynb
```

**Expected Training Time**: 2-3 hours (200 epochs)
**Expected Results**: FID ~40-60

## Model Comparison

| Aspect | DDPM | VAE | Winner |
|--------|------|-----|--------|
| **Sample Quality (FID)** | 20-40 | 40-60 | DDPM |
| **Training Time** | 20-30 hours | 2-3 hours | VAE (10x faster) |
| **Sampling Speed** | 5-10s/image | 0.01s/image | VAE (500x faster) |
| **Memory Usage** | ~6GB | ~4GB | VAE |
| **Reconstruction** | No | Yes | VAE |
| **Latent Space** | Implicit | Explicit | VAE |

### When to Use Each Model

**Use DDPM when**:
- Best possible image quality is required
- Computational resources are available
- Inference speed is not critical
- State-of-the-art results needed

**Use VAE when**:
- Fast inference is critical
- Need reconstruction capabilities
- Limited computational resources
- Latent space interpretability matters
- Quick experimentation/prototyping

## Evaluation Metrics

Both experiments use identical evaluation protocols:

1. **FID (Fréchet Inception Distance)** - Lower is better
   - Separate FID for Strabismus and Normal classes
   - Combined FID

2. **PSNR (Peak Signal-to-Noise Ratio)** - Higher is better
   - VAE only (measures reconstruction quality)

3. **SSIM (Structural Similarity Index)** - Higher is better
   - VAE only (0-1 scale)

4. **Inception Score** - Higher is better
   - Both models

5. **Latent Space Visualization**
   - VAE: t-SNE plot of latent representations
   - DDPM: Implicit latent space

## Training Progress Tracking

Both experiments use **Weights & Biases** (wandb) for experiment tracking:
- Loss curves
- Generated samples
- Evaluation metrics
- Model checkpoints

Login to wandb:
```bash
wandb login
```

## Results Location

### DDPM Results
```
ddpm_experiment/
├── outputs/
│   ├── checkpoints/
│   │   ├── best_checkpoint.pt
│   │   └── checkpoint_epoch_XXXX.pt
│   ├── samples/
│   │   └── epoch_XXXX.png
│   └── evaluation/
│       └── evaluation_results_per_class.json
```

### VAE Results
```
vae_experiment/
├── checkpoints_vae/
│   ├── vae_best_checkpoint.pt
│   ├── vae_per_class_fid.json
│   ├── vae_all_metrics.json
│   ├── final_samples_strabismus.png
│   ├── final_samples_normal.png
│   └── latent_space_tsne.png
```

## Key Features

### Both Models
- Class conditioning (Strabismus vs Normal)
- Per-class evaluation
- Separate sample generation by class
- Weighted sampling for class imbalance
- Comprehensive evaluation metrics

### DDPM Specific
- Cosine beta schedule (optimal for 256x256)
- EMA (Exponential Moving Average) for better quality
- U-Net architecture with attention
- 1000 diffusion timesteps

### VAE Specific
- KL annealing (prevents posterior collapse)
- Explicit latent space (256-dimensional)
- Fast single-pass generation
- Reconstruction capability
- t-SNE visualization

## Hardware Requirements

- **GPU**: NVIDIA GPU with 6GB+ VRAM (tested on RTX A6000)
- **RAM**: 16GB+ recommended
- **Storage**: 5GB+ for checkpoints and samples

## Dependencies

Core dependencies (both experiments):
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- wandb >= 0.15.0
- torch-fidelity >= 0.3.0
- scikit-image >= 0.20.0
- matplotlib >= 3.7.0

See `requirements_ddpm.txt` and `requirements_vae.txt` for complete lists.

## Preprocessing

The dataset preprocessing script is available at the root level:

```bash
python preprocess_dataset.py
```

This script:
- Resizes images to 256x256
- Normalizes to [-1, 1]
- Organizes into data/STRABISMUS/ and data/NORMAL/ folders

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{conditional_generative_strabismus2025,
  title={Conditional Generative Models for Strabismus Classification: DDPM vs VAE},
  author={Your Name},
  year={2025},
  howpublished={GitHub Repository}
}
```

## References

### DDPM Papers
- Ho et al. (2020). "Denoising Diffusion Probabilistic Models"
- Nichol & Dhariwal (2021). "Improved Denoising Diffusion Probabilistic Models"
- Recent medical imaging DDPM papers (2024-2025)

### VAE Papers
- Kingma & Welling (2013). "Auto-Encoding Variational Bayes"
- Sohn et al. (2015). "Learning Structured Output Representation using Deep Conditional Generative Models"
- Bowman et al. (2016). "Generating Sentences from a Continuous Space" (KL Annealing)

## License

[Your License Here]

## Contact

For questions or issues, please open an issue on the GitHub repository.

---

**Last Updated**: November 2025
**Status**: Both experiments ready for training
**Next Steps**: Run training and compare results
