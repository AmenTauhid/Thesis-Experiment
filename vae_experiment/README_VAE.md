# Conditional VAE for Strabismus Classification

This experiment implements a **Conditional Variational Autoencoder (CVAE)** for the strabismus dataset, designed for comparison with the DDPM approach.

## Overview

Conditional VAE is a generative model that learns a probabilistic mapping between images and a latent space, conditioned on class labels. Unlike DDPMs which iteratively denoise images, VAEs provide instant generation and explicit latent space structure.

## Architecture

### Encoder
- Input: 256x256 RGB images
- Progressive downsampling: 256 → 128 → 64 → 32 → 16 → 8
- 5 convolutional layers: [32, 64, 128, 256, 512] channels
- Output: μ and log(σ²) for 256-dimensional latent space
- Reparameterization: z = μ + σ * ε

### Decoder
- Input: 256-dim latent vector + 128-dim class embedding
- Progressive upsampling: 8 → 16 → 32 → 64 → 128 → 256
- 5 transposed convolutional layers: [512, 256, 128, 64, 32] channels
- Output: Reconstructed 256x256 RGB image (Tanh activation)

### Class Conditioning
- Embedding layer: 2 classes → 128 dimensions
- Concatenated with latent vector before decoder
- Enables class-specific generation

## Key Features

### 1. KL Annealing
- Prevents posterior collapse in early training
- β schedule: 1e-5 → 1e-3 over first 20 epochs
- Balances reconstruction vs. latent structure

### 2. Weighted Sampling
- Handles class imbalance (209 strabismus, 105 normal)
- Weight ratio: ~1:2 (normal:strabismus)
- Prevents bias toward majority class

### 3. Early Stopping
- Patience: 50 epochs
- Tracks validation loss
- Saves best checkpoint automatically

### 4. Per-Class Evaluation
- Separate FID for Strabismus and Normal classes
- Matches DDPM evaluation protocol
- Enables fair comparison

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Latent Dimension | 256 | Optimal for 256x256 images (research-backed) |
| Learning Rate | 1e-3 | Standard for VAE training |
| Batch Size | 8 | Matches DDPM for fair comparison |
| Epochs | 200 | Much faster than DDPM's 1000 |
| KL β Start | 1e-5 | Prevents early posterior collapse |
| KL β End | 1e-3 | Standard KL weight |
| KL Annealing | 20 epochs | Gradual increase period |
| Early Stop Patience | 50 epochs | Prevents overfitting |
| Optimizer | Adam | β1=0.9, β2=0.999 |
| Scheduler | ReduceLROnPlateau | Factor=0.5, Patience=10 |
| Dropout | 0.1 | Regularization for small dataset |

## Dataset

- **Source**: `../data/`
- **Total Images**: 314
  - Strabismus: 209 (66.6%)
  - Normal: 105 (33.4%)
- **Split**: 80% train, 20% validation
- **Preprocessing**:
  - Resize to 256x256
  - Normalize to [-1, 1]
  - Random horizontal flips (training only)

## Training

### Loss Function
```
Total Loss = Reconstruction Loss + β * KL Divergence

Reconstruction Loss = MSE(original, reconstructed)
KL Divergence = -0.5 * sum(1 + log(σ²) - μ² - σ²)
```

### Training Time
- Expected: 2-3 hours on single GPU (RTX 3090/4090)
- ~36 seconds per epoch
- Much faster than DDPM (20-30 hours)

### Monitoring
- Loss curves (total, reconstruction, KL)
- Sample quality visualization
- Validation reconstruction quality
- Weights & Biases logging

## Evaluation Metrics

### 1. FID (Fréchet Inception Distance)
- Per-class FID for Strabismus and Normal
- Combined FID
- Lower is better (measures distribution similarity)

### 2. PSNR (Peak Signal-to-Noise Ratio)
- Measures reconstruction quality
- Typical range: 20-25 dB
- Higher is better

### 3. SSIM (Structural Similarity Index)
- Perceptual reconstruction quality
- Range: 0-1
- Higher is better (typically 0.70-0.85)

### 4. Inception Score
- Measures sample quality and diversity
- Higher is better

### 5. Latent Space Visualization
- t-SNE plot of latent representations
- Class separation analysis
- Validates learned structure

## Expected Results

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| FID Strabismus | 40-60 | Higher than DDPM (20-30) |
| FID Normal | 45-65 | May be higher due to less data |
| PSNR | 20-25 dB | Reconstruction quality |
| SSIM | 0.70-0.85 | Perceptual similarity |
| Inception Score | 2.0-3.0 | Medical images have lower IS |
| Training Time | 2-3 hours | 10x faster than DDPM |
| Sampling Speed | 0.01s/image | 500-1000x faster than DDPM |

## VAE vs DDPM Comparison

### VAE Advantages
- **10x faster training**: 2-3 hours vs 20-30 hours
- **500x faster sampling**: 0.01s vs 5-10s per image
- **Lower memory**: ~4GB vs ~6GB
- **Reconstruction capability**: Can encode/decode images
- **Explicit latent space**: Good for interpolation and manipulation
- **Deterministic**: Same input → same output (when using μ)

### DDPM Advantages
- **Better quality**: Lower FID scores (20-40 vs 40-60)
- **More diverse samples**: Stochastic diffusion process
- **Fine details**: Better at capturing subtle features
- **State-of-the-art**: Currently best for image generation

### When to Use VAE
- Fast inference is critical
- Need reconstruction capabilities
- Limited computational resources
- Latent space interpretability matters
- Quick experimentation/prototyping
- Real-time applications

### When to Use DDPM
- Best possible quality is required
- Computational resources available
- Inference speed not critical
- State-of-the-art results needed
- Publication/production quality

## Usage

### Setup
```bash
# Install dependencies
pip install -r requirements_vae.txt

# Verify dataset exists
ls ../data/
```

### Training
```bash
# Open Jupyter notebook
jupyter notebook conditional_vae_strabismus.ipynb

# Run cells sequentially:
# 1. Cells 0-6: Setup and architecture
# 2. Cells 7-10: Training (monitor progress)
# 3. Cell 11: View training history
# 4. Cell 12: Generate final samples
```

### Evaluation
```bash
# Run evaluation cells:
# Cell 13: Setup evaluation metrics
# Cell 14: Per-class FID (separate for Strabismus/Normal)
# Cell 15: PSNR, SSIM, Inception Score
# Cell 16: Latent space t-SNE visualization
# Cells 17-18: Compare with DDPM results
```

### Results Location
```
vae_experiment/
├── checkpoints_vae/
│   ├── vae_best_checkpoint.pt          # Best model weights
│   ├── vae_final_checkpoint.pt         # Final epoch weights
│   ├── vae_per_class_fid.json          # FID scores by class
│   ├── vae_all_metrics.json            # All evaluation metrics
│   ├── final_samples_strabismus.png    # Generated samples
│   ├── final_samples_normal.png        # Generated samples
│   ├── final_reconstructions.png       # Reconstruction quality
│   └── latent_space_tsne.png           # Latent space visualization
```

## Troubleshooting

### Issue: Posterior Collapse
**Symptom**: KL loss drops to ~0, reconstructions are blurry
**Solution**:
- Ensure KL annealing is enabled (β starts at 1e-5)
- Increase β_end if needed (e.g., 1e-2)
- Extend annealing period (e.g., 30 epochs)

### Issue: Poor Reconstruction Quality
**Symptom**: PSNR < 15 dB, SSIM < 0.5
**Solution**:
- Decrease β (try 5e-4 instead of 1e-3)
- Increase latent dimension (try 512)
- Check normalization (should be [-1, 1])

### Issue: High FID Scores
**Symptom**: FID > 100
**Solution**:
- Train longer (extend epochs)
- Increase model capacity (more channels)
- Check class balance (use weighted sampling)
- Verify samples are properly denormalized

### Issue: Class Imbalance
**Symptom**: Model generates mostly Strabismus samples
**Solution**:
- Verify WeightedRandomSampler is used
- Check class weights are correct (inverse frequency)
- Generate samples separately by class (cells 12, 14)

### Issue: GPU Out of Memory
**Symptom**: CUDA OOM error
**Solution**:
- Reduce batch size (try 4 or 2)
- Reduce latent dimension (try 128)
- Use gradient accumulation
- Enable mixed precision training (add to training loop)

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{cvae_strabismus2025,
  title={Conditional VAE for Strabismus Classification},
  author={Your Name},
  year={2025},
  howpublished={GitHub Repository}
}
```

## References

1. Kingma & Welling (2013). "Auto-Encoding Variational Bayes"
2. Sohn et al. (2015). "Learning Structured Output Representation using Deep Conditional Generative Models"
3. Bowman et al. (2016). "Generating Sentences from a Continuous Space" (KL Annealing)
4. Recent VAE research for medical imaging (2024-2025)

## Project Structure

```
vae_experiment/
├── conditional_vae_strabismus.ipynb    # Main notebook (31 cells)
├── requirements_vae.txt                # Dependencies
├── README_VAE.md                       # This file
├── create_vae_notebook.py              # Notebook generation script
├── add_remaining_cells.py              # Training cells script
├── add_evaluation_cells.py             # Evaluation cells script
└── checkpoints_vae/                    # Created during training
```

## Contact

For questions or issues, please open an issue on the GitHub repository.

---

**Last Updated**: 2025
**Status**: Ready for experimentation
**Estimated Runtime**: 2-3 hours training + 30 min evaluation
