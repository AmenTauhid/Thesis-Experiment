# Conditional DDPM for Strabismus Detection

This project implements a Conditional Denoising Diffusion Probabilistic Model (DDPM) for generating eye images conditioned on class labels: Strabismus vs Normal.

## Dataset Structure

The dataset contains 314 eye images organized into 5 categories:
- **Normal**: 105 images
- **Strabismus** (4 subtypes combined): 209 images
  - ESOTROPIA: 63 images
  - EXOTROPIA: 50 images
  - HYPERTROPIA: 51 images
  - HYPOTROPIA: 45 images

For binary classification, all strabismus subtypes are grouped as class 0, and normal images as class 1.

## Hardware Requirements

- **GPU**: NVIDIA RTX A6000 (or similar with 12GB+ VRAM)
- **CUDA**: 12.4
- **Python**: 3.11.14
- **Memory**: 16GB+ RAM recommended

## Setup Instructions

### 1. Activate Your Conda Environment

```bash
conda activate your_env_name
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify PyTorch CUDA Installation

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook conditional_ddpm_strabismus.ipynb
```

## Notebook Overview

The notebook is organized into 14 sections:

1. **Setup and Imports** - Import libraries and check GPU availability
2. **Configuration** - Set hyperparameters and paths
3. **Dataset Preparation** - Custom dataset class for loading images
4. **Visualize Dataset** - Display sample images from each class
5. **DDPM Noise Scheduler** - Implement noise schedule for diffusion
6. **U-Net Model** - Conditional U-Net architecture
7. **Training Functions** - Training loop and utilities
8. **Initialize Training** - Setup optimizer and scheduler
9. **Training Loop** - Main training execution
10. **Plot Training History** - Visualize loss curves
11. **Generate Final Samples** - Create samples from trained model
12. **Load Checkpoint** - Resume training from checkpoint
13. **Utilities** - Helper functions for generating class-specific samples
14. **Export Model** - Save final model for inference

## Key Features

### Model Architecture
- **Conditional U-Net** with residual blocks and self-attention
- **Time embeddings** using sinusoidal position encoding
- **Class embeddings** for conditional generation
- **Base channels**: 64 with channel multipliers (1, 2, 4, 8)
- **Total parameters**: ~10M parameters

### Training Configuration
- **Image size**: 64x64 (configurable to 128 or 256)
- **Batch size**: 16 (adjust based on GPU memory)
- **Timesteps**: 1000
- **Epochs**: 500
- **Learning rate**: 2e-4 with cosine annealing
- **Optimizer**: AdamW with weight decay

### Data Augmentation
- Random horizontal flip
- Resize to target size
- Normalize to [-1, 1]

## Usage

### Training from Scratch

1. Open the notebook in Jupyter
2. Run cells sequentially from Section 1-9
3. Monitor training progress in the progress bar
4. Samples will be generated every 25 epochs
5. Checkpoints saved every 50 epochs

### Generate Samples

After training, use the utility functions:

```python
# Generate 16 strabismus samples
strab_samples, _ = generate_class_samples(model, scheduler, class_label=0, num_samples=16)

# Generate 16 normal samples
normal_samples, _ = generate_class_samples(model, scheduler, class_label=1, num_samples=16)
```

### Resume Training from Checkpoint

Modify and run Section 12:

```python
checkpoint_path = CHECKPOINT_DIR / "checkpoint_epoch_0100.pt"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## Output Directory Structure

```
outputs/
├── checkpoints/          # Model checkpoints
│   ├── checkpoint_epoch_0050.pt
│   ├── checkpoint_epoch_0100.pt
│   └── ...
├── samples/              # Generated samples during training
│   ├── epoch_0001.png
│   ├── epoch_0025.png
│   └── ...
├── training_loss.png     # Training loss plot
├── training_history.json # Training metrics
├── final_samples.png     # Final generated samples
└── final_model.pt        # Final trained model
```

## Default Configuration (Updated for 256×256)

The notebook is now configured for high-quality 256×256 image generation:
- **Image size**: 256×256
- **Batch size**: 8
- **Base channels**: 128
- **Model parameters**: ~40M
- **Estimated training time**: ~40 hours for 500 epochs on RTX A6000

## Hyperparameter Tuning

### If You Encounter OOM (Out of Memory) Errors
- Reduce `BATCH_SIZE` to 4 or 6
- Reduce `base_channels` to 96 or 64
- Reduce `IMG_SIZE` to 128

### For Faster Training (Lower Quality)
- Decrease `IMG_SIZE` to 128 or 64
- Decrease `base_channels` to 64
- Increase `BATCH_SIZE` to 16 or 32
- Decrease `TIMESTEPS` to 500
- Decrease `EPOCHS` to 200-300

### For Better Convergence
- Decrease `LEARNING_RATE` to 1e-4
- Increase `EPOCHS` to 1000
- Add gradient accumulation steps

## Expected Training Time

On RTX A6000:
- **64x64 images**: ~30 seconds per epoch → ~4 hours for 500 epochs
- **128x128 images**: ~90 seconds per epoch → ~12 hours for 500 epochs
- **256x256 images**: ~5 minutes per epoch → ~40 hours for 500 epochs

## Evaluation

The model quality improves over time. Check generated samples at different epochs:
- **Early (epochs 1-50)**: Blurry, low quality
- **Mid (epochs 50-200)**: Recognizable features
- **Late (epochs 200-500)**: High quality, realistic images

## Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` (try 8, 4, or 2)
- Reduce `IMG_SIZE` (try 32 or 48)
- Reduce `base_channels` (try 32)

### Training Not Converging
- Check data preprocessing (images should be normalized to [-1, 1])
- Reduce learning rate
- Increase training epochs
- Check for NaN values in loss

### Poor Sample Quality
- Train for more epochs (500-1000)
- Increase model capacity (more channels)
- Adjust noise schedule (beta_start, beta_end)
- Ensure balanced dataset

## References

- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)

## License

This project is for research and educational purposes.
