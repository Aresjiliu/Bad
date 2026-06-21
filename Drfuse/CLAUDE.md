# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based deep learning research project for multimodal remote sensing data fusion. The project implements "Drfuse" - a model that fuses different types of remote sensing data (HSI, LiDAR, SAR, Multispectral) for improved land cover classification.

## Key Commands

### Training
```bash
# Basic training command
python main.py --train_epoch 300 --gpu 0 --identity 'experiment_name'

# Run with output logging
python main.py --train_epoch 300 --gpu 0 --identity 'experiment_name' > experiment_name.out 2>&1 &
```

### Common Parameters
- `--pair_modalities`: Data modality combinations (default: 'hsi+lidar')
- `--train_epoch`: Training epochs (default: 300)
- `--batch_size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 0.0005)
- `--class_num`: Number of classes (default: 15)
- `--gpu`: GPU device selection

## Architecture Overview

### Core Model Components

1. **Drfuse Model** (`models1.py`): Main multimodal fusion architecture that combines different remote sensing data types

2. **SFDModule (Shared Feature Decomposition)**: Decomposes features into shared and modality-specific components for better fusion

3. **SpectralSqueeze**: Dynamic spectral band suppression mechanism for hyperspectral data processing

4. **SS_DecoupledBlock**: Spectral-spatial decoupled convolution for efficient hyperspectral feature extraction

5. **OrientedGradientBlock**: Multi-directional gradient enhancement specifically designed for radar data

6. **HierarchicalFusionClassifier**: Combines shared and modality-specific features for final classification

### Training Pipeline

- **Entry Point**: `main.py` - Orchestrates the entire training process
- **Configuration**: `config.py` - Contains all hyperparameters and settings
- **Training Logic**: `train_model.py` - Implements training loop, loss functions, and evaluation metrics
- **Loss Functions**: Includes intra-class compactness loss and cross-entropy loss

### Key Dependencies

The project requires several missing components that need to be provided:
- `src.huston2013_dataloader`: Custom dataloader for Houston2013 dataset
- `lib.model_develop_utils`: Contains learning rate schedulers and model utilities
- Data directory: `../data/Huston2013/` (expected location)
- Output directories: `../output/models/`, `../output/logs/`, `../output/depose/`

### Technical Notes

- Uses cosine annealing learning rate scheduler
- Implements custom loss functions for intra-class compactness
- Supports multiple GPU training
- Uses tqdm for progress tracking
- Saves model checkpoints and training logs