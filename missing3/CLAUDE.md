# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based deep learning research project implementing "DrFuse: Learning Disentangled Representation for Classification" - a multimodal fusion approach for remote sensing data (hyperspectral imaging and LiDAR).

## Key Commands

### Training
```bash
# Basic training with default settings
python main.py

# Training with specific configuration
python main.py --train_epoch 300 --gpu 0 --identity 'res0' --use_md_loss True

# Run multiple experimental configurations
bash missing.sh
```

### Configuration
All training parameters are configured in `config.py`. Key parameters include:
- `--train_epoch`: Number of training epochs (default: 300)
- `--gpu`: GPU device ID (default: 0)
- `--identity`: Experiment identifier for output naming
- `--use_md_loss`: Enable modality dropout loss
- `--use_id_loss`: Enable identity classification loss
- `--use_intra_loss`: Enable intra-class compactness loss

## Architecture Overview

### Core Model Components (`models1.py`)
- **Mdfuse**: Main multimodal fusion model combining HSI (144 channels) and LiDAR (1 channel)
- **SFDModuleMD**: Shared Feature Disentanglement Module for modality-invariant representations
- **CrossAttentionFusion**: Cross-modal attention mechanism
- **CozeFusionClassifier**: Enhanced fusion classifier

### Loss Functions
The model supports multiple configurable loss functions:
- **MD Loss** (`SpatialMDLoss`): Modality dropout for robustness
- **ID Loss** (`IdentityClassificationLoss`): Identity preservation
- **Intra Loss** (`IntraClassCompactnessLoss`): Intra-class compactness
- **DrFuse Loss** (`DrfuseLoss`): Main fusion objective

### Data Flow
1. **Input**: Houston 2013 dataset with HSI (144 bands) + LiDAR (1 band)
2. **Preprocessing**: Data loading via `src/huston2013_dataloader.py`
3. **Training**: Main loop in `train_model.py` with evaluation metrics
4. **Output**: Models, logs, and visualizations saved to `../output/`

### Key Dependencies
- PyTorch (torch, torch.nn, torch.optim)
- NumPy, scikit-learn
- matplotlib, seaborn (visualization)
- CUDA support required for GPU training

## Important Patterns

### Adding New Modalities
The model supports MS (8 channels) and SAR (4 channels) in addition to HSI and LiDAR. Modify the modality configuration in `config.py` and update the model architecture accordingly.

### Loss Function Combinations
Loss functions can be individually enabled/disabled via config flags. The total loss is a weighted combination of enabled losses.

### Output Structure
Training outputs are organized in `../output/` with subdirectories named by the experiment identity parameter.

## Dataset Information
- **Houston 2013**: 15 land cover classes
- **Spatial dimensions**: Variable patch sizes
- **Spectral bands**: 144 (HSI) + 1 (LiDAR)
- **Training/Testing**: Configurable splits via dataloader