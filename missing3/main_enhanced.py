'''
Enhanced main script for adversarial feature generation with FMC compensation
'''

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

sys.path.append('..')
from models1 import Mdfuse
from fmcnet_modules import FMCEnhancedDrFuse
from enhanced_adversarial_generation import EnhancedAdversarialFramework
from train_enhanced_adversarial import EnhancedAdversarialTrainer
from src.huston2013_dataloader import huston2013_multi_dataloader
from config import args


def seed_torch(seed=0):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_models(args, device):
    """
    Setup models based on configuration
    """
    modality_to_channel = {'hsi': 144, 'hsi1': 244, 'ms': 8, 'lidar': 1, 'sar': 4}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    # Choose model architecture based on configuration
    if args.use_fmc_enhancement:
        print("Using FMC-enhanced DrFuse model")
        model = FMCEnhancedDrFuse(
            args, modality_1_channel, modality_2_channel,
            feature_dim=args.feature_dim if hasattr(args, 'feature_dim') else 128,
            shared_dim=args.shared_dim if hasattr(args, 'shared_dim') else 64,
            specific_dim=args.specific_dim if hasattr(args, 'specific_dim') else 64
        )
    else:
        print("Using standard DrFuse model")
        model = Mdfuse(args, modality_1_channel, modality_2_channel,
                      feature_dim=args.feature_dim if hasattr(args, 'feature_dim') else 128)

    # Create adversarial framework
    if args.use_adversarial_training:
        print("Using adversarial training framework")
        adversarial_framework = EnhancedAdversarialFramework(
            shared_dim=args.shared_dim if hasattr(args, 'shared_dim') else 64,
            specific_dim=args.specific_dim if hasattr(args, 'specific_dim') else 64,
            feature_dim=args.feature_dim if hasattr(args, 'feature_dim') else 128,
            num_modalities=2
        )
    else:
        adversarial_framework = None

    return model.to(device), adversarial_framework


def setup_training_config(args):
    """
    Setup enhanced training configuration
    """
    # Add enhanced configuration parameters
    args.use_fmc_enhancement = getattr(args, 'use_fmc_enhancement', True)
    args.use_adversarial_training = getattr(args, 'use_adversarial_training', True)
    args.feature_dim = getattr(args, 'feature_dim', 128)
    args.shared_dim = getattr(args, 'shared_dim', 64)
    args.specific_dim = getattr(args, 'specific_dim', 64)
    args.compensation_mode = getattr(args, 'compensation_mode', 'adaptive')

    # Adversarial training parameters
    args.lambda_adv = getattr(args, 'lambda_adv', 1.0)
    args.lambda_rec = getattr(args, 'lambda_rec', 0.5)
    args.lambda_cls = getattr(args, 'lambda_cls', 0.3)
    args.lambda_consistency = getattr(args, 'lambda_consistency', 0.2)
    args.lambda_feature = getattr(args, 'lambda_feature', 0.1)
    args.lambda_fmc = getattr(args, 'lambda_fmc', 0.3)

    # Modality missing simulation
    args.missing_modality_rate = getattr(args, 'missing_modality_rate', 0.3)
    args.curriculum_learning = getattr(args, 'curriculum_learning', True)

    return args


def train_enhanced_model(args):
    """
    Train enhanced model with adversarial generation and FMC compensation
    """
    print("Starting enhanced training with adversarial generation and FMC compensation...")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup enhanced configuration
    args = setup_training_config(args)

    # Set random seed
    seed_torch(args.seed if hasattr(args, 'seed') else 42)

    # Load data
    print("Loading data...")
    train_loader = huston2013_multi_dataloader(train=True, args=args)
    val_loader = huston2013_multi_dataloader(train=False, args=args)

    # Setup models
    print("Setting up models...")
    model, adversarial_framework = setup_models(args, device)

    # Setup optimizer
    if args.use_adversarial_training and adversarial_framework is not None:
        # Use enhanced adversarial trainer
        trainer = EnhancedAdversarialTrainer(args, model, adversarial_framework, device)

        # Start training
        print("Starting adversarial training...")
        trainer.train(train_loader, val_loader)

    else:
        # Use standard training
        print("Starting standard training...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Standard training loop
        for epoch in range(1, args.train_epoch + 1):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch_idx, (data, targets) in enumerate(train_loader):
                if isinstance(data, (list, tuple)) and len(data) == 2:
                    x1, x2 = data[0].to(device), data[1].to(device)
                else:
                    x1 = data.to(device)
                    x2 = torch.zeros_like(x1)

                targets = targets.to(device)

                optimizer.zero_grad()

                # Forward pass
                if isinstance(model, FMCEnhancedDrFuse):
                    outputs, _ = model(x1, x2)
                else:
                    outputs = model(x1, x2)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                # Statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if batch_idx % args.log_interval == 0:
                    print(f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                          f'Loss: {total_loss/(batch_idx+1):.6f} '
                          f'Acc: {100.*correct/total:.3f}%')

            # Validation
            if val_loader is not None:
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for data, targets in val_loader:
                        if isinstance(data, (list, tuple)) and len(data) == 2:
                            x1, x2 = data[0].to(device), data[1].to(device)
                        else:
                            x1 = data.to(device)
                            x2 = torch.zeros_like(x1)

                        targets = targets.to(device)

                        if isinstance(model, FMCEnhancedDrFuse):
                            outputs, _ = model(x1, x2)
                        else:
                            outputs = model(x1, x2)

                        loss = criterion(outputs, targets)
                        val_loss += loss.item()

                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()

                print(f'Validation - Loss: {val_loss/len(val_loader):.6f} '
                      f'Acc: {100.*val_correct/val_total:.3f}%')

    print("Training completed!")


def test_missing_modality_scenarios(args):
    """
    Test model performance under different missing modality scenarios
    """
    print("Testing missing modality scenarios...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test data
    test_loader = huston2013_multi_dataloader(train=False, args=args)

    # Setup models
    model, adversarial_framework = setup_models(args, device)

    # Test scenarios
    scenarios = [
        {'name': 'Complete', 'missing_mask': [1, 1]},  # Both modalities available
        {'name': 'HSI_Missing', 'missing_mask': [0, 1]},  # HSI missing
        {'name': 'LiDAR_Missing', 'missing_mask': [1, 0]},  # LiDAR missing
        {'name': 'High_Missing_Rate', 'missing_mask': [0.3, 0.3]}  # Random missing
    ]

    results = {}

    for scenario in scenarios:
        print(f"\nTesting scenario: {scenario['name']}")

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in test_loader:
                if isinstance(data, (list, tuple)) and len(data) == 2:
                    x1, x2 = data[0].to(device), data[1].to(device)
                else:
                    x1 = data.to(device)
                    x2 = torch.zeros_like(x1)

                targets = targets.to(device)

                # Apply missing modality mask
                if scenario['name'] == 'High_Missing_Rate':
                    # Random missing for each sample
                    batch_mask = torch.rand(x1.size(0), 2) > scenario['missing_mask'][0]
                    # Ensure at least one modality
                    missing_all = (batch_mask.sum(dim=1) == 0)
                    if missing_all.any():
                        rand_modality = torch.randint(0, 2, (missing_all.sum(),))
                        batch_mask[missing_all, rand_modality] = True
                else:
                    batch_mask = torch.tensor([scenario['missing_mask']] * x1.size(0))

                batch_mask = batch_mask.float().to(device)

                # Forward pass with missing modality handling
                if isinstance(model, FMCEnhancedDrFuse):
                    outputs, _ = model(x1, x2, modality_mask=batch_mask)
                else:
                    outputs = model(x1, x2)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100. * correct / total
        results[scenario['name']] = accuracy
        print(f"{scenario['name']} Accuracy: {accuracy:.3f}%")

    return results


def main():
    """
    Main function
    """
    # Set CUDA environment
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Parse arguments
    args.log_name = args.name
    args.model_name = args.name

    # Enhanced configuration
    args.use_fmc_enhancement = True
    args.use_adversarial_training = True
    args.feature_dim = 128
    args.shared_dim = 64
    args.specific_dim = 64
    args.compensation_mode = 'adaptive'
    args.missing_modality_rate = 0.3
    args.curriculum_learning = True
    args.seed = 42

    # Training modes
    print("Select training mode:")
    print("1. Standard training")
    print("2. Adversarial training with FMC")
    print("3. Test missing modality scenarios")

    choice = input("Enter choice (1-3): ").strip()

    if choice == '1':
        args.use_adversarial_training = False
        train_enhanced_model(args)
    elif choice == '2':
        args.use_adversarial_training = True
        train_enhanced_model(args)
    elif choice == '3':
        results = test_missing_modality_scenarios(args)
        print("\nMissing Modality Test Results:")
        for scenario, accuracy in results.items():
            print(f"{scenario}: {accuracy:.3f}%")
    else:
        print("Invalid choice. Defaulting to adversarial training with FMC.")
        args.use_adversarial_training = True
        train_enhanced_model(args)


if __name__ == '__main__':
    main()