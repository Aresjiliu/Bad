import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from tqdm import tqdm
import csv

from models1 import Mdfuse
from fmcnet_modules import FMCEnhancedDrFuse, FMCCompensationLoss
from enhanced_adversarial_generation import (
    EnhancedAdversarialFramework, EnhancedAdversarialLoss,
    EnhancedModalityGenerator, MultiScaleDiscriminator
)
from config import args


class EnhancedAdversarialTrainer:
    """
    Enhanced trainer for adversarial feature generation with FMC compensation
    """

    def __init__(self, args, model, adversarial_framework, device='cuda'):
        self.args = args
        self.device = device
        self.model = model.to(device)
        self.adversarial_framework = adversarial_framework.to(device)

        # Separate optimizers for generator and discriminator
        self.generator_optimizer = optim.Adam(
            self.adversarial_framework.generators.parameters(),
            lr=args.lr * 0.5,  # Lower learning rate for generators
            betas=(0.5, 0.999),
            weight_decay=args.weight_decay
        )

        self.discriminator_optimizer = optim.Adam(
            self.adversarial_framework.discriminators.parameters(),
            lr=args.lr * 0.5,
            betas=(0.5, 0.999),
            weight_decay=args.weight_decay
        )

        # Model optimizer for classification
        self.model_optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # Loss functions
        self.adversarial_loss = EnhancedAdversarialLoss(
            lambda_adv=1.0,
            lambda_rec=0.5,
            lambda_cls=0.3,
            lambda_consistency=0.2,
            lambda_feature=0.1,
            lambda_fmc=0.3
        )

        self.compensation_loss = FMCCompensationLoss(
            lambda_rec=1.0,
            lambda_cls=0.5,
            lambda_adv=0.1,
            lambda_consistency=0.3
        )

        self.classification_loss = nn.CrossEntropyLoss()

        # Training history
        self.training_history = []

    def create_modality_mask(self, batch_size, missing_rate=0.3):
        """
        Create random modality missing masks for training

        Args:
            batch_size: Batch size
            missing_rate: Probability of modality being missing

        Returns:
            modality_mask: [B, 2] tensor indicating available modalities
        """
        modality_mask = torch.rand(batch_size, 2) > missing_rate

        # Ensure at least one modality is available
        missing_all = (modality_mask.sum(dim=1) == 0)
        if missing_all.any():
            # Randomly enable one modality for samples with all missing
            rand_modality = torch.randint(0, 2, (missing_all.sum(),))
            modality_mask[missing_all, rand_modality] = True

        return modality_mask.float().to(self.device)

    def extract_features_from_model(self, model, x1, x2):
        """
        Extract intermediate features from the DrFuse model

        Args:
            model: DrFuse model
            x1: First modality input
            x2: Second modality input

        Returns:
            feature_dict: Dictionary containing extracted features
        """
        # Extract features using SFD module
        if hasattr(model, 'sfd_module'):
            f_sh_1, f_sh_2, f_sp_1, f_sp_2 = model.sfd_module(x1, x2)
        else:
            # Fallback for models without explicit SFD module
            # This would need to be adapted based on actual model architecture
            f_sh_1, f_sh_2, f_sp_1, f_sp_2 = model(x1, x2)

        feature_dict = {
            'hsi_shared': f_sh_1,
            'lidar_shared': f_sh_2,
            'hsi_specific': f_sp_1,
            'lidar_specific': f_sp_2
        }

        return feature_dict

    def train_generator_step(self, available_features, missing_modality_mask, real_features, targets):
        """
        Train generator networks
        """
        self.generator_optimizer.zero_grad()

        # Generate missing features
        generated_features, generation_info = self.adversarial_framework.generate_missing_features(
            available_features, missing_modality_mask
        )

        # Create complete feature set for model input
        complete_features = {}
        for key in ['hsi_shared', 'lidar_shared', 'hsi_specific', 'lidar_specific']:
            if key in available_features:
                complete_features[key] = available_features[key]
            elif key in generated_features:
                complete_features[key] = generated_features[key]
            else:
                # Use zeros for completely missing features
                if 'shared' in key:
                    complete_features[key] = torch.zeros_like(available_features['hsi_shared'] if 'hsi' in key else available_features['lidar_shared'])
                else:
                    complete_features[key] = torch.zeros_like(available_features['hsi_specific'] if 'hsi' in key else available_features['lidar_specific'])

        # Forward pass through model
        # This would need to be adapted based on actual model architecture
        if isinstance(self.model, FMCEnhancedDrFuse):
            predictions, model_features = self.model(
                complete_features['hsi_shared'], complete_features['lidar_shared'],
                modality_mask=missing_modality_mask
            )
        else:
            # Fallback for standard models
            predictions = self.model(
                complete_features['hsi_shared'], complete_features['lidar_shared']
            )

        # Discriminate generated features
        total_disc_loss = 0
        total_gen_loss = 0
        loss_dict = {}

        for modality in ['hsi', 'lidar']:
            if f'{modality}_specific' in generated_features:
                # Get real features for comparison
                real_modality_features = real_features.get(f'{modality}_specific')
                generated_modality_features = generated_features[f'{modality}_specific']

                # Discriminate
                discriminator_output, feature_stats, fused_features = self.adversarial_framework.discriminate_features(
                    generated_modality_features, modality
                )

                # Compute generator loss
                gen_loss, gen_loss_dict = self.adversarial_loss.generator_loss(
                    generated_modality_features,
                    real_modality_features,
                    predictions,
                    targets,
                    discriminator_output,
                    {**generation_info.get(modality, {}), 'fused_features': fused_features}
                )

                total_gen_loss += gen_loss
                loss_dict.update({f'gen_{modality}_{k}': v for k, v in gen_loss_dict.items()})

        # Add classification loss
        cls_loss = self.classification_loss(predictions, targets)
        total_gen_loss += cls_loss
        loss_dict['classification_loss'] = cls_loss

        # Backward pass
        total_gen_loss.backward()
        self.generator_optimizer.step()

        return loss_dict, predictions, complete_features

    def train_discriminator_step(self, real_features, generated_features, missing_modality_mask):
        """
        Train discriminator networks
        """
        self.discriminator_optimizer.zero_grad()

        total_disc_loss = 0
        loss_dict = {}

        for modality in ['hsi', 'lidar']:
            if f'{modality}_specific' in real_features and f'{modality}_specific' in generated_features:
                real_modality_features = real_features[f'{modality}_specific']
                generated_modality_features = generated_features[f'{modality}_specific'].detach()

                # Discriminate real features
                real_validity, real_stats, _ = self.adversarial_framework.discriminate_features(
                    real_modality_features, modality
                )

                # Discriminate generated features
                fake_validity, fake_stats, _ = self.adversarial_framework.discriminate_features(
                    generated_modality_features, modality
                )

                # Compute discriminator loss
                disc_loss, disc_loss_dict = self.adversarial_loss.discriminator_loss(
                    real_validity, fake_validity
                )

                total_disc_loss += disc_loss
                loss_dict.update({f'disc_{modality}_{k}': v for k, v in disc_loss_dict.items()})

        # Backward pass
        total_disc_loss.backward()
        self.discriminator_optimizer.step()

        return loss_dict

    def train_model_step(self, features, targets, missing_modality_mask):
        """
        Train the main model with compensated features
        """
        self.model_optimizer.zero_grad()

        # Forward pass through model
        if isinstance(self.model, FMCEnhancedDrFuse):
            predictions, model_features = self.model(
                features['hsi_shared'], features['lidar_shared'],
                modality_mask=missing_modality_mask
            )
        else:
            predictions = self.model(
                features['hsi_shared'], features['lidar_shared']
            )

        # Classification loss
        cls_loss = self.classification_loss(predictions, targets)

        # FMC compensation loss if applicable
        comp_loss = 0
        if isinstance(self.model, FMCEnhancedDrFuse) and missing_modality_mask is not None:
            comp_loss, comp_loss_dict = self.compensation_loss(
                model_features.get('compensated_features'),
                features,  # Original features
                predictions,
                targets,
                attention_weights=model_features.get('attention_weights')
            )

        # Total loss
        total_loss = cls_loss + comp_loss

        # Backward pass
        total_loss.backward()
        self.model_optimizer.step()

        return {
            'model_classification_loss': cls_loss.item(),
            'model_compensation_loss': comp_loss.item() if isinstance(comp_loss, torch.Tensor) else comp_loss,
            'model_total_loss': total_loss.item()
        }, predictions

    def train_epoch(self, train_loader, epoch):
        """
        Train one epoch
        """
        self.model.train()
        self.adversarial_framework.train()

        epoch_losses = []
        epoch_predictions = []
        epoch_targets = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{self.args.train_epoch}')

        for batch_idx, (data, targets) in enumerate(pbar):
            # Move data to device
            if isinstance(data, (list, tuple)) and len(data) == 2:
                x1, x2 = data[0].to(self.device), data[1].to(self.device)
            else:
                # Handle single tensor input
                x1 = data.to(self.device)
                x2 = torch.zeros_like(x1)  # Placeholder for missing second modality

            targets = targets.to(self.device)
            batch_size = x1.size(0)

            # Create random modality missing masks
            missing_modality_mask = self.create_modality_mask(batch_size)

            # Extract real features from model
            with torch.no_grad():
                real_features = self.extract_features_from_model(self.model, x1, x2)

            # Create available features dictionary
            available_features = {}
            for i, (key, value) in enumerate(real_features.items()):
                if missing_modality_mask[0, i % 2] == 1:  # Modality is available
                    available_features[key] = value

            # Train discriminator (every other batch for stability)
            if batch_idx % 2 == 0:
                # Generate features for discriminator training
                with torch.no_grad():
                    generated_features, _ = self.adversarial_framework.generate_missing_features(
                        available_features, missing_modality_mask
                    )

                disc_losses = self.train_discriminator_step(
                    real_features, generated_features, missing_modality_mask
                )

            # Train generator and model
            gen_losses, predictions, complete_features = self.train_generator_step(
                available_features, missing_modality_mask, real_features, targets
            )

            # Train model separately if needed
            model_losses, _ = self.train_model_step(complete_features, targets, missing_modality_mask)

            # Combine losses
            all_losses = {**gen_losses, **disc_losses, **model_losses}
            epoch_losses.append(all_losses)

            # Store predictions and targets
            epoch_predictions.extend(predictions.cpu().numpy())
            epoch_targets.extend(targets.cpu().numpy())

            # Update progress bar
            avg_losses = {k: np.mean([d[k] for d in epoch_losses if k in d])
                         for k in set().union(*epoch_losses)}

            pbar.set_postfix({
                'Gen Loss': f"{avg_losses.get('total_loss', 0):.4f}",
                'Disc Loss': f"{avg_losses.get('total_disc_loss', 0):.4f}",
                'Model Loss': f"{avg_losses.get('model_total_loss', 0):.4f}"
            })

            # Save intermediate results
            if batch_idx % self.args.save_interval == 0:
                self.save_checkpoint(epoch, batch_idx)

        return epoch_losses, epoch_predictions, epoch_targets

    def validate(self, val_loader, epoch):
        """
        Validate the model
        """
        self.model.eval()
        self.adversarial_framework.eval()

        val_losses = []
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(tqdm(val_loader, desc='Validation')):
                # Similar to training but without gradient updates
                if isinstance(data, (list, tuple)) and len(data) == 2:
                    x1, x2 = data[0].to(self.device), data[1].to(self.device)
                else:
                    x1 = data.to(self.device)
                    x2 = torch.zeros_like(x1)

                targets = targets.to(self.device)

                # Create missing modality masks for validation
                missing_modality_mask = self.create_modality_mask(x1.size(0), missing_rate=0.5)

                # Extract features and generate missing ones
                real_features = self.extract_features_from_model(self.model, x1, x2)
                available_features = {k: v for k, v in real_features.items()}

                generated_features, _ = self.adversarial_framework.generate_missing_features(
                    available_features, missing_modality_mask
                )

                # Create complete feature set
                complete_features = {}
                for key in ['hsi_shared', 'lidar_shared', 'hsi_specific', 'lidar_specific']:
                    if key in available_features:
                        complete_features[key] = available_features[key]
                    elif key in generated_features:
                        complete_features[key] = generated_features[key]
                    else:
                        if 'shared' in key:
                            complete_features[key] = torch.zeros_like(available_features['hsi_shared'] if 'hsi' in key else available_features['lidar_shared'])
                        else:
                            complete_features[key] = torch.zeros_like(available_features['hsi_specific'] if 'hsi' in key else available_features['lidar_specific'])

                # Forward pass
                if isinstance(self.model, FMCEnhancedDrFuse):
                    predictions, _ = self.model(
                        complete_features['hsi_shared'], complete_features['lidar_shared'],
                        modality_mask=missing_modality_mask
                    )
                else:
                    predictions = self.model(
                        complete_features['hsi_shared'], complete_features['lidar_shared']
                    )

                # Calculate loss
                cls_loss = self.classification_loss(predictions, targets)

                val_losses.append({'validation_loss': cls_loss.item()})
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        return val_losses, val_predictions, val_targets

    def save_checkpoint(self, epoch, batch_idx):
        """
        Save training checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state_dict': self.model.state_dict(),
            'adversarial_framework_state_dict': self.adversarial_framework.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'model_optimizer_state_dict': self.model_optimizer.state_dict(),
            'args': self.args
        }

        checkpoint_path = os.path.join(
            self.args.model_root,
            f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pth'
        )
        torch.save(checkpoint, checkpoint_path)

    def save_training_history(self):
        """
        Save training history to CSV
        """
        history_path = os.path.join(self.args.log_root, 'training_history.csv')

        # Flatten training history
        flat_history = []
        for epoch_data in self.training_history:
            for batch_data in epoch_data:
                flat_history.append(batch_data)

        if flat_history:
            # Get all keys
            keys = set().union(*flat_history)

            with open(history_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(keys))
                writer.writeheader()
                writer.writerows(flat_history)

    def train(self, train_loader, val_loader=None):
        """
        Main training loop
        """
        print("Starting enhanced adversarial training with FMC compensation...")

        best_val_accuracy = 0.0

        for epoch in range(1, self.args.train_epoch + 1):
            print(f"\nEpoch {epoch}/{self.args.train_epoch}")

            # Train one epoch
            train_losses, train_predictions, train_targets = self.train_epoch(train_loader, epoch)

            # Validate if validation loader is provided
            if val_loader is not None:
                val_losses, val_predictions, val_targets = self.validate(val_loader, epoch)

                # Calculate validation accuracy
                val_accuracy = np.mean(np.array(val_predictions) == np.array(val_targets))
                print(f"Validation Accuracy: {val_accuracy:.4f}")

                # Save best model
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    self.save_checkpoint(epoch, 0)
                    print(f"New best model saved with accuracy: {val_accuracy:.4f}")

            # Store training history
            self.training_history.append(train_losses)

            # Save training history periodically
            if epoch % 10 == 0:
                self.save_training_history()

        # Final save
        self.save_checkpoint(self.args.train_epoch, 0)
        self.save_training_history()

        print("Training completed!")


def main():
    """
    Main function to run enhanced adversarial training
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    from src.huston2013_dataloader import huston2013_multi_dataloader
    train_loader = huston2013_multi_dataloader(train=True, args=args)
    val_loader = huston2013_multi_dataloader(train=False, args=args)

    # Initialize models
    modality_to_channel = {'hsi': 144, 'hsi1': 244, 'ms': 8, 'lidar': 1, 'sar': 4}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    # Create enhanced DrFuse model with FMC compensation
    if args.use_fmc_enhancement:
        model = FMCEnhancedDrFuse(
            args, modality_1_channel, modality_2_channel,
            feature_dim=128, shared_dim=64, specific_dim=64
        )
    else:
        model = Mdfuse(args, modality_1_channel, modality_2_channel, feature_dim=128)

    # Create adversarial framework
    adversarial_framework = EnhancedAdversarialFramework(
        shared_dim=64, specific_dim=64, feature_dim=128, num_modalities=2
    )

    # Create trainer
    trainer = EnhancedAdversarialTrainer(args, model, adversarial_framework, device)

    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    # Add FMC enhancement flag to args
    args.use_fmc_enhancement = True
    args.compensation_mode = 'adaptive'

    main()