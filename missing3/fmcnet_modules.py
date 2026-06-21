import torch
import torch.nn as nn
import torch.nn.functional as F
from models1 import SFDModuleMD, SpectralSqueeze, SS_DecoupledBlock, OrientedGradientBlock


class FMCModule(nn.Module):
    """
    Feature-level Modality Compensation (FMC) Module
    Based on FMCNet paper for handling missing modality scenarios
    """

    def __init__(self, feature_dim=128, spatial_size=7):
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial_size = spatial_size

        # Spatial attention - identifies important spatial locations (7×7)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 8, 1),
            nn.BatchNorm2d(feature_dim // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 8, 1, 1),
            nn.Sigmoid()
        )

        # Channel attention - identifies important feature channels (128)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, feature_dim // 16, 1),
            nn.BatchNorm2d(feature_dim // 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 16, feature_dim, 1),
            nn.Sigmoid()
        )

        # Cross-modal transformation - maps available modality to missing modality space
        self.cross_modal_transform = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim)
        )

        # Modality-specific refinement
        self.modality_refinement = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, feature_dim, 1),
            nn.BatchNorm2d(feature_dim)
        )

        # Residual scaling parameter
        self.residual_scale = nn.Parameter(torch.ones(1))

        # Feature normalization
        self.feature_norm = nn.BatchNorm2d(feature_dim)

    def forward(self, available_features, target_modality='hsi'):
        """
        Compensate missing modality features using available modality features

        Args:
            available_features: Features from available modality [B, feature_dim, H, W]
            target_modality: Target missing modality ('hsi' or 'lidar')

        Returns:
            compensated_features: Compensated features for missing modality
            attention_weights: Dictionary containing spatial and channel attention weights
        """
        batch_size = available_features.size(0)

        # Generate spatial attention weights [B, 1, H, W]
        spatial_weights = self.spatial_attention(available_features)

        # Generate channel attention weights [B, feature_dim, 1, 1]
        channel_weights = self.channel_attention(available_features)

        # Cross-modal transformation
        transformed_features = self.cross_modal_transform(available_features)

        # Apply attention mechanisms
        attended_features = transformed_features * spatial_weights * channel_weights

        # Modality-specific refinement
        refined_features = self.modality_refinement(attended_features)

        # Residual connection with scaling
        compensated_features = available_features + self.residual_scale * refined_features

        # Final normalization
        compensated_features = self.feature_norm(compensated_features)

        attention_weights = {
            'spatial': spatial_weights,
            'channel': channel_weights,
            'combined': spatial_weights * channel_weights
        }

        return compensated_features, attention_weights


class SharedFeatureFMC(nn.Module):
    """
    FMC module specifically for shared features in DrFuse architecture
    """

    def __init__(self, shared_dim=64, specific_dim=64):
        super().__init__()
        self.shared_dim = shared_dim
        self.specific_dim = specific_dim

        # HSI-specific compensation
        self.hsi_fmc = FMCModule(feature_dim=shared_dim)

        # LiDAR-specific compensation
        self.lidar_fmc = FMCModule(feature_dim=shared_dim)

        # Cross-modality consistency
        self.cross_consistency = nn.Sequential(
            nn.Conv2d(shared_dim * 2, shared_dim, 1),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_dim, shared_dim, 3, padding=1),
            nn.BatchNorm2d(shared_dim)
        )

    def compensate_hsi_shared(self, lidar_shared_features):
        """Compensate HSI shared features using LiDAR shared features"""
        compensated_hsi, attention = self.hsi_fmc(lidar_shared_features, target_modality='hsi')
        return compensated_hsi, attention

    def compensate_lidar_shared(self, hsi_shared_features):
        """Compensate LiDAR shared features using HSI shared features"""
        compensated_lidar, attention = self.lidar_fmc(hsi_shared_features, target_modality='lidar')
        return compensated_lidar, attention

    def cross_modality_consistency(self, hsi_features, lidar_features):
        """Ensure cross-modality consistency between compensated features"""
        combined_features = torch.cat([hsi_features, lidar_features], dim=1)
        consistency_features = self.cross_consistency(combined_features)
        return consistency_features


class SpecificFeatureFMC(nn.Module):
    """
    FMC module specifically for modality-specific features in DrFuse architecture
    """

    def __init__(self, specific_dim=64):
        super().__init__()
        self.specific_dim = specific_dim

        # HSI-specific feature compensation
        self.hsi_specific_fmc = FMCModule(feature_dim=specific_dim)

        # LiDAR-specific feature compensation
        self.lidar_specific_fmc = FMCModule(feature_dim=specific_dim)

        # Feature refinement for specific features
        self.specific_refinement = nn.Sequential(
            SS_DecoupledBlock(specific_dim, specific_dim // 2),
            nn.Conv2d(specific_dim, specific_dim, 3, padding=1),
            nn.BatchNorm2d(specific_dim),
            nn.ReLU(inplace=True)
        )

        # Spectral processing for HSI-specific features
        self.spectral_processing = SpectralSqueeze(specific_dim, ratio=8)

        # Spatial processing for LiDAR-specific features
        self.spatial_processing = OrientedGradientBlock(specific_dim, specific_dim // 2)

    def compensate_hsi_specific(self, lidar_specific_features):
        """Compensate HSI-specific features using LiDAR-specific features"""
        compensated_hsi, attention = self.hsi_specific_fmc(lidar_specific_features, target_modality='hsi')

        # Apply spectral processing for HSI
        compensated_hsi = self.spectral_processing(compensated_hsi)

        # Refine specific features
        compensated_hsi = self.specific_refinement(compensated_hsi)

        return compensated_hsi, attention

    def compensate_lidar_specific(self, hsi_specific_features):
        """Compensate LiDAR-specific features using HSI-specific features"""
        compensated_lidar, attention = self.lidar_specific_fmc(hsi_specific_features, target_modality='lidar')

        # Apply spatial processing for LiDAR
        compensated_lidar = self.spatial_processing(compensated_lidar)

        # Refine specific features
        compensated_lidar = self.specific_refinement(compensated_lidar)

        return compensated_lidar, attention


class FMCEnhancedDrFuse(nn.Module):
    """
    Enhanced DrFuse model with FMC modules for missing modality compensation
    """

    def __init__(self, args, modality_1_channel, modality_2_channel, feature_dim=128,
                 shared_dim=64, specific_dim=64):
        super().__init__()
        self.args = args
        self.feature_dim = feature_dim
        self.shared_dim = shared_dim
        self.specific_dim = specific_dim

        # Original DrFuse components
        self.sfd_module = SFDModuleMD(feature_dim=feature_dim)

        # FMC modules for compensation
        self.shared_fmc = SharedFeatureFMC(shared_dim=shared_dim, specific_dim=specific_dim)
        self.specific_fmc = SpecificFeatureFMC(specific_dim=specific_dim)

        # Fusion and classification components
        self.fusion_classifier = self._build_fusion_classifier()

        # Compensation mode (trainable parameter)
        self.compensation_mode = args.get('compensation_mode', 'adaptive')

    def _build_fusion_classifier(self):
        """Build fusion classifier compatible with compensated features"""
        return nn.Sequential(
            nn.Conv2d(self.shared_dim * 2 + self.specific_dim * 2, self.feature_dim, 1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, self.args.class_num)
        )

    def forward(self, x1, x2, modality_mask=None):
        """
        Forward pass with FMC compensation

        Args:
            x1: First modality input
            x2: Second modality input
            modality_mask: Binary mask indicating available modalities [B, 2]
                          1 = available, 0 = missing
        """
        # Extract features using SFD module
        f_sh_1, f_sh_2, f_sp_1, f_sp_2 = self.sfd_module(x1, x2)

        # Apply FMC compensation if modality_mask is provided
        if modality_mask is not None:
            batch_size = modality_mask.size(0)

            # Process each sample in the batch
            compensated_features = []

            for i in range(batch_size):
                mask = modality_mask[i]  # [2] - [hsi_available, lidar_available]

                # Handle missing HSI (modality 1)
                if mask[0] == 0 and mask[1] == 1:  # HSI missing, LiDAR available
                    comp_f_sh_1, _ = self.shared_fmc.compensate_hsi_shared(f_sh_2[i:i+1])
                    comp_f_sp_1, _ = self.specific_fmc.compensate_hsi_specific(f_sp_2[i:i+1])

                    # Use original features for available modality
                    comp_f_sh_2 = f_sh_2[i:i+1]
                    comp_f_sp_2 = f_sp_2[i:i+1]

                # Handle missing LiDAR (modality 2)
                elif mask[0] == 1 and mask[1] == 0:  # HSI available, LiDAR missing
                    comp_f_sh_2, _ = self.shared_fmc.compensate_lidar_shared(f_sh_1[i:i+1])
                    comp_f_sp_2, _ = self.specific_fmc.compensate_lidar_specific(f_sp_1[i:i+1])

                    # Use original features for available modality
                    comp_f_sh_1 = f_sh_1[i:i+1]
                    comp_f_sp_1 = f_sp_1[i:i+1]

                # Both modalities available - use original features
                else:
                    comp_f_sh_1 = f_sh_1[i:i+1]
                    comp_f_sh_2 = f_sh_2[i:i+1]
                    comp_f_sp_1 = f_sp_1[i:i+1]
                    comp_f_sp_2 = f_sp_2[i:i+1]

                # Combine compensated features
                sample_features = torch.cat([
                    comp_f_sh_1, comp_f_sh_2, comp_f_sp_1, comp_f_sp_2
                ], dim=1)
                compensated_features.append(sample_features)

            # Concatenate all compensated features
            fusion_input = torch.cat(compensated_features, dim=0)

        else:
            # No compensation - use original features
            fusion_input = torch.cat([f_sh_1, f_sh_2, f_sp_1, f_sp_2], dim=1)

        # Fusion and classification
        output = self.fusion_classifier(fusion_input)

        return output, {
            'f_sh_1': f_sh_1, 'f_sh_2': f_sh_2,
            'f_sp_1': f_sp_1, 'f_sp_2': f_sp_2,
            'fusion_input': fusion_input
        }


class FMCCompensationLoss(nn.Module):
    """
    Loss functions for FMC compensation training
    """

    def __init__(self, lambda_rec=1.0, lambda_cls=0.5, lambda_adv=0.1, lambda_consistency=0.3):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_cls = lambda_cls
        self.lambda_adv = lambda_adv
        self.lambda_consistency = lambda_consistency

        self.reconstruction_loss = nn.MSELoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.consistency_loss = nn.SmoothL1Loss()

    def forward(self, compensated_features, original_features, predictions, targets,
                attention_weights=None, consistency_features=None):
        """
        Calculate total FMC compensation loss

        Args:
            compensated_features: Features after FMC compensation
            original_features: Original complete features (for reconstruction loss)
            predictions: Model predictions
            targets: Ground truth labels
            attention_weights: Attention weights from FMC modules
            consistency_features: Cross-modality consistency features
        """
        # Reconstruction loss - ensure compensated features match original features
        rec_loss = self.reconstruction_loss(compensated_features, original_features)

        # Classification loss - ensure compensated features work well for classification
        cls_loss = self.classification_loss(predictions, targets)

        # Consistency loss - ensure cross-modality consistency
        consistency_loss = 0.0
        if consistency_features is not None:
            # Encourage consistency between compensated features
            consistency_loss = self.consistency_loss(
                consistency_features,
                torch.zeros_like(consistency_features)
            )

        # Attention regularization - encourage diverse attention patterns
        attention_reg = 0.0
        if attention_weights is not None:
            # Prevent attention collapse
            spatial_attn_mean = attention_weights['spatial'].mean()
            channel_attn_mean = attention_weights['channel'].mean()
            attention_reg = torch.abs(spatial_attn_mean - 0.5) + torch.abs(channel_attn_mean - 0.5)

        # Total loss
        total_loss = (
            self.lambda_rec * rec_loss +
            self.lambda_cls * cls_loss +
            self.lambda_consistency * consistency_loss +
            0.01 * attention_reg  # Small attention regularization
        )

        loss_dict = {
            'total_loss': total_loss,
            'reconstruction_loss': rec_loss,
            'classification_loss': cls_loss,
            'consistency_loss': consistency_loss,
            'attention_regularization': attention_reg
        }

        return total_loss, loss_dict