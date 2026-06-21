import torch
import torch.nn as nn
import torch.nn.functional as F
from models1 import SFDModuleMD, SpectralSqueeze, SS_DecoupledBlock, OrientedGradientBlock
from fmcnet_modules import FMCModule, SharedFeatureFMC, SpecificFeatureFMC


class EnhancedModalityGenerator(nn.Module):
    """
    Enhanced modality generator combining adversarial learning with FMC compensation
    """

    def __init__(self, shared_dim=64, specific_dim=64, feature_dim=128, target_modality='hsi'):
        super().__init__()
        self.shared_dim = shared_dim
        self.specific_dim = specific_dim
        self.feature_dim = feature_dim
        self.target_modality = target_modality

        # FMC-based feature compensation
        self.fmc_module = FMCModule(feature_dim=shared_dim)

        # Multi-scale feature extraction
        self.multi_scale_extractor = nn.ModuleDict({
            'scale1': nn.Conv2d(shared_dim, shared_dim // 2, 3, padding=1),
            'scale2': nn.Conv2d(shared_dim, shared_dim // 2, 3, padding=2, dilation=2),
            'scale3': nn.Conv2d(shared_dim, shared_dim // 2, 3, padding=3, dilation=3)
        })

        # Feature fusion
        self.scale_fusion = nn.Sequential(
            nn.Conv2d(shared_dim * 3 // 2, shared_dim, 1),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU(inplace=True)
        )

        # Modality-specific generation network
        if target_modality == 'hsi':
            self.modality_generator = self._build_hsi_generator()
        else:  # lidar
            self.modality_generator = self._build_lidar_generator()

        # Feature refinement
        self.feature_refinement = nn.Sequential(
            nn.Conv2d(specific_dim, specific_dim, 3, padding=1),
            nn.BatchNorm2d(specific_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(specific_dim, specific_dim, 3, padding=1),
            nn.BatchNorm2d(specific_dim)
        )

        # Attention mechanism
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(specific_dim, specific_dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(specific_dim // 8, 1, 1),
            nn.Sigmoid()
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(specific_dim, specific_dim // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(specific_dim // 16, specific_dim, 1),
            nn.Sigmoid()
        )

        # Residual connection
        self.residual_scale = nn.Parameter(torch.ones(1))

    def _build_hsi_generator(self):
        """Build HSI-specific generator with spectral attention"""
        return nn.Sequential(
            # Spectral-spatial feature generation
            nn.Conv2d(self.shared_dim, self.specific_dim, 3, padding=1),
            nn.BatchNorm2d(self.specific_dim),
            nn.ReLU(inplace=True),

            # Multi-scale spectral processing
            SpectralSqueeze(self.specific_dim, ratio=8),

            # Dilated convolutions for multi-scale spatial features
            nn.Conv2d(self.specific_dim, self.specific_dim, 3, padding=1, dilation=1),
            nn.BatchNorm2d(self.specific_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.specific_dim, self.specific_dim, 3, padding=2, dilation=2),
            nn.BatchNorm2d(self.specific_dim),
            nn.ReLU(inplace=True),

            # Final generation layer
            nn.Conv2d(self.specific_dim, self.specific_dim, 3, padding=1),
            nn.Tanh()
        )

    def _build_lidar_generator(self):
        """Build LiDAR-specific generator with spatial processing"""
        return nn.Sequential(
            # Spatial feature generation
            nn.Conv2d(self.shared_dim, self.specific_dim, 3, padding=1),
            nn.BatchNorm2d(self.specific_dim),
            nn.ReLU(inplace=True),

            # Oriented gradient processing
            OrientedGradientBlock(self.specific_dim, self.specific_dim // 2),

            # Spatial attention and refinement
            nn.Conv2d(self.specific_dim, self.specific_dim, 3, padding=1),
            nn.BatchNorm2d(self.specific_dim),
            nn.ReLU(inplace=True),

            # Multi-scale spatial processing
            nn.Conv2d(self.specific_dim, self.specific_dim, 3, padding=2, dilation=2),
            nn.BatchNorm2d(self.specific_dim),
            nn.ReLU(inplace=True),

            # Final generation layer
            nn.Conv2d(self.specific_dim, self.specific_dim, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, shared_features, available_modality_features=None, modality_mask=None):
        """
        Generate missing modality features using enhanced adversarial approach

        Args:
            shared_features: Available modality shared features [B, shared_dim, H, W]
            available_modality_features: Additional features from available modality
            modality_mask: Modality availability mask

        Returns:
            generated_specific: Generated specific features
            generation_info: Dictionary with generation details
        """
        # Apply FMC compensation first
        if available_modality_features is not None:
            compensated_features, fmc_attention = self.fmc_module(
                available_modality_features, target_modality=self.target_modality
            )
            # Combine shared features with compensated features
            combined_input = shared_features + 0.5 * compensated_features
        else:
            combined_input = shared_features
            fmc_attention = None

        # Multi-scale feature extraction
        scale_features = []
        for scale_name, extractor in self.multi_scale_extractor.items():
            scale_feat = extractor(combined_input)
            scale_features.append(scale_feat)

        # Concatenate multi-scale features
        multi_scale_input = torch.cat(scale_features, dim=1)

        # Fuse multi-scale features
        fused_features = self.scale_fusion(multi_scale_input)

        # Generate modality-specific features
        generated_features = self.modality_generator(fused_features)

        # Apply attention mechanisms
        spatial_attn = self.spatial_attention(generated_features)
        channel_attn = self.channel_attention(generated_features)
        attended_features = generated_features * spatial_attn * channel_attn

        # Feature refinement
        refined_features = self.feature_refinement(attended_features)

        # Residual connection
        residual_features = self.residual_scale * refined_features
        final_features = generated_features + residual_features

        generation_info = {
            'fmc_attention': fmc_attention,
            'spatial_attention': spatial_attn,
            'channel_attention': channel_attn,
            'multi_scale_features': scale_features,
            'generated_features': generated_features,
            'refined_features': refined_features,
            'final_features': final_features
        }

        return final_features, generation_info


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for adversarial training
    """

    def __init__(self, input_dim=64, feature_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim

        # Multi-scale feature extraction
        self.scale_extractors = nn.ModuleDict({
            'original': self._build_scale_extractor(input_dim, feature_dim),
            'downsampled_2': self._build_scale_extractor(input_dim, feature_dim),
            'downsampled_4': self._build_scale_extractor(input_dim, feature_dim)
        })

        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 3, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

        # Discriminator head
        self.discriminator = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_dim // 2, feature_dim // 4, 3, padding=1),
            nn.BatchNorm2d(feature_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim // 4, 1)
        )

        # Feature statistics network
        self.statistics_network = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, 1)
        )

    def _build_scale_extractor(self, input_dim, feature_dim):
        """Build scale-specific feature extractor"""
        return nn.Sequential(
            nn.Conv2d(input_dim, feature_dim // 2, 3, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_dim // 2, feature_dim // 2, 3, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_dim // 2, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, features, scale='original'):
        """
        Discriminate between real and generated features

        Args:
            features: Input features [B, input_dim, H, W]
            scale: Scale level for discrimination

        Returns:
            validity: Discriminator output
            feature_stats: Feature statistics
        """
        # Multi-scale feature extraction
        scale_features = []
        for scale_name, extractor in self.scale_extractors.items():
            if scale_name == 'original':
                input_features = features
            elif scale_name == 'downsampled_2':
                input_features = F.avg_pool2d(features, kernel_size=2, stride=2)
            else:  # downsampled_4
                input_features = F.avg_pool2d(features, kernel_size=4, stride=4)

            scale_feat = extractor(input_features)
            # Upsample to original size for fusion
            if scale_name != 'original':
                target_size = features.size()[2:]
                scale_feat = F.interpolate(scale_feat, size=target_size, mode='bilinear', align_corners=False)
            scale_features.append(scale_feat)

        # Fuse multi-scale features
        fused_features = torch.cat(scale_features, dim=1)
        fused_features = self.feature_fusion(fused_features)

        # Discriminate
        validity = self.discriminator(fused_features)

        # Extract feature statistics
        feature_stats = self.statistics_network(features)

        return validity, feature_stats, fused_features


class EnhancedAdversarialFramework(nn.Module):
    """
    Enhanced adversarial framework combining FMC compensation with adversarial generation
    """

    def __init__(self, shared_dim=64, specific_dim=64, feature_dim=128, num_modalities=2):
        super().__init__()
        self.shared_dim = shared_dim
        self.specific_dim = specific_dim
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities

        # Enhanced generators for each modality
        self.generators = nn.ModuleDict({
            'hsi': EnhancedModalityGenerator(shared_dim, specific_dim, feature_dim, 'hsi'),
            'lidar': EnhancedModalityGenerator(shared_dim, specific_dim, feature_dim, 'lidar')
        })

        # Multi-scale discriminators
        self.discriminators = nn.ModuleDict({
            'hsi': MultiScaleDiscriminator(specific_dim, feature_dim),
            'lidar': MultiScaleDiscriminator(specific_dim, feature_dim)
        })

        # FMC modules for feature compensation
        self.shared_fmc = SharedFeatureFMC(shared_dim=shared_dim, specific_dim=specific_dim)
        self.specific_fmc = SpecificFeatureFMC(specific_dim=specific_dim)

        # Feature consistency network
        self.consistency_network = nn.Sequential(
            nn.Conv2d(specific_dim * 2, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, specific_dim, 1),
            nn.BatchNorm2d(specific_dim)
        )

        # Modality alignment network
        self.modality_alignment = nn.Sequential(
            nn.Conv2d(specific_dim, specific_dim, 3, padding=1),
            nn.BatchNorm2d(specific_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(specific_dim, specific_dim, 3, padding=1),
            nn.BatchNorm2d(specific_dim)
        )

    def generate_missing_features(self, available_features_dict, modality_mask):
        """
        Generate features for missing modalities using adversarial approach

        Args:
            available_features_dict: Dictionary of available features
            modality_mask: Modality availability mask

        Returns:
            generated_features: Dictionary of generated features
            generation_info: Generation details
        """
        generated_features = {}
        generation_info = {}

        # Generate HSI features if missing
        if modality_mask[0] == 0 and 'lidar_shared' in available_features_dict:
            hsi_features, hsi_info = self.generators['hsi'](
                available_features_dict['lidar_shared'],
                available_features_dict.get('lidar_specific'),
                modality_mask
            )
            generated_features['hsi_specific'] = hsi_features
            generation_info['hsi'] = hsi_info

        # Generate LiDAR features if missing
        if modality_mask[1] == 0 and 'hsi_shared' in available_features_dict:
            lidar_features, lidar_info = self.generators['lidar'](
                available_features_dict['hsi_shared'],
                available_features_dict.get('hsi_specific'),
                modality_mask
            )
            generated_features['lidar_specific'] = lidar_features
            generation_info['lidar'] = lidar_info

        return generated_features, generation_info

    def discriminate_features(self, features, modality, scale='original'):
        """
        Discriminate between real and generated features

        Args:
            features: Features to discriminate
            modality: Target modality ('hsi' or 'lidar')
            scale: Scale level

        Returns:
            validity: Discriminator output
            feature_stats: Feature statistics
        """
        validity, feature_stats, fused_features = self.discriminators[modality](features, scale)
        return validity, feature_stats, fused_features

    def compute_consistency_loss(self, real_features, generated_features):
        """
        Compute consistency loss between real and generated features
        """
        if real_features is None or generated_features is None:
            return torch.tensor(0.0, device=generated_features.device if generated_features is not None else real_features.device)

        # Concatenate features
        combined_features = torch.cat([real_features, generated_features], dim=1)

        # Compute consistency features
        consistency_features = self.consistency_network(combined_features)

        # Alignment loss
        aligned_real = self.modality_alignment(real_features)
        aligned_generated = self.modality_alignment(generated_features)

        consistency_loss = F.mse_loss(aligned_real, aligned_generated) + \
                          F.mse_loss(consistency_features, torch.zeros_like(consistency_features))

        return consistency_loss


class EnhancedAdversarialLoss(nn.Module):
    """
    Enhanced loss functions for adversarial training with FMC integration
    """

    def __init__(self, lambda_adv=1.0, lambda_rec=0.5, lambda_cls=0.3, lambda_consistency=0.2,
                 lambda_feature=0.1, lambda_fmc=0.3):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_rec = lambda_rec
        self.lambda_cls = lambda_cls
        self.lambda_consistency = lambda_consistency
        self.lambda_feature = lambda_feature
        self.lambda_fmc = lambda_fmc

        # Loss components
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.MSELoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.feature_matching_loss = nn.L1Loss()

    def generator_loss(self, generated_features, real_features, predictions, targets,
                      discriminator_output, generation_info):
        """
        Compute generator loss
        """
        # Adversarial loss (generator wants to fool discriminator)
        valid = torch.ones_like(discriminator_output)
        adv_loss = self.adversarial_loss(discriminator_output, valid)

        # Reconstruction loss
        rec_loss = self.reconstruction_loss(generated_features, real_features)

        # Classification loss
        cls_loss = self.classification_loss(predictions, targets)

        # Feature matching loss
        if 'fused_features' in generation_info:
            feature_loss = self.feature_matching_loss(
                generation_info['fused_features'],
                torch.ones_like(generation_info['fused_features']) * 0.5
            )
        else:
            feature_loss = torch.tensor(0.0, device=generated_features.device)

        # FMC attention regularization
        fmc_reg = 0.0
        if generation_info.get('fmc_attention') is not None:
            fmc_weights = generation_info['fmc_attention']
            if 'spatial' in fmc_weights:
                fmc_reg += torch.mean(torch.abs(fmc_weights['spatial'] - 0.5))
            if 'channel' in fmc_weights:
                fmc_reg += torch.mean(torch.abs(fmc_weights['channel'] - 0.5))

        # Total generator loss
        total_loss = (
            self.lambda_adv * adv_loss +
            self.lambda_rec * rec_loss +
            self.lambda_cls * cls_loss +
            self.lambda_feature * feature_loss +
            self.lambda_fmc * fmc_reg
        )

        loss_dict = {
            'total_loss': total_loss,
            'adversarial_loss': adv_loss,
            'reconstruction_loss': rec_loss,
            'classification_loss': cls_loss,
            'feature_matching_loss': feature_loss,
            'fmc_regularization': fmc_reg
        }

        return total_loss, loss_dict

    def discriminator_loss(self, real_validity, fake_validity):
        """
        Compute discriminator loss
        """
        # Real samples should be classified as real
        real_loss = self.adversarial_loss(real_validity, torch.ones_like(real_validity))

        # Fake samples should be classified as fake
        fake_loss = self.adversarial_loss(fake_validity, torch.zeros_like(fake_validity))

        # Total discriminator loss
        total_loss = real_loss + fake_loss

        loss_dict = {
            'total_loss': total_loss,
            'real_loss': real_loss,
            'fake_loss': fake_loss
        }

        return total_loss, loss_dict