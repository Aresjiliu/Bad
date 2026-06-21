import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np


class AdversarialLoss(nn.Module):
    """
    对抗损失函数基类
    """

    def __init__(self, loss_type='wgan-gp'):
        super().__init__()
        self.loss_type = loss_type

    def generator_loss(self, fake_validity):
        """
        生成器损失

        Args:
            fake_validity: 判别器对生成样本的输出

        Returns:
            gen_loss: 生成器损失
        """
        if self.loss_type == 'gan':
            # 标准GAN损失
            return F.binary_cross_entropy_with_logits(fake_validity, torch.ones_like(fake_validity))
        elif self.loss_type == 'lsgan':
            # LSGAN损失
            return F.mse_loss(fake_validity, torch.ones_like(fake_validity))
        elif self.loss_type == 'wgan':
            # WGAN损失
            return -fake_validity.mean()
        elif self.loss_type == 'wgan-gp':
            # WGAN-GP损失
            return -fake_validity.mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def discriminator_loss(self, real_validity, fake_validity):
        """
        判别器损失

        Args:
            real_validity: 判别器对真实样本的输出
            fake_validity: 判别器对生成样本的输出

        Returns:
            dis_loss: 判别器损失
        """
        if self.loss_type == 'gan':
            # 标准GAN损失
            real_loss = F.binary_cross_entropy_with_logits(real_validity, torch.ones_like(real_validity))
            fake_loss = F.binary_cross_entropy_with_logits(fake_validity, torch.zeros_like(fake_validity))
            return real_loss + fake_loss
        elif self.loss_type == 'lsgan':
            # LSGAN损失
            real_loss = F.mse_loss(real_validity, torch.ones_like(real_validity))
            fake_loss = F.mse_loss(fake_validity, torch.zeros_like(fake_validity))
            return real_loss + fake_loss
        elif self.loss_type == 'wgan':
            # WGAN损失
            return fake_validity.mean() - real_validity.mean()
        elif self.loss_type == 'wgan-gp':
            # WGAN-GP损失（不包含梯度惩罚，需要单独计算）
            return fake_validity.mean() - real_validity.mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class FeatureConsistencyLoss(nn.Module):
    """
    特征一致性损失
    确保生成的特征与真实特征在语义上保持一致
    """

    def __init__(self, consistency_type='mse', lambda_consistency=1.0):
        super().__init__()
        self.consistency_type = consistency_type
        self.lambda_consistency = lambda_consistency

        if consistency_type == 'mse':
            self.criterion = nn.MSELoss()
        elif consistency_type == 'l1':
            self.criterion = nn.L1Loss()
        elif consistency_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        elif consistency_type == 'cosine':
            self.criterion = nn.CosineEmbeddingLoss()
        else:
            raise ValueError(f"Unknown consistency type: {consistency_type}")

    def forward(self, generated_features, target_features=None, shared_features=None):
        """
        计算特征一致性损失

        Args:
            generated_features: 生成的特征
            target_features: 目标特征（如果有的话）
            shared_features: 共享特征（用于一致性约束）

        Returns:
            consistency_loss: 一致性损失
        """
        if target_features is not None:
            # 有目标特征，直接比较
            if self.consistency_type == 'cosine':
                # Cosine相似度损失
                batch_size = generated_features.size(0)
                target = torch.ones(batch_size, device=generated_features.device)
                generated_flat = generated_features.view(batch_size, -1)
                target_flat = target_features.view(batch_size, -1)
                return self.lambda_consistency * self.criterion(generated_flat, target_flat, target)
            else:
                return self.lambda_consistency * self.criterion(generated_features, target_features)
        elif shared_features is not None:
            # 基于共享特征的一致性约束
            # 生成的特征应该与共享特征保持一定的语义一致性
            generated_mean = generated_features.mean(dim=[2, 3], keepdim=True)
            shared_mean = shared_features.mean(dim=[2, 3], keepdim=True)
            return self.lambda_consistency * self.criterion(generated_mean, shared_mean)
        else:
            # 无约束情况，返回零损失
            return torch.tensor(0.0, device=generated_features.device, requires_grad=True)


class ModalityAlignmentLoss(nn.Module):
    """
    模态对齐损失
    确保生成的特征与对应模态的真实特征在分布上对齐
    """

    def __init__(self, alignment_type='mmd', lambda_alignment=1.0):
        super().__init__()
        self.alignment_type = alignment_type
        self.lambda_alignment = lambda_alignment

    def mmd_loss(self, source_features, target_features, kernel_type='rbf'):
        """
        最大均值差异（MMD）损失

        Args:
            source_features: 源特征
            target_features: 目标特征
            kernel_type: 核函数类型

        Returns:
            mmd_loss: MMD损失
        """
        batch_size = source_features.size(0)
        device = source_features.device

        # 展平特征
        source_flat = source_features.view(batch_size, -1)
        target_flat = target_features.view(batch_size, -1)

        if kernel_type == 'linear':
            # 线性核
            source_mean = source_flat.mean(dim=0)
            target_mean = target_flat.mean(dim=0)
            return torch.norm(source_mean - target_mean, p=2)
        elif kernel_type == 'rbf':
            # RBF核
            # 计算核矩阵
            def rbf_kernel(x, y):
                xx = torch.mm(x, x.t())
                yy = torch.mm(y, y.t())
                xy = torch.mm(x, y.t())

                rx = xx.diag().unsqueeze(0).expand_as(xx)
                ry = yy.diag().unsqueeze(0).expand_as(yy)

                dxx = rx.t() + rx - 2 * xx
                dyy = ry.t() + ry - 2 * yy
                dxy = rx.t() + ry - 2 * xy

                bandwidth = 0.5 * (dxx.mean() + dyy.mean())
                k_xx = torch.exp(-dxx / (2 * bandwidth))
                k_yy = torch.exp(-dyy / (2 * bandwidth))
                k_xy = torch.exp(-dxy / (2 * bandwidth))

                return k_xx, k_yy, k_xy

            k_xx, k_yy, k_xy = rbf_kernel(source_flat, target_flat)

            # MMD统计量
            mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
            return mmd
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    def coral_loss(self, source_features, target_features):
        """
        CORAL（Correlation Alignment）损失

        Args:
            source_features: 源特征
            target_features: 目标特征

        Returns:
            coral_loss: CORAL损失
        """
        batch_size = source_features.size(0)
        device = source_features.device

        # 展平特征
        source_flat = source_features.view(batch_size, -1)
        target_flat = target_features.view(batch_size, -1)

        # 计算协方差矩阵
        def compute_covariance(features):
            mean = features.mean(dim=0, keepdim=True)
            centered = features - mean
            cov = torch.mm(centered.t(), centered) / (batch_size - 1)
            return cov

        source_cov = compute_covariance(source_flat)
        target_cov = compute_covariance(target_flat)

        # CORAL损失
        coral_loss = torch.norm(source_cov - target_cov, p='fro') ** 2
        return coral_loss

    def forward(self, generated_features, real_features, shared_features=None):
        """
        计算模态对齐损失

        Args:
            generated_features: 生成的特征
            real_features: 真实特征
            shared_features: 共享特征（可选）

        Returns:
            alignment_loss: 对齐损失
        """
        if self.alignment_type == 'mmd':
            return self.lambda_alignment * self.mmd_loss(generated_features, real_features)
        elif self.alignment_type == 'coral':
            return self.lambda_alignment * self.coral_loss(generated_features, real_features)
        elif self.alignment_type == 'mse':
            return self.lambda_alignment * F.mse_loss(generated_features, real_features)
        else:
            raise ValueError(f"Unknown alignment type: {self.alignment_type}")


class AdversarialFeatureLoss(nn.Module):
    """
    对抗特征生成损失函数
    整合所有对抗学习相关的损失
    """

    def __init__(self, adversarial_type='wgan-gp', consistency_type='mse', alignment_type='mmd',
                 lambda_adv=1.0, lambda_consistency=1.0, lambda_alignment=1.0, lambda_gp=10.0):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_consistency = lambda_consistency
        self.lambda_alignment = lambda_alignment
        self.lambda_gp = lambda_gp

        # 对抗损失
        self.adversarial_loss = AdversarialLoss(adversarial_type)

        # 特征一致性损失
        self.consistency_loss = FeatureConsistencyLoss(consistency_type, lambda_consistency)

        # 模态对齐损失
        self.alignment_loss = ModalityAlignmentLoss(alignment_type, lambda_alignment)

    def generator_loss(self, generated_features_dict, real_features_dict, shared_features_dict,
                       discriminator_validity_dict, modality_mask):
        """
        生成器的总损失

        Args:
            generated_features_dict: 生成的特征字典
            real_features_dict: 真实特征字典
            shared_features_dict: 共享特征字典
            discriminator_validity_dict: 判别器输出字典
            modality_mask: 模态存在性标记

        Returns:
            total_gen_loss: 生成器总损失
            loss_dict: 各个损失分量的字典
        """
        total_gen_loss = 0
        loss_dict = {}

        # 1. 对抗损失（生成器部分）
        adv_gen_loss = 0
        for modality, validity in discriminator_validity_dict.items():
            if 'generated' in modality or 'fake' in modality:
                adv_gen_loss += self.adversarial_loss.generator_loss(validity)

        total_gen_loss += self.lambda_adv * adv_gen_loss
        loss_dict['adv_gen'] = adv_gen_loss.item() if isinstance(adv_gen_loss, torch.Tensor) else adv_gen_loss

        # 2. 特征一致性损失
        consistency_loss = 0
        for modality, generated_feat in generated_features_dict.items():
            if modality in real_features_dict and real_features_dict[modality] is not None:
                # 有对应的真实特征，直接比较
                consistency_loss += self.consistency_loss(
                    generated_feat, real_features_dict[modality]
                )
            elif modality in shared_features_dict and shared_features_dict[modality] is not None:
                # 基于共享特征的一致性约束
                consistency_loss += self.consistency_loss(
                    generated_feat, shared_features=shared_features_dict[modality]
                )

        total_gen_loss += consistency_loss
        loss_dict['consistency'] = consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss

        # 3. 模态对齐损失
        alignment_loss = 0
        for modality, generated_feat in generated_features_dict.items():
            if modality in real_features_dict and real_features_dict[modality] is not None:
                alignment_loss += self.alignment_loss(
                    generated_feat, real_features_dict[modality]
                )

        total_gen_loss += alignment_loss
        loss_dict['alignment'] = alignment_loss.item() if isinstance(alignment_loss, torch.Tensor) else alignment_loss

        loss_dict['total_gen'] = total_gen_loss.item() if isinstance(total_gen_loss, torch.Tensor) else total_gen_loss

        return total_gen_loss, loss_dict

    def discriminator_loss(self, real_validity_dict, fake_validity_dict, gradient_penalties=None):
        """
        判别器的总损失

        Args:
            real_validity_dict: 真实样本的判别器输出
            fake_validity_dict: 生成样本的判别器输出
            gradient_penalties: 梯度惩罚字典（可选）

        Returns:
            total_dis_loss: 判别器总损失
            loss_dict: 各个损失分量的字典
        """
        total_dis_loss = 0
        loss_dict = {}

        # 1. 对抗损失（判别器部分）
        real_validity_list = []
        fake_validity_list = []

        for modality, validity in real_validity_dict.items():
            if 'real' in modality or modality in ['hsi', 'lidar']:
                real_validity_list.append(validity)

        for modality, validity in fake_validity_dict.items():
            if 'generated' in modality or 'fake' in modality:
                fake_validity_list.append(validity)

        if real_validity_list and fake_validity_list:
            real_validity = torch.cat(real_validity_list, dim=0)
            fake_validity = torch.cat(fake_validity_list, dim=0)
            adv_dis_loss = self.adversarial_loss.discriminator_loss(real_validity, fake_validity)
        else:
            adv_dis_loss = torch.tensor(0.0, device=list(real_validity_dict.values())[0].device)

        total_dis_loss += adv_dis_loss
        loss_dict['adv_dis'] = adv_dis_loss.item() if isinstance(adv_dis_loss, torch.Tensor) else adv_dis_loss

        # 2. 梯度惩罚（如果使用WGAN-GP）
        if gradient_penalties is not None and self.lambda_gp > 0:
            gp_loss = 0
            for modality, gp in gradient_penalties.items():
                gp_loss += gp
            total_gp_loss = self.lambda_gp * gp_loss
            total_dis_loss += total_gp_loss
            loss_dict['gradient_penalty'] = total_gp_loss.item() if isinstance(total_gp_loss, torch.Tensor) else total_gp_loss

        loss_dict['total_dis'] = total_dis_loss.item() if isinstance(total_dis_loss, torch.Tensor) else total_dis_loss

        return total_dis_loss, loss_dict


class FeatureReconstructionLoss(nn.Module):
    """
    特征重构损失
    确保生成的特征可以被重构回原始模态
    """

    def __init__(self, lambda_recon=1.0):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.mse_loss = nn.MSELoss()

    def forward(self, reconstructed_features, original_features, mask=None):
        """
        计算重构损失

        Args:
            reconstructed_features: 重构的特征
            original_features: 原始特征
            mask: 掩码（可选）

        Returns:
            recon_loss: 重构损失
        """
        if mask is not None:
            # 应用掩码
            recon_loss = self.mse_loss(reconstructed_features * mask, original_features * mask)
            # 归一化
            recon_loss = recon_loss / (mask.sum() + 1e-8)
        else:
            recon_loss = self.mse_loss(reconstructed_features, original_features)

        return self.lambda_recon * recon_loss


class CrossModalConsistencyLoss(nn.Module):
    """
    跨模态一致性损失
    确保生成的特征与对应模态的共享特征保持一致性
    """

    def __init__(self, lambda_cross=1.0, temperature=0.1):
        super().__init__()
        self.lambda_cross = lambda_cross
        self.temperature = temperature

    def forward(self, generated_specific, shared_features, labels=None):
        """
        计算跨模态一致性损失

        Args:
            generated_specific: 生成的特定特征
            shared_features: 共享特征
            labels: 标签（可选，用于监督一致性）

        Returns:
            cross_consistency_loss: 跨模态一致性损失
        """
        B, C, H, W = generated_specific.shape

        # 全局平均池化
        generated_global = F.adaptive_avg_pool2d(generated_specific, 1).view(B, C)
        shared_global = F.adaptive_avg_pool2d(shared_features, 1).view(B, C)

        # 归一化
        generated_norm = F.normalize(generated_global, p=2, dim=1)
        shared_norm = F.normalize(shared_global, p=2, dim=1)

        # 计算相似度
        similarity = torch.mm(generated_norm, shared_norm.t()) / self.temperature

        if labels is not None:
            # 有监督一致性：同类样本应该更相似
            mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
            positive_pairs = similarity * mask
            negative_pairs = similarity * (1 - mask)

            # InfoNCE损失
            numerator = torch.exp(positive_pairs).sum(dim=1)
            denominator = torch.exp(similarity).sum(dim=1)
            consistency_loss = -torch.log(numerator / (denominator + 1e-8) + 1e-8)
            return self.lambda_cross * consistency_loss.mean()
        else:
            # 无监督一致性：最大化生成特征和共享特征的相似度
            positive_pairs = torch.diagonal(similarity, 0)
            consistency_loss = -torch.log(torch.sigmoid(positive_pairs) + 1e-8)
            return self.lambda_cross * consistency_loss.mean()


# 测试函数
def test_adversarial_losses():
    """测试对抗损失函数"""
    print("测试对抗损失函数...")

    # 测试数据
    B = 4
    C = 64
    H = W = 7
    device = 'cpu'

    # 创建损失函数
    adversarial_loss = AdversarialLoss('wgan-gp')
    consistency_loss = FeatureConsistencyLoss('mse', 1.0)
    alignment_loss = ModalityAlignmentLoss('mmd', 1.0)
    full_loss = AdversarialFeatureLoss(
        adversarial_type='wgan-gp',
        consistency_type='mse',
        alignment_type='mmd',
        lambda_adv=1.0,
        lambda_consistency=1.0,
        lambda_alignment=1.0,
        lambda_gp=10.0
    )

    # 测试数据
    generated_features = torch.randn(B, C, H, W, device=device)
    real_features = torch.randn(B, C, H, W, device=device)
    shared_features = torch.randn(B, C, H, W, device=device)
    fake_validity = torch.randn(B, 1, device=device)
    real_validity = torch.randn(B, 1, device=device)

    # 测试对抗损失
    gen_loss = adversarial_loss.generator_loss(fake_validity)
    dis_loss = adversarial_loss.discriminator_loss(real_validity, fake_validity)
    print(f"生成器损失: {gen_loss.item():.4f}")
    print(f"判别器损失: {dis_loss.item():.4f}")

    # 测试特征一致性损失
    consistency = consistency_loss(generated_features, real_features)
    print(f"特征一致性损失: {consistency.item():.4f}")

    # 测试模态对齐损失
    alignment = alignment_loss(generated_features, real_features)
    print(f"模态对齐损失: {alignment.item():.4f}")

    # 测试完整的对抗特征损失
    generated_dict = {'hsi': generated_features}
    real_dict = {'hsi': real_features}
    shared_dict = {'hsi': shared_features}
    fake_validity_dict = {'hsi_generated': fake_validity}
    real_validity_dict = {'hsi_real': real_validity}

    total_gen_loss, gen_loss_dict = full_loss.generator_loss(
        generated_dict, real_dict, shared_dict, fake_validity_dict, None
    )

    total_dis_loss, dis_loss_dict = full_loss.discriminator_loss(
        real_validity_dict, fake_validity_dict
    )

    print(f"生成器总损失: {total_gen_loss.item():.4f}")
    print(f"判别器总损失: {total_dis_loss.item():.4f}")
    print(f"生成器损失分量: {gen_loss_dict}")
    print(f"判别器损失分量: {dis_loss_dict}")

    print("对抗损失函数测试通过!")


if __name__ == "__main__":
    test_adversarial_losses()