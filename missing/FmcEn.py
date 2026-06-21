import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim, 256)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(128, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, F_gen, F_real, labels):
        pos_dist = torch.norm(F_gen - F_real, dim=1)
        # 随机采样不同类特征
        neg_mask = labels.unsqueeze(0) != labels.unsqueeze(1)
        neg_dist = torch.norm(F_gen.unsqueeze(1) - F_real.unsqueeze(0), dim=2)
        neg_dist = neg_dist[neg_mask].view(F_gen.size(0), -1)
        loss = torch.mean(torch.relu(pos_dist.unsqueeze(1) - neg_dist + self.margin))
        return loss


class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dim = dim

    def forward(self, source, target):
        Q = self.query(source)
        K = self.key(target)
        V = self.value(target)
        attn = self.softmax(Q @ K.transpose(-2, -1) / (self.dim ** 0.5))
        return attn @ V


class EnhancedGenerator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.attn = CrossModalAttention(512)
        self.fc2 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()

    def forward(self, F_sh):
        x = self.relu(self.fc1(F_sh))
        x = self.attn(x, x)  # 自注意力增强特征交互
        return self.fc2(x)


class EnhancedFMCModule(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.generator = EnhancedGenerator(feature_dim, feature_dim)
        self.discriminator = MultiLayerDiscriminator(feature_dim)
        self.identity_classifier = nn.Linear(feature_dim, num_classes)
        self.contrastive_loss = ContrastiveLoss(margin=1.0)

    def forward(self, F_sh_source, F_sp_target, labels):
        F_sp_gen = self.generator(F_sh_source)

        # 对抗损失
        real_score = self.discriminator(F_sp_target)
        fake_score = self.discriminator(F_sp_gen)
        adversarial_loss = -torch.mean(torch.log(real_score + 1e-8)) - torch.mean(torch.log(1 - fake_score + 1e-8))

        # 特征一致性损失
        feature_consistency_loss = F.l1_loss(F_sp_gen, F_sp_target)

        # 身份一致性损失
        identity_scores = self.identity_classifier(F_sp_gen)
        identity_consistency_loss = F.cross_entropy(identity_scores, labels)

        # 对比学习损失
        contrastive_loss = self.contrastive_loss(F_sp_gen, F_sp_target, labels)

        # 动态权重调整
        total_loss = (
                adversarial_loss +
                0.5 * feature_consistency_loss +
                0.5 * identity_consistency_loss +
                0.3 * contrastive_loss
        )

        return F_sp_gen, total_loss