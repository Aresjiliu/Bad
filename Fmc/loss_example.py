import torch
import torch.nn as nn
import torch.nn.functional as F


# 对抗损失 (原始GAN损失)
def adversarial_loss(real_pred, fake_pred, for_discriminator=True):
    if for_discriminator:
        real_loss = -torch.mean(torch.log(real_pred + 1e-8))
        fake_loss = -torch.mean(torch.log(1 - fake_pred + 1e-8))
        return real_loss + fake_loss
    else:
        return -torch.mean(torch.log(fake_pred + 1e-8))


# 特征一致性损失 (L1损失)
def feature_consistency_loss(fake_features, centers, labels):
    # centers: 字典或张量，存储每个身份的特征中心
    target_centers = centers[labels]
    return torch.mean(torch.abs(fake_features - target_centers))


# 身份一致性损失 (交叉熵损失)
def identity_consistency_loss(fake_features, identity_classifier, labels):
    pred_scores = identity_classifier(fake_features)
    return F.cross_entropy(pred_scores, labels)


# 完整的FMC模块损失计算
def compute_fmc_losses(G_V2I, G_I2V, D_V2I, D_I2V,
                       F_sh_V, F_sh_I, F_sp_V, F_sp_I,
                       centers_sp_V, centers_sp_I,
                       P_sp_V, P_sp_I,
                       Y_V, Y_I,
                       λ_fc=1.0, λ_id=1.0):
    # 生成假特征
    F_sp_I_fake = G_V2I(F_sh_V)
    F_sp_V_fake = G_I2V(F_sh_I)

    # 判别器预测
    D_real_V2I = D_V2I(F_sp_I)
    D_fake_V2I = D_V2I(F_sp_I_fake.detach())
    D_real_I2V = D_I2V(F_sp_V)
    D_fake_I2V = D_I2V(F_sp_V_fake.detach())

    # 判别器损失
    loss_D_V2I = adversarial_loss(D_real_V2I, D_fake_V2I, for_discriminator=True)
    loss_D_I2V = adversarial_loss(D_real_I2V, D_fake_I2V, for_discriminator=True)
    loss_D = loss_D_V2I + loss_D_I2V

    # 重新通过判别器(用于生成器训练)
    D_fake_V2I = D_V2I(F_sp_I_fake)
    D_fake_I2V = D_I2V(F_sp_V_fake)

    # 生成器损失
    adv_loss_V2I = adversarial_loss(None, D_fake_V2I, for_discriminator=False)
    adv_loss_I2V = adversarial_loss(None, D_fake_I2V, for_discriminator=False)
    adv_loss = adv_loss_V2I + adv_loss_I2V

    fc_loss_V2I = feature_consistency_loss(F_sp_I_fake, centers_sp_I, Y_I)
    fc_loss_I2V = feature_consistency_loss(F_sp_V_fake, centers_sp_V, Y_V)
    fc_loss = fc_loss_V2I + fc_loss_I2V

    id_loss_V2I = identity_consistency_loss(F_sp_I_fake, P_sp_I, Y_I)
    id_loss_I2V = identity_consistency_loss(F_sp_V_fake, P_sp_V, Y_V)
    id_loss = id_loss_V2I + id_loss_I2V

    loss_G = adv_loss + λ_fc * fc_loss + λ_id * id_loss

    return loss_D, loss_G, {
        'adv_loss': adv_loss.item(),
        'fc_loss': fc_loss.item(),
        'id_loss': id_loss.item(),
        'F_sp_I_fake': F_sp_I_fake,
        'F_sp_V_fake': F_sp_V_fake
    }