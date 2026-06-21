# 初始化所有网络和优化器
G_V2I, G_I2V = init_generators()
D_V2I, D_I2V = init_discriminators()
optimizer_G
optimizer_D

# 预计算身份特征中心 (使用内存库或当前批次)
centers_sp_I = compute_centers(F_sp_I, Y_I)  # 红外模态特定特征中心
centers_sp_V = compute_centers(F_sp_V, Y_V)  # 可见光模态特定特征中心

for epoch in range(num_epochs):
    for batch_idx, (F_sh_V, F_sh_I, F_sp_V, F_sp_I, Y) in enumerate(dataloader):

        # ==================== 训练判别器 ====================
        optimizer_D.zero_grad()

        # 生成假特征
        F_sp_I_fake = G_V2I(F_sh_V)
        F_sp_V_fake = G_I2V(F_sh_I)

        # 计算判别器损失
        real_loss_V2I = -torch.mean(torch.log(D_V2I(F_sp_I) + 1e-8))
        fake_loss_V2I = -torch.mean(torch.log(1 - D_V2I(F_sp_I_fake.detach()) + 1e-8))
        loss_D_V2I = real_loss_V2I + fake_loss_V2I

        real_loss_I2V = -torch.mean(torch.log(D_I2V(F_sp_V) + 1e-8))
        fake_loss_I2V = -torch.mean(torch.log(1 - D_I2V(F_sp_V_fake.detach()) + 1e-8))
        loss_D_I2V = real_loss_I2V + fake_loss_I2V

        loss_D = loss_D_V2I + loss_D_I2V
        loss_D.backward()
        optimizer_D.step()

        # ==================== 训练生成器 ====================
        optimizer_G.zero_grad()

        # 重新生成特征(因为判别器已更新)
        F_sp_I_fake = G_V2I(F_sh_V)
        F_sp_V_fake = G_I2V(F_sh_I)

        # 计算生成器损失
        # 1. 对抗损失
        adv_loss_V2I = -torch.mean(torch.log(D_V2I(F_sp_I_fake) + 1e-8))
        adv_loss_I2V = -torch.mean(torch.log(D_I2V(F_sp_V_fake) + 1e-8))
        adv_loss = adv_loss_V2I + adv_loss_I2V

        # 2. 特征一致性损失 (L1损失)
        fc_loss_V2I = torch.mean(torch.abs(F_sp_I_fake - centers_sp_I[Y_I]))
        fc_loss_I2V = torch.mean(torch.abs(F_sp_V_fake - centers_sp_V[Y_V]))
        fc_loss = fc_loss_V2I + fc_loss_I2V

        # 3. 身份一致性损失 (交叉熵损失)
        # 假设已有预训练的身份分类器 P_sp_I 和 P_sp_V
        S_sp_I_fake = P_sp_I(F_sp_I_fake)
        S_sp_V_fake = P_sp_V(F_sp_V_fake)
        id_loss_V2I = F.cross_entropy(S_sp_I_fake, Y)
        id_loss_I2V = F.cross_entropy(S_sp_V_fake, Y)
        id_loss = id_loss_V2I + id_loss_I2V

        # 总生成器损失
        loss_G = adv_loss + λ_fc * fc_loss + λ_id * id_loss

        loss_G.backward()
        optimizer_G.step()

        # 记录损失和可视化
        if batch_idx % log_interval == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, D_loss: {loss_D.item()}, G_loss: {loss_G.item()}")
            # 可选: 使用TensorBoard记录损失
            # writer.add_scalar('Loss/D', loss_D.item(), global_step)
            # writer.add_scalar('Loss/G', loss_G.item(), global_step)