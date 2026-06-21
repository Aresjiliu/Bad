from models1 import *
import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设输入特征维度 [batch_size, 256, 7, 7]
batch_size = 128
dim = 128
spatial_size = 7

# # 创建模拟输入
# F_sp_V = torch.randn(batch_size, dim, spatial_size, spatial_size)
# F_sh_V = torch.randn(batch_size, dim, spatial_size, spatial_size)
# F_sp_I = torch.randn(batch_size, dim, spatial_size, spatial_size)
# F_sh_I = torch.randn(batch_size, dim, spatial_size, spatial_size)
#
# # 创建模拟标签 (假设有4个身份)
# Y_V = torch.randint(0, 15, (batch_size,))
#
#
# # 初始化损失函数
# md_loss = SpatialMDLoss()
#
# # 计算损失
# loss = md_loss(F_sp_V, F_sh_V, F_sp_I, F_sh_I, Y_V)
# print(f"MD Loss: {loss.item():.4f}")
batch_size = 16
dim = 256
spatial_size = 7
num_classes = 16

# 创建模拟输入
F_sh = torch.randn(batch_size, dim, spatial_size, spatial_size)
F_sp = torch.randn(batch_size, dim, spatial_size, spatial_size)

# 初始化模型
processor = SingleModalityFusionClassifier(dim, num_classes)

# 前向传播
outputs = processor(F_sh, F_sp)


print("融合特征形状:", outputs[0].shape)  # [16, 256, 7, 7]
print("融合分类logits形状:", outputs[1].shape)  # [16, 100]