import torch
import matplotlib.pyplot as plt
# 加载.pth文件
model_state_dict = torch.load('../output/models/transfer/0.5fin.pth')

# 获取所有的键
# keys = list(model_state_dict.keys())

# # 每三个键一行打印
# for i in range(0, len(keys), 3):
#     print(', '.join(keys[i:i+3]))
hsi = model_state_dict["special_bone_hsi.block3.0.weight"]
lidar = model_state_dict["special_bone_lidar.block3.0.weight"]
# hsi_mean = lidar.mean(dim=[1, 2, 3])
# vector = hsi_mean.cpu().numpy()
#
# # 创建一个折线图
# plt.plot(vector)
#
# # 设置图表标题和坐标轴标签
# plt.title('Tensor Vector Line Plot')
# plt.xlabel('Index')
# plt.ylabel('Value')
#
# # 显示图表
# plt.show()
print(hsi.shape)
print(lidar.shape)