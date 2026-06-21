import torch

# 假设`model_state_dict`是你加载的模型的状态字典
model_state_dict = torch.load('../output/models/transfer_berlin/1.0fin.pth')

# 需要删除的键列表
keys_to_remove = ["share_bone.block_6.0.weight", "share_bone.block_6.0.bias"]

# 删除指定的键
for key in keys_to_remove:
    if key in model_state_dict:
        del model_state_dict[key]

# 保存修改后的状态字典，如果你需要的话
torch.save(model_state_dict, '../output/models/transfer_berlin/1.0fin.pth')