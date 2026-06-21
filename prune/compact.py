import torch
import torch.nn as nn

def process_keys_and_save(new_pth_path, original_pth_path):
    """
    处理.pth文件中的键，对符合条件的键进行处理并保存新的.pth文件。

    :param new_pth_path: 新的.pth文件的路径
    :param original_pth_path: 原始.pth文件的路径
    :param transfer_func: 用于处理权重的函数
    """
    # 加载原始.pth文件
    state_dict = torch.load(original_pth_path)
    # 用于存储新的键值对
    new_state_dict = {}
    # 遍历原始状态字典中的所有键值对
    keys = list(state_dict.keys())
    i = 0
    while i < (len(keys) - 3):
        # 检查当前键以及接下来的两个键是否符合模式
        if 'weight' in keys[i] and 'sc' in keys[i + 1] and 'temp' in keys[i + 2]:
            # 提取三个连续的键
            weight_key, sc_key, temp_key, saved = keys[i], keys[i + 1], keys[i + 2], keys[i+3]
            new_state_dict[weight_key] = state_dict[saved]
            i = i + 4
        else:
            # 对于不符合条件的键，直接复制到新的字典
            new_state_dict[keys[i]] = state_dict[keys[i]]
            i = i + 1

    # 处理剩余的键（如果有的话）
    for j in range(i, len(keys)):
        new_state_dict[keys[j]] = state_dict[keys[j]]
    # 保存新的字典到新的.pth文件
    torch.save(new_state_dict, new_pth_path)


def compact_conv2d(origin_weights):
    # 移除权重为0的通道
    non_zero_indices = torch.nonzero(origin_weights.sum(dim=[1, 2, 3]) != 0, as_tuple=False).squeeze()
    compact_weights = origin_weights[non_zero_indices]
    # 返回被剪枝的通道索引
    return compact_weights, non_zero_indices

def get_dim2(weights, dices):
    a = weights.clone()
    b = dices.view(1, -1, 1, 1).expand(a.size(0), -1, a.size(2), a.size(3))
    return torch.gather(a, 1, b)


def compact_model(original_pth_path, new_path):
    state_dict = torch.load(original_pth_path)
    new_state_dict = {}
    keys = list(state_dict.keys())
    for i in range(18):
        name = keys[i]
        name_dict = name.split(".")
        if name_dict[-2] == "0":
            if name_dict[-3] != "block1":
                a = get_dim2(state_dict[name], dices1)
            else:
                a = state_dict[name]
            b, dices1 = compact_conv2d(a)
            new_state_dict[name] = b
        else:
            a = state_dict[name]
            if name_dict[-1] == "num_batches_tracked":
                new_state_dict[name] = a
            else:
                new_state_dict[name] = a[dices1]
    for i in range(18, 36):
        name = keys[i]
        name_dict = name.split(".")
        if name_dict[-2] == "0":
            if name_dict[-3] != "block1":
                a = get_dim2(state_dict[name], dices2)
            else:
                a = state_dict[name]
            b, dices2 = compact_conv2d(a)
            new_state_dict[name] = b
        else:
            a = state_dict[name]
            if name_dict[-1] == "num_batches_tracked":
                new_state_dict[name] = a
            else:
                new_state_dict[name] = a[dices2]
    dices3 = torch.cat((dices1, dices2.add(128)), dim=0)
    for i in range(36, len(keys)):
        name = keys[i]
        name_dict = name.split(".")
        if name_dict[-2] == "0":
            a = get_dim2(state_dict[name], dices3)
            b, dices3 = compact_conv2d(a)
            new_state_dict[name] = b
        else:
            a = state_dict[name]
            if name_dict[-1] == "num_batches_tracked":
                new_state_dict[name] = a
            elif name == "share_bone.fc.bias":
                new_state_dict[name] = a
            elif name == "share_bone.fc.weight":
                new_state_dict[name] = a[:, dices3]
            else:
                new_state_dict[name] = a[dices3]
    torch.save(new_state_dict, new_path)

process_keys_and_save('../output/models/transfer_berlin/0.05su.pth',
                      "../output/models/prune_berlin_succ/hsi1_sar_osc_0.05_l1_0.001_gama_1.01.pth")
process_keys_and_save('../output/models/transfer_berlin/0.1su.pth',
                      "../output/models/prune_berlin_succ/hsi1_sar_osc_0.1_l1_0.001_gama_1.01.pth")
process_keys_and_save('../output/models/transfer_berlin/0.5su.pth',
                      "../output/models/prune_berlin_succ/hsi1_sar_osc_0.5_l1_0.001_gama_1.01.pth")
process_keys_and_save('../output/models/transfer_berlin/0.7su.pth',
                      "../output/models/prune_berlin_succ/hsi1_sar_osc_0.7_l1_0.001_gama_1.01.pth")
process_keys_and_save('../output/models/transfer_berlin/0.9su.pth',
                      "../output/models/prune_berlin_succ/hsi1_sar_osc_0.9_l1_0.001_gama_1.01.pth")

compact_model("../output/models/transfer_berlin/0.05su.pth", "../output/models/transfer_berlin/0.05fin.pth")
compact_model("../output/models/transfer_berlin/0.1su.pth", "../output/models/transfer_berlin/0.1fin.pth")
compact_model("../output/models/transfer_berlin/0.5su.pth", "../output/models/transfer_berlin/0.5fin.pth")
compact_model("../output/models/transfer_berlin/0.7su.pth", "../output/models/transfer_berlin/0.7fin.pth")
compact_model("../output/models/transfer_berlin/0.9su.pth", "../output/models/transfer_berlin/0.9fin.pth")
