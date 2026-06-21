import torch


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

path1 = "../output/models/prune_huston_succ/hsi_lidar_osc_0.9_l1_0.001_gama_1.01.pth"
# 使用示例
process_keys_and_save('../output/models/transfer_berlin/0.05su.pth',
                      "../output/models/prune_berlin_succ/hsi1_sar_osc_0.1_l1_0.001_gama_1.01.pth")
