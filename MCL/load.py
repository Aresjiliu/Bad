# -*- coding: utf-8 -*-
import torch

def load_and_print_keys(pth_file_path):
    """
    加载一个 .pth 文件，并以每行三个键的格式打印所有的键。

    参数：
    - pth_file_path (str): .pth 文件的路径

    返回：
    - None
    """
    try:
        # 加载 .pth 文件
        checkpoint = torch.load(pth_file_path, map_location='cpu')  # 使用 'cpu' 避免 GPU 相关问题

        # 检查加载的对象是否为字典
        if not isinstance(checkpoint, dict):
            print(f"加载的对象不是字典类型，而是 {type(checkpoint)}。请确保 .pth 文件包含一个字典。")
            return

        # 获取所有的键
        keys = list(checkpoint.keys())

        if not keys:
            print("加载的字典中没有键。")
            return

        print(f"总共有 {len(keys)} 个键。以每行三个键的格式打印如下：\n")

        # 以每行三个键的格式打印
        for i in range(0, len(keys), 3):
            # 获取当前行的三个键
            current_keys = keys[i:i+3]
            # 将键连接成一个字符串，以逗号分隔
            line = ', '.join(current_keys)
            print(line)

    except FileNotFoundError:
        print(f"文件未找到：{pth_file_path}")
    except Exception as e:
        print(f"加载 .pth 文件时发生错误：{e}")

# 示例用法
if __name__ == "__main__":
    pth_path = "hsi_lidardw_prune.pth"  # 替换为您的 .pth 文件路径
    load_and_print_keys(pth_path)
