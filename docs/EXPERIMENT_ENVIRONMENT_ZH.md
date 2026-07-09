# 实验环境记录

日期：2026-07-10

## Conda 环境

后续 BRM-Net / HSLiNets 相关实验默认使用：

```powershell
conda activate hslinets
```

非交互脚本推荐写成：

```powershell
conda run -n hslinets python <script.py> ...
```

## 已验证环境信息

```text
python = D:\software\anaconda3\envs\hslinets\python.exe
torch = 2.5.1
cuda_available = True
device_count = 1
device_name = NVIDIA GeForce RTX 4060 Laptop GPU
```

## 当前可用数据路径

HSLiNets 预切片 Houston/Huston 数据目录：

```text
D:\Academic\HSLiNets-main\Huston2013
```

当前 `scripts/run_brmnet_houston.py` 可使用 `--data-format legacy` 读取该目录。

