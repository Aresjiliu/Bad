from pathlib import Path
import cv2
import numpy as np

root = Path('/data1/yco/TGRS/output/depose/depose_intra')                   # 根目录
subdirs = [root / f'depose_intra{i}' for i in [49, 99, 149, 199, 249, 299]]
out_dir = root / 'merged'
out_dir.mkdir(exist_ok=True)

# 图片尺寸（可按需修改）
H, W = 256, 256        # 假设原图已统一尺寸；若不同请 cv2.resize

for cls in range(15):
    imgs = []
    for sd in subdirs:
        img_path = sd / f'class{cls}.png'
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.resize(img, (W, H))
        imgs.append(img)

    # 拼成 2×3 网格
    row1 = np.hstack(imgs[:3])
    row2 = np.hstack(imgs[3:])
    merged = np.vstack([row1, row2])

    cv2.imwrite(str(out_dir / f'class{cls}.png'), merged)

print('15 张拼接图已生成到', out_dir.resolve())