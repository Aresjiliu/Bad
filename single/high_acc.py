import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SpectralTransformerBlock(nn.Module):
    """光谱维度自注意力模块（动态适配输入维度）"""

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b (h w) (c d)')

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.proj(x)
        x = rearrange(x, 'b (h w) (c d) -> b c d h w', h=H, w=W, c=C, d=D)
        return x


class HybridConvTransformer(nn.Module):
    """混合卷积-Transformer超高精度模型（修复版）"""

    def __init__(self, in_channels=144, num_classes=10, embed_dim=512):
        super().__init__()

        # 光谱特征强化路径
        self.spectral_path = nn.Sequential(
            nn.Conv3d(1, 32, (16, 1, 1)),  # 关键修复：kernel_size=(16,1,1)
            nn.BatchNorm3d(32),
            nn.GELU(),
            SpectralTransformerBlock(32 * (in_channels // 16)),  # 32*(144//16)=288
            nn.Conv3d(32, 64, (8, 3, 3), padding=(0, 1, 1)),
            nn.MaxPool3d((1, 2, 2))
        )

        # 空间特征强化路径
        self.spatial_path = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 512, 3, groups=256, padding=1),
            nn.BatchNorm2d(512),
            SpectralTransformerBlock(512),
            nn.Conv2d(512, embed_dim, 1)
        )

        # 多尺度融合
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            SpatialAttention(embed_dim)
        )

        # 分类头
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x_spec = x.unsqueeze(1)  # [B,1,144,7,7]
        x_spec = self.spectral_path(x_spec)  # [B,32,9,7,7] → [B,64,4,7,7]
        x_spec = rearrange(x_spec, 'b c d h w -> b (c d) h w')  # [B,64*4,7,7]

        x_spat = self.spatial_path(x)  # [B,512,7,7]

        x = torch.cat([x_spec, x_spat], dim=1)  # [B,64*4+512,7,7]
        x = self.fusion(x)
        return self.head(x)


class SpatialAttention(nn.Module):
    """空间注意力（带坐标编码）"""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, in_channels, 3, padding=1)
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, _, H, W = x.shape
        coord_x = torch.linspace(-1, 1, W).view(1, 1, 1, W).expand(B, 1, H, W)
        coord_y = torch.linspace(-1, 1, H).view(1, 1, H, 1).expand(B, 1, H, W)
        coords = torch.cat([coord_x, coord_y], dim=1).to(x.device)

        x_coord = torch.cat([x, coords], dim=1)
        x_conv = self.conv(x_coord)
        attn = self.attn(x_conv)
        return x * attn.expand_as(x)


# ---------------------------
# 测试运行
# ---------------------------
if __name__ == "__main__":
    model = HybridConvTransformer(in_channels=144, num_classes=10)
    x = torch.randn(2, 144, 7, 7)
    print("输出尺寸:", model(x).shape)  # 预期输出: [2,10]