"""
Vision Transformer (ViT) 模型定义。

超参数：图像 28x28、patch 7x7（切分为 4x4=16 个 patch）、嵌入维度 64、
编码器层数 4、注意力头数 4、MLP 扩张倍数 4.0、dropout 0.1。
"""
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    图像分块嵌入层

    将 28x28 的 MNIST 图像切分成固定大小的 patch，
    然后通过卷积操作将每个 patch 映射到高维向量空间。

    例如：28x28 图像，patch_size=7，则切分为 4x4=16 个 patch
    """

    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=64):
        """
        参数说明：
        img_size: 输入图像尺寸（MNIST 为 28x28）
        patch_size: 每个 patch 的大小（7x7 像素）
        in_channels: 输入通道数（灰度图为 1）
        embed_dim: 嵌入维度，即每个 patch 映射后的向量长度
        """
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2

        # 使用卷积层实现 patch 划分 + 线性投影
        # kernel_size 和 stride 都等于 patch_size，相当于非重叠地切分图像
        self.projection = nn.Conv2d(
            in_channels,  # 输入通道：1
            embed_dim,  # 输出通道：64（即嵌入维度）
            kernel_size=patch_size,  # 卷积核大小 = patch 大小
            stride=patch_size  # 步长 = patch 大小（不重叠）
        )

    def forward(self, x):
        """
        前向传播
        输入：x - (batch_size, 1, 28, 28) 的图像 tensor
        输出：(batch_size, 16, 64) 的 patch 嵌入序列
        """
        x = self.projection(x)  # (B, 64, 4, 4) - 卷积提取 patch 特征
        x = x.flatten(2).transpose(1, 2)  # (B, 16, 64) - 展平并转置为序列格式
        return x


class TransformerBlock(nn.Module):
    """
    Transformer 编码器块
    包含：
    1. 多头自注意力机制（Multi-Head Self-Attention）
    2. 前馈神经网络（Feed Forward Network）
    3. 残差连接 + 层归一化
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        """
        参数说明：
        embed_dim: 嵌入维度（64）
        num_heads: 注意力头数（4）
        mlp_ratio: MLP 隐藏层维度的放大倍数
        dropout: Dropout 比率，防止过拟合
        """
        super().__init__()
        # 第一个子层：多头自注意力
        self.norm1 = nn.LayerNorm(embed_dim)  # 层归一化，在特征维度上归一化
        self.attn = nn.MultiheadAttention(
            embed_dim,  # 输入维度
            num_heads,  # 注意力头数，每个头处理 embed_dim//num_heads 维
            dropout=dropout,
            batch_first=True  # 输入格式为 (batch, seq, feature)
        )

        # 第二个子层：前馈神经网络
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            # 第一层线性变换：64 -> 256（扩张）
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),  # 激活函数，比 ReLU 更平滑
            nn.Dropout(dropout),
            # 第二层线性变换：256 -> 64（压缩回原维度）
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        前向传播：LayerNorm -> Self-Attention -> 残差连接
                 LayerNorm -> MLP -> 残差连接
        """
        # 自注意力子层：先归一化，再做注意力，最后残差连接
        # (Pre-Norm 结构，比 Post-Norm 训练更稳定)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)  # Q=K=V，自注意力
        x = x + attn_out  # 残差连接，缓解梯度消失

        # 前馈网络子层
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) 完整模型

    整体流程：
    1. 图像分块并嵌入
    2. 添加 CLS token 和位置编码
    3. 通过多层 Transformer 编码
    4. 取 CLS token 输出进行分类
    """

    def __init__(self, img_size=28, patch_size=7, in_channels=1, num_classes=10,
                 embed_dim=64, depth=4, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        """
        参数说明：
        img_size: 图像尺寸
        patch_size: patch 大小
        in_channels: 输入通道
        num_classes: 分类数（MNIST 为 10 类数字 0-9）
        embed_dim: 嵌入维度
        depth: Transformer 编码器层数
        num_heads: 注意力头数
        mlp_ratio: MLP 扩张倍数
        dropout: Dropout 比率
        """
        super().__init__()
        # 图像分块嵌入模块
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches  # patch 数量 = 16

        # 可学习的 CLS token（分类令牌）
        # 用于聚合整个图像的信息，最终用于分类
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 可学习的位置编码
        # 给每个 patch 和 CLS token 添加位置信息（Transformer 本身不感知位置）
        # 长度 = 16 个 patch + 1 个 CLS = 17
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(dropout)  # 位置编码后的 dropout

        # 堆叠多层 Transformer 编码器
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)  # 重复 4 层
        ])

        # 最终的层归一化
        self.norm = nn.LayerNorm(embed_dim)

        # 分类头：将 CLS token 的输出映射到类别数
        self.head = nn.Linear(embed_dim, num_classes)

        # 参数初始化（截断正态分布初始化）
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        """
        前向传播
        输入：x - (batch_size, 1, 28, 28)
        输出：logits - (batch_size, 10) 未归一化的分类分数
        """
        B = x.shape[0]  # 获取 batch 大小

        # Step 1: 图像分块嵌入
        # (B, 1, 28, 28) -> (B, 16, 64)
        x = self.patch_embed(x)

        # Step 2: 添加 CLS token
        # CLS token 扩展到 batch 大小，然后拼接到序列最前面
        # (B, 16, 64) -> (B, 17, 64)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Step 3: 添加位置编码 + Dropout
        x = self.pos_drop(x + self.pos_embed)

        # Step 4: 通过 Transformer 编码器层
        x = self.blocks(x)

        # Step 5: 最终归一化
        x = self.norm(x)

        # Step 6: 取 CLS token（第一个位置）的输出进行分类
        return self.head(x[:, 0])
