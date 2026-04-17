# models/loop_lm_jax.py
import flax.linen as nn
import jax.numpy as jnp

class TransformerEncoderLayer(nn.Module):
    """JAX/Flax 严谨复刻的 Transformer 层 (Norm First)"""
    d_model: int
    nhead: int
    dim_feedforward: int

    @nn.compact
    def __call__(self, x):
        # Self Attention (Norm First)
        x_norm = nn.LayerNorm()(x)
        x_attn = nn.SelfAttention(num_heads=self.nhead)(x_norm)
        x = x + x_attn

        # Feed Forward (Norm First)
        x_norm_ff = nn.LayerNorm()(x)
        x_ff = nn.Dense(self.dim_feedforward)(x_norm_ff)
        x_ff = nn.gelu(x_ff)
        x_ff = nn.Dense(self.d_model)(x_ff)
        x = x + x_ff
        return x

class LoopLMJax(nn.Module):
    num_bin_channels: int
    num_global_channels: int
    d_model: int = 256
    n_loops: int = 4

    @nn.compact
    def __call__(self, board_x, global_x, train: bool = False):
        # 1. 强制 NCHW -> NHWC 转换 (TPU/JAX 极致优化的硬件格式)
        board_x = jnp.transpose(board_x, (0, 2, 3, 1))
        B = board_x.shape[0]

        # 2. 空间前端 (注意：BatchNorm 需要传递 use_running_average)
        x_b = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(board_x)
        x_b = nn.BatchNorm(use_running_average=not train, momentum=0.9)(x_b)
        x_b = nn.gelu(x_b)

        x_b = nn.Conv(features=self.d_model, kernel_size=(3, 3), padding='SAME')(x_b)
        x_b = nn.BatchNorm(use_running_average=not train, momentum=0.9)(x_b)
        x_b = nn.gelu(x_b)

        # 展平: [Batch, 19, 19, d_model] -> [Batch, 361, d_model]
        x_b = x_b.reshape((B, 361, self.d_model))

        # 3. 全局前端
        x_g = nn.Dense(self.d_model)(global_x)
        x_g = nn.gelu(x_g).reshape((B, 1, self.d_model))

        # 拼接: [Batch, 362, d_model]
        x = jnp.concatenate([x_b, x_g], axis=1)

        # 4. 绝对位置编码
        pos_embedding = self.param('pos_embedding', nn.initializers.normal(stddev=0.02), (1, 362, self.d_model))
        x = x + pos_embedding

        # 5. O(361^2) 全局推演循环
        for _ in range(self.n_loops):
            x = TransformerEncoderLayer(d_model=self.d_model, nhead=8, dim_feedforward=self.d_model * 4)(x)

        return x
