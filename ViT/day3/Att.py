# ViT Online Class
# Author: Dr. Zhu
# Project: PaddleViT (https://github.com/BR-IDL/PaddleViT)
# 2021.11
import paddle
import paddle.nn as nn
import pdb

paddle.set_device('cpu')


class Attention(nn.Layer):
    def __init__(self, embed_dim, num_heads, qkv_bias=False, qk_scale=None, dropout=0., attention_dropout=0.):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = int(embed_dim / num_heads)
        self.all_head_dim = self.head_dim * num_heads
        self.qkv = nn.Linear(embed_dim,
                             self.all_head_dim * 3,
                             bias_attr=False if qkv_bias is False else None)
        self.scale = self.head_dim ** -0.5 if qk_scale is None else qk_scale
        self.softmax = nn.Softmax(-1)
        self.proj = nn.Linear(self.all_head_dim, embed_dim)

    def transpose_multi_head(self, x):
        # x:[B, N, num_patches, head_dim]
        new_shape = x.shape[:-1] + [self.num_heads, self.head_dim]
        x = x.reshape(new_shape)
        # x: [B, N, num_heads, head_dim]
        x = x.transpose([0, 2, 1, 3])
        # x: [B, num_heads, num_patches, head_dim]
        return x

    def forward(self, x):
        pdb.set_trace()
        B, N, _ = x.shape
        qkv = self.qkv(x).chunk(3, -1)  # [B, N, all_head_dim] * 3
        q, k, v = map(self.transpose_multi_head, qkv)  # q,k,v:[B, num_heads, num_patches, head_dim]

        attn = paddle.matmul(q, k, transpose_y=True)  # q * k'
        attn = self.scale * attn
        attn = self.softmax(attn)
        attn_weight = attn
        # attn: [B, num_heas, num_patches, num_patches]

        out = paddle.matmul(attn, v)  # softmax(scale*(1*k'))*v
        out = out.transpose([0, 2, 1, 3])
        # attn: [B, num_heas, num_patches, head_dim]
        out = out.reshape([B, N, -1])
        out = self.proj(out)
        return out, attn_weight


def main():
    t = paddle.randn([4, 16, 96])
    print('input shape = ', t.shape)

    model = Attention(embed_dim=96, num_heads=8,
                      qkv_bias=False, qk_scale=None, dropout=0., attention_dropout=0.)
    print(model)

    out, attn_weights = model(t)
    print(out.shape)
    print(attn_weights.shape)


if __name__ == "__main__":
    main()
