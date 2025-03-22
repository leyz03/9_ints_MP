import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(), # 激活函数 nn.GELU()：用于增加非线性。
            nn.Dropout(dropout), # nn.Dropout(dropout)：防止过拟合的 dropout 层。
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module): # PreNorm 类用于对输入进行归一化后再调用传入的函数 fn。
    def __init__(self, dim, fn): # __init__ 方法中使用 nn.LayerNorm(dim) 进行层归一化。
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) # forward 方法先对输入 x 进行归一化，然后调用 fn。

class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x1, x2, **kwargs):
        return self.fn(self.norm(x1), self.norm(x2), **kwargs)

class Attention(nn.Module): # Attention 类实现了多头自注意力机制。
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.): # 在 __init__ 中，inner_dim 是每个头的维度乘以头的数量，self.scale 用于缩放点积。
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim) # 其他属性包括层归一化、softmax 操作和 dropout。

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False) # to_qkv 是用于生成查询、键和值的线性层。

        self.to_out = nn.Sequential( # to_out 用于将输出映射回原始维度，条件是是否需要投影。
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x) # 在 forward 方法中，输入首先经过归一化。

        qkv = self.to_qkv(x).chunk(3, dim=-1) # qkv 通过 to_qkv 生成，并通过 chunk 拆分为查询 q、键 k 和值 v。
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # 通过 torch.matmul 计算查询和键的点积，得到注意力分数，随后进行 softmax。

        attn = self.attend(dots)
        attn = self.dropout(attn) # 使用 dropout，然后计算输出。

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)') # 最后将输出重塑并通过 to_out 映射回原始维度。
        return self.to_out(out)

class CrossAttention(nn.Module): # 定义交叉注意力模块 CrossAttention 类与 Attention 类相似，但在计算注意力时使用了两个输入。
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False) # to_q 用于处理第一个输入的查询，而 to_qkv 处理第二个输入的键和值。

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x1, x2): # forward 方法对两个输入 x1 和 x2 进行归一化。
        x1 = self.norm(x1)
        x2 = self.norm(x2)

        q1 = self.to_q(x1) # 查询通过 to_q 生成，并重塑为适合多头注意力的形状。
        q1 = rearrange(q1, 'b n (h d) -> b h n d', h=self.heads)
        qkv2 = self.to_qkv(x2).chunk(3, dim=-1)
        q2, k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv2)

        dots = torch.matmul(q1, k2.transpose(-1, -2)) * self.scale

        attn = self.attend(dots) # 计算查询和键的点积得到注意力分数，之后进行 softmax 和 dropout。
        attn = self.dropout(attn)

        out = torch.matmul(attn, v2)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out) # 计算输出并重塑，最后通过 to_out 映射回原始维度。

class Transformer(nn.Module): # 定义变压器模块
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.): # Transformer 类实现了一个标准的变压器结构，由多个层堆叠而成。
        super().__init__()
        self.layers = nn.ModuleList([]) # 每层包括一个自注意力模块和一个前馈网络，均经过归一化处理。
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x): # forward 方法依次通过每一层，执行自注意力和前馈操作，并加上残差连接。
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class CrossTransformer(nn.Module): # CrossTransformer 类实现了交叉变压器结构，允许使用两个输入进行注意力计算。
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([]) # 每层包括交叉注意力和前馈模块，前者使用 `PreNorm # 创建一个空的 ModuleList，用于存储每一层的子模块。每层会包括交叉注意力模块和前馈网络模块。
        for _ in range(depth): # 这是一个循环，用于添加多层（depth）到 self.layers 中。在每一层中，使用 nn.ModuleList 包含两个模块：
            self.layers.append(nn.ModuleList([
                PreNorm2(dim, CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)), # 第一个是交叉注意力模块 CrossAttention，它通过 PreNorm2 包装，PreNorm2 用于进行层归一化。
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)) # 第二个是前馈网络 FeedForward，同样通过 PreNorm 包装。
            ]))

    def forward(self, x1, x2): # 在这个循环中，遍历每层的注意力模块和前馈模块。对于每一层：
        for attn, ff in self.layers:
            x = attn(x1, x2) + x1 # 先使用交叉注意力模块 attn 计算注意力，输入为 x1 和 x2，然后将结果与 x1 相加。
            x = ff(x) + x # 然后将结果 x 输入到前馈模块 ff 中，得到的结果再与 x 相加。这样可以实现残差连接，帮助训练深层网络。
        return x

if __name__ == '__main__':
    input1 = torch.randn(4, 16, 32) # 生成一个随机输入张量 input1，形状为 (4, 16, 32)，表示有 4 个样本，每个样本有 16 个时间步（或特征），每个时间步有 32 个特征。
    # transformer = Transformer(dim=32, depth=4, heads=8, dim_head=64, mlp_dim=64, dropout=0.1) # 这段代码被注释掉了，如果取消注释，将会创建一个 Transformer 实例并对 input1 进行前向传播，得到 output1。
    # output1 = transformer(input1)

    input2 = torch.randn(12, 16, 32) # 生成另一个随机输入张量 input2，形状为 (12, 16, 32)，表示有 12 个样本。
    ca = CrossAttention(dim=32, heads=8, dim_head=64, dropout=0.1) # 创建一个 CrossAttention 的实例 ca，使用相应的参数设置。
    input1 = repeat(input1, 'b c n -> (repeat b) c n', repeat=3) # 使用 einops 库的 repeat 函数，将 input1 复制 3 次，使其形状变为 (12, 16, 32)。这里的 b 表示批次大小，c 是特征维度，n 是时间步数。
    output2 = ca(input1, input2) # 使用交叉注意力模块 ca 对 input1 和 input2 进行前向传播，得到 output2。
    print(0) # 最后打印 0，用于表示程序顺利执行完毕。