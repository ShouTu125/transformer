import torch
import copy
import math

import torch.nn.functional as F
import torch.nn as nn


# 克隆模块
def clone_module(module, N):
    """
        克隆模块
        
        Args:
            N (int): 模块副本数
        Returns:
            [torch.nn.Module]
    """
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 层标准化
class LayerNorm(torch.nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.gamma = torch.nn.Parameter(torch.ones(features))
        self.beta = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
def scaled_dot_product_attention(q, k, v, mask = None, dropout = None):
    """
    scaled dot product attention

    Args:
        q (Tensor): query
        k (Tensor): key
        v (Tensor): value
        mask (Tensor): mask
        dropout (float): dropout
    """
    d_k = q.size(-1)
    # 注意力分数矩阵
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # 加掩码
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    
    p = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        p = dropout(p)
    
    return torch.matmul(p, v), p


# 多头注意力
class MultiHeadAttention(torch.nn.Module):
    """
    多头注意力机制

    Args:
        h (int): 多头数量
        d_model (int): 输入输出维度
        dropout (float): dropout比例

    Returns:
        [torch.nn.Module]
    """
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clone_module(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask = None):
        nbatches = q.size(0)

        # 将输入最后一层转化成 d_model -> h * d_k
        _q, _k, _v = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) 
            for l, x in zip(self.linears, (q, k, v))]

        # 每个head执行attn
        x, self.attn = scaled_dot_product_attention(_q, _k, _v, mask, self.dropout)

        # 连接
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # feed-forward
        return self.linears[-1](x)

# 子层连接
class SublayerConnection(torch.nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()

        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer): 
        # 子层残差连接
        return x + self.dropout(self.norm(sublayer(x)))


# 编码器子层
class EncoderLayer(torch.nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size

        self.self_attn = self_attn
        self.feed_forward = feed_forward

        self.sublayer = clone_module(SublayerConnection(size, dropout), 2)
    
    def forward(self, x, mask):
        # self attention sublayer
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # position wise fully connected feed-forward sublayer
        return self.sublayer[1](x, self.feed_forward)

# 编码器
class Encoder(torch.nn.Module):
    def __init__(self, layer, N):
        """
        LayerNorm(x + Sublayer(x)) * N
        """
        super(Encoder, self).__init__()

        self.layers = clone_module(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)
    

# 解码器子层
class DecoderLayer(torch.nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size

        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward

        self.sublayer = clone_module(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # self attention sublayer
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # source attention sublayer
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        # position wise fully connected feed-forward sublayer
        return self.sublayer[2](x, self.feed_forward)

# 解码器
class Decoder(torch.nn.Module):
    def __init__(self, layer, N):
        """
        LayerNorm(x + Sublayer(x)) * N
        """
        super(Decoder, self).__init__()

        self.layers = clone_module(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return self.norm(x)
    

# 位置感知前馈网络
class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = torch.nn.Linear(d_model, d_ff)
        self.w_2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    

# 输入输出嵌入层
class Embeddings(torch.nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()

        self.lut = torch.nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    

# 位置编码
class PositionEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout, max_len = 5000):
        """
        位置编码, 使用`Sinusoidal Positional Encoding`

        Args:
            d_model (int): 输入输出维度
            dropout (float): dropout比例
            max_len (int): 最大长度

        Returns:
            [torch.Tensor]: 位置编码        
        """
        super(PositionEncoding, self).__init__()

        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register_buffer方法用于向模型中注册一个不需要梯度的缓冲区（buffer）
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)
    


# 模型输出头
class Generator(torch.nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()

        self.proj = torch.nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x)

class Transformer(torch.nn.Module):
    """
        A standard Transformer architecture
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

        self.generator = generator

    
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.generator(self.decoder(
            self.tgt_embed(tgt), 
            self.encoder(self.src_embed(src), src_mask), 
            src_mask,  
            tgt_mask))


# 定义模型
def make_model(src_vocab_size, tgt_vocab_size, n_layers=6, d_model=512, d_ff=2048, n_heads=8, dropout=0.1):
    """
    make model

    Args:
        src_vocab_size (int): source vocab size
        tgt_vocab_size (int): target vocab size
        n_layers (int): number of encoder/decoder layers
        d_model (int): model dimension
        d_ff (int): feed-forward dimension
        n_heads (int): number of heads
        dropout (float): dropout

    Returns:
        model (Transformer)
    """
    c = copy.deepcopy
    attn = MultiHeadAttention(h=n_heads, d_model=d_model)
    ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)
    position = PositionEncoding(d_model=d_model, dropout=dropout)


    gen = Generator(d_model=d_model, vocab=tgt_vocab_size)

    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n_layers),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n_layers),

        torch.nn.Sequential(Embeddings(d_model, src_vocab_size), c(position)),
        torch.nn.Sequential(Embeddings(d_model, tgt_vocab_size), c(position)),

        gen,
    )

    # 初始化模型参数
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model