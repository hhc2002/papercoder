# PaperCoder — Attention Is All You Need

好的，根据您提供的论文描述和关键公式，我将为您生成 Transformer 模型的核心组件的 Python 代码骨架。

```python
# pip install torch
# pip install numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

# ----------------------------------------------------------------------------
# 1. Positional Encoding
# ----------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    在输入嵌入中加入位置信息。
    对应论文 Section 3.5 "Positional Encoding"。

    Args:
        d_model (int): 模型的维度 (embedding dimension)。
        dropout (float): Dropout 概率。
        max_len (int): 输入序列的最大长度。
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe) # 将 pe 注册为 buffer，不会被优化器更新

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 (seq_len, batch_size, d_model)。
        Returns:
            torch.Tensor: 加入位置编码后的张量。
        """
        # 将位置编码加到输入上
        # x.size(0) 是序列长度
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# ----------------------------------------------------------------------------
# 2. Scaled Dot-Product Attention
# ----------------------------------------------------------------------------

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 Scaled Dot-Product Attention。
    对应论文公式 (3)。

    Args:
        query (torch.Tensor): 查询矩阵，形状为 (batch_size, num_heads, seq_len_q, d_k)。
        key (torch.Tensor): 键矩阵，形状为 (batch_size, num_heads, seq_len_k, d_k)。
        value (torch.Tensor): 值矩阵，形状为 (batch_size, num_heads, seq_len_v, d_v)。
        mask (Optional[torch.Tensor]): 掩码张量，用于屏蔽掉不应被注意的元素。
                                     形状通常为 (batch_size, 1, seq_len_q, seq_len_k) 或 (batch_size, 1, 1, seq_len_k)。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - attention_output (torch.Tensor): 注意力加权后的值，形状为 (batch_size, num_heads, seq_len_q, d_v)。
            - attention_weights (torch.Tensor): 注意力权重，形状为 (batch_size, num_heads, seq_len_q, seq_len_k)。
    """
    d_k = query.size(-1)
    # 计算 QK^T
    # (batch_size, num_heads, seq_len_q, d_k) @ (batch_size, num_heads, d_k, seq_len_k)
    # -> (batch_size, num_heads, seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1))

    # 缩放
    scores = scores / math.sqrt(d_k)

    # 应用掩码 (如果存在)
    if mask is not None:
        # 将掩码中的 0 替换为一个非常小的负数，这样 softmax 后权重会接近 0
        scores = scores.masked_fill(mask == 0, -1e9)

    # 计算 Softmax 得到注意力权重
    attention_weights = F.softmax(scores, dim=-1)

    # 加权求和 V
    # (batch_size, num_heads, seq_len_q, seq_len_k) @ (batch_size, num_heads, seq_len_v, d_v)
    # -> (batch_size, num_heads, seq_len_q, d_v)
    attention_output = torch.matmul(attention_weights, value)

    return attention_output, attention_weights

# ----------------------------------------------------------------------------
# 3. Multi-Head Attention
# ----------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    实现 Multi-Head Attention 机制。
    对应论文 Section 3.2.2 "Multi-Head Attention"。

    Args:
        d_model (int): 模型的总维度 (embedding dimension)。
        num_heads (int): 注意力头的数量。
        dropout (float): Dropout 概率。
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # 每个头的维度

        # 线性层，用于将输入 Q, K, V 投影到不同的子空间
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 最终的输出线性层
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query (torch.Tensor): 查询输入，形状为 (batch_size, seq_len_q, d_model)。
            key (torch.Tensor): 键输入，形状为 (batch_size, seq_len_k, d_model)。
            value (torch.Tensor): 值输入，形状为 (batch_size, seq_len_v, d_model)。
            mask (Optional[torch.Tensor]): 掩码张量。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - attention_output (torch.Tensor): Multi-Head Attention 的输出，形状为 (batch_size, seq_len_q, d_model)。
                - attention_weights (torch.Tensor): 最终的注意力权重（所有头的平均或拼接后的权重）。
        """
        batch_size = query.size(0)

        # 1. 线性投影并分割到多个头
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)

        # 将 d_model 分割到 num_heads
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        q = q.view(batch_size, -1, self.num_heads, self.d_k)
        k = k.view(batch_size, -1, self.num_heads, self.d_k)
        v = v.view(batch_size, -1, self.num_heads, self.d_k)

        # 转置以匹配 scaled_dot_product_attention 的输入格式
        # (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 2. 计算 Scaled Dot-Product Attention
        # attention_output 形状: (batch_size, num_heads, seq_len_q, d_k)
        # attention_weights 形状: (batch_size, num_heads, seq_len_q, seq_len_k)
        attention_output, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # 3. 拼接所有头的输出
        # (batch_size, num_heads, seq_len_q, d_k) -> (batch_size, seq_len_q, num_heads, d_k)
        attention_output = attention_output.transpose(1, 2).contiguous()
        # (batch_size, seq_len_q, num_heads, d_k) -> (batch_size, seq_len_q, d_model)
        attention_output = attention_output.view(batch_size, -1, self.d_model)

        # 4. 应用最终的线性层
        # (batch_size, seq_len_q, d_model) -> (batch_size, seq_len_q, d_model)
        output = self.W_o(attention_output)
        output = self.dropout(output)

        # 注意力权重可能需要根据具体应用进行处理（例如，平均或返回第一个头的权重）
        # 这里返回所有头的权重，形状为 (batch_size, num_heads, seq_len_q, seq_len_k)
        return output, attention_weights

# ----------------------------------------------------------------------------
# 4. Position-wise Feed-Forward Network
# ----------------------------------------------------------------------------

class PositionwiseFeedForward(nn.Module):
    """
    实现 Position-wise Feed-Forward Network。
    对应论文 Section 3.2.2 "Feed Forward Network"。

    Args:
        d_model (int): 模型的维度。
        d_ff (int): 前馈网络中间层的维度。
        dropout (float): Dropout 概率。
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)。
        Returns:
            torch.Tensor: 前馈网络输出，形状为 (batch_size, seq_len, d_model)。
        """
        # TODO: 实现前馈网络逻辑
        # 1. 第一个线性层 + ReLU 激活
        # 2. Dropout
        # 3. 第二个线性层
        # 4. 返回结果
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# ----------------------------------------------------------------------------
# 5. Encoder Layer
# ----------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    """
    Transformer 的一个编码器层。
    包含 Multi-Head Self-Attention 和 Position-wise Feed-Forward Network。
    对应论文 Section 3.2 "The Transformer - Encoder and Decoder Stacks"。

    Args:
        d_model (int): 模型的维度。
        num_heads (int): 注意力头的数量。
        d_ff (int): 前馈网络中间层的维度。
        dropout (float): Dropout 概率。
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)。
            mask (Optional[torch.Tensor]): 自注意力掩码。

        Returns:
            torch.Tensor: 编码器层输出，形状为 (batch_size, seq_len, d_model)。
        """
        # TODO: 实现编码器层的前向传播
        # 1. Multi-Head Self-Attention
        #    - 应用残差连接 (x + self.dropout1(attn_output))
        #    - 应用层归一化 (self.norm1(...))
        # 2. Position-wise Feed-Forward Network
        #    - 应用残差连接 (sublayer_output + self.dropout2(ff_output))
        #    - 应用层归一化 (self.norm2(...))
        # 3. 返回最终输出

        # Self-Attention sub-layer
        attn_output, _ = self.self_attn(x, x, x, mask)
        x_residual = x + self.dropout1(attn_output)
        x_norm1 = self.norm1(x_residual)

        # Feed-Forward sub-layer
        ff_output = self.feed_forward(x_norm1)
        output = x_norm1 + self.dropout2(ff_output)
        output = self.norm2(output)

        return output

# ----------------------------------------------------------------------------
# 6. Encoder
# ----------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Transformer 的编码器堆栈。
    由 N 个 EncoderLayer 组成。
    对应论文 Section 3.2 "The Transformer - Encoder and Decoder Stacks"。

    Args:
        num_layers (int): 编码器层的数量。
        d_model (int): 模型的维度。
        num_heads (int): 注意力头的数量。
        d_ff (int): 前馈网络中间层的维度。
        dropout (float): Dropout 概率。
    """
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model) # 论文中最后有一个 LayerNorm

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)。
            mask (Optional[torch.Tensor]): 自注意力掩码。

        Returns:
            torch.Tensor: 编码器输出，形状为 (batch_size, seq_len, d_model)。
        """
        # TODO: 实现编码器堆栈的前向传播
        # 遍历所有 EncoderLayer，并将前一层的输出作为下一层的输入
        # 最后应用 LayerNorm
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# ----------------------------------------------------------------------------
# 7. Decoder Layer
# ----------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """
    Transformer 的一个解码器层。
    包含 Masked Multi-Head Self-Attention, Multi-Head Encoder-Decoder Attention,
    和 Position-wise Feed-Forward Network。
    对应论文 Section 3.2 "The Transformer - Encoder and Decoder Stacks"。

    Args:
        d_model (int): 模型的维度。
        num_heads (int): 注意力头的数量。
        d_ff (int): 前馈网络中间层的维度。
        dropout (float): Dropout 概率。
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.encoder_decoder_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 解码器输入（目标序列的嵌入），形状为 (batch_size, seq_len_tgt, d_model)。
            encoder_output (torch.Tensor): 编码器的输出，形状为 (batch_size, seq_len_src, d_model)。
            src_mask (Optional[torch.Tensor]): 编码器自注意力掩码。
            tgt_mask (Optional[torch.Tensor]): 解码器自注意力掩码（用于防止看向未来）。

        Returns:
            torch.Tensor: 解码器层输出，形状为 (batch_size, seq_len_tgt, d_model)。
        """
        # TODO: 实现解码器层的前向传播
        # 1. Masked Multi-Head Self-Attention
        #    - 应用残差连接和层归一化
        # 2. Multi-Head Encoder-Decoder Attention
        #    - Query 来自上一步的输出
        #    - Key 和 Value 来自 encoder_output
        #    - 应用残差连接和层归一化
        # 3. Position-wise Feed-Forward Network
        #    - 应用残差连接和层归一化
        # 4. 返回最终输出

        # Masked Self-Attention sub-layer
        masked_attn_output, _ = self.masked_self_attn(x, x, x, tgt_mask)
        x_residual1 = x + self.dropout1(masked_attn_output)
        x_norm1 = self.norm1(x_residual1)

        # Encoder-Decoder Attention sub-layer
        # Query: x_norm1, Key: encoder_output, Value: encoder_output
        enc_dec_attn_output, _ = self.encoder_decoder_attn(x_norm1, encoder_output, encoder_output, src_mask)
        x_residual2 = x_norm1 + self.dropout2(enc_dec_attn_output)
        x_norm2 = self.norm2(x_residual2)

        # Feed-Forward sub-layer
        ff_output = self.feed_forward(x_norm2)
        output = x_norm2 + self.dropout3(ff_output)
        output = self.norm3(output)

        return output

# ----------------------------------------------------------------------------
# 8. Decoder
# ----------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    Transformer 的解码器堆栈。
    由 N 个 DecoderLayer 组成。
    对应论文 Section 3.2 "The Transformer - Encoder and Decoder Stacks"。

    Args:
        num_layers (int): 解码器层的数量。
        d_model (int): 模型的维度。
        num_heads (int): 注意力头的数量。
        d_ff (int): 前馈网络中间层的维度。
        dropout (float): Dropout 概率。
    """
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model) # 论文中最后有一个 LayerNorm

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 解码器输入（目标序列的嵌入），形状为 (batch_size, seq_len_tgt, d_model)。
            encoder_output (torch.Tensor): 编码器的输出，形状为 (batch_size, seq_len_src, d_model)。
            src_mask (Optional[torch.Tensor]): 编码器自注意力掩码。
            tgt_mask (Optional[torch.Tensor]): 解码器自注意力掩码。

        Returns:
            torch.Tensor: 解码器输出，形状为 (batch_size, seq_len_tgt, d_model)。
        """
        # TODO: 实现解码器堆栈的前向传播
        # 遍历所有 DecoderLayer，并将前一层的输出和 encoder_output 作为下一层的输入
        # 最后应用 LayerNorm
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

# ----------------------------------------------------------------------------
# 9. Transformer Model
# ----------------------------------------------------------------------------

class Transformer(nn.Module):
    """
    完整的 Transformer 模型。
    包含编码器、解码器、嵌入层和输出层。
    对应论文 Section 3 "Attention Is All You Need"。

    Args:
        src_vocab_size (int): 源语言词汇表大小。
        tgt_vocab_size (int): 目标语言词汇表大小。
        d_model (int): 模型的维度。
        num_heads (int): 注意力头的数量。
        d_ff (int): 前馈网络中间层的维度。
        num_encoder_layers (int): 编码器层的数量。
        num_decoder_layers (int): 解码器层的数量。
        dropout (float): Dropout 概率。
        max_seq_len (int): 最大序列长度，用于位置编码。
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float = 0.1,
        max_seq_len: int = 5000
    ):
        super().__init__()

        self.d_model = d_model

        # 源语言嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        # 目标语言嵌入层
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_len)

        # 编码器堆栈
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)

        # 解码器堆栈
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)

        # 输出线性层
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None # 对应论文中的 src_padding_mask
    ) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): 源序列输入，形状为 (batch_size, seq_len_src)。
            tgt (torch.Tensor): 目标序列输入（训练时为移位后的目标序列），形状为 (batch_size, seq_len_tgt)。
            src_mask (Optional[torch.Tensor]): 源序列的掩码（例如，padding mask）。
            tgt_mask (Optional[torch.Tensor]): 目标序列的掩码（例如，padding mask 和 look-ahead mask）。
            memory_mask (Optional[torch.Tensor]): 编码器输出的掩码（通常与 src_mask 相同）。

        Returns:
            torch.Tensor: 模型输出的 logits，形状为 (batch_size, seq_len_tgt, tgt_vocab_size)。
        """
        # TODO: 实现 Transformer 模型的前向传播
        # 1. 源序列和目标序列的嵌入
        # 2. 加入位置编码
        # 3. 编码器处理源序列
        # 4. 解码器处理目标序列，并使用编码器的输出
        # 5. 最终的线性层输出 logits

        # 1. 嵌入层
        # src_emb 形状: (batch_size, seq_len_src, d_model)
        src_emb = self.src_embedding(src)
        # tgt_emb 形状: (batch_size, seq_len_tgt, d_model)
        tgt_emb = self.tgt_embedding(tgt)

        # 2. 加入位置编码
        # 注意：PyTorch 的 Transformer 模型通常期望输入形状为 (seq_len, batch_size, d_model)
        # 我们需要调整这里的维度以匹配 PositionalEncoding 的预期
        # src_pos_encoded 形状: (seq_len_src, batch_size, d_model)
        src_pos_encoded = self.positional_encoding(src_emb.permute(1, 0, 2))
        # tgt_pos_encoded 形状: (seq_len_tgt, batch_size, d_model)
        tgt_pos_encoded = self.positional_encoding(tgt_emb.permute(1, 0, 2))

        # 3. 编码器
        # encoder_output 形状: (seq_len_src, batch_size, d_model)
        encoder_output = self.encoder(src_pos_encoded, src_mask)

        # 4. 解码器
        # decoder_output 形状: (seq_len_tgt, batch_size, d_model)
        decoder_output = self.decoder(tgt_pos_encoded, encoder_output, memory_mask, tgt_mask)

        # 5. 输出层
        # 将维度调整回 (batch_size, seq_len_tgt, d_model) 以便应用线性层
        # decoder_output_permuted 形状: (batch_size, seq_len_tgt, d_model)
        decoder_output_permuted = decoder_output.permute(1, 0, 2)
        # logits 形状: (batch_size, seq_len_tgt, tgt_vocab_size)
        logits = self.output_linear(decoder_output_permuted)

        return logits

    def create_masks(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        pad_idx: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        创建 Transformer 模型所需的各种掩码。

        Args:
            src (torch.Tensor): 源序列，形状为 (batch_size, seq_len_src)。
            tgt (torch.Tensor): 目标序列，形状为 (batch_size, seq_len_tgt)。
            pad_idx (int): 填充符的索引。

        Returns:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
                - src_mask: 源序列的 padding mask。
                - tgt_mask: 目标序列的 padding mask 和 look-ahead mask。
                - memory_mask: 编码器输出的掩码 (通常与 src_mask 相同)。
        """
        # TODO: 实现掩码的创建逻辑
        # 1. 源序列 padding mask: 标记源序列中的 padding 元素
        # 2. 目标序列 padding mask: 标记目标序列中的 padding 元素
        # 3. Look-ahead mask: 确保解码器在预测当前位置时只能看到之前的位置
        # 4. 组合 tgt_mask (padding mask 和 look-ahead mask)
        # 5. memory_mask 通常与 src_mask 相同

        seq_len_src = src.size(1)
        seq_len_tgt = tgt.size(1)

        # Source Padding Mask
        # src_pad_mask 形状: (batch_size, seq_len_src)
        src_pad_mask = (src != pad_idx)
        # 扩展维度以匹配 attention 的计算 (batch_size, 1, 1, seq_len_src)
        src_mask = src_pad_mask.unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_len_src)

        # Target Padding Mask
        # tgt_pad_mask 形状: (batch_size, seq_len_tgt)
        tgt_pad_mask = (tgt != pad_idx)
        # 扩展维度以匹配 attention 的计算 (batch_size, 1, seq_len_tgt, 1)
        tgt_mask_pad = tgt_pad_mask.unsqueeze(1).unsqueeze(2) # (batch_size, 1, seq_len_tgt, 1)

        # Look-ahead Mask
        # 创建一个上三角矩阵，对角线及以下为 True (需要被 mask)
        # look_ahead_mask 形状: (seq_len_tgt, seq_len_tgt)
        look_ahead_mask = torch.triu(torch.ones((seq_len_tgt, seq_len_tgt), device=tgt.device), diagonal=1)
        # 扩展维度以匹配 attention 的计算 (1, 1, seq_len_tgt, seq_len_tgt)
        look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len_tgt, seq_len_tgt)

        # 组合 Target Mask
        # tgt_mask 形状: (batch_size, 1, seq_len_tgt, seq_len_tgt)
        # 使用 & 进行逻辑与操作，确保 padding 和 look-ahead 都被考虑
        tgt_mask = tgt_mask_pad & look_ahead_mask

        # Memory Mask (for encoder-decoder attention)
        # 通常与 src_mask 相同，用于屏蔽编码器的 padding
        memory_mask = src_mask

        return src_mask, tgt_mask, memory_mask

# ----------------------------------------------------------------------------
# 示例用法 (仅为演示，实际训练需要数据加载器和优化器)
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # 模型参数示例
    SRC_VOCAB_SIZE = 10000
    TGT_VOCAB_SIZE = 10000
    D_MODEL = 512
    NUM_HEADS = 8
    D_FF = 2048
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    DROPOUT = 0.1
    MAX_SEQ_LEN = 100
    PAD_IDX = 0 # 假设 padding 索引为 0

    # 实例化 Transformer 模型
    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN
    )

    # 示例输入数据
    batch_size = 32
    src_seq_len = 50
    tgt_seq_len = 60

    # 随机生成输入张量
    # 注意：实际应用中，这些应该是 token IDs
    src_data = torch.randint(1, SRC_VOCAB_SIZE, (batch_size, src_seq_len)) # 避免 PAD_IDX
    tgt_data = torch.randint(1, TGT_VOCAB_SIZE, (batch_size, tgt_seq_len)) # 避免 PAD_IDX

    # 添加一些 padding
    src_data[0, 40:] = PAD_IDX
    tgt_data[1, 50:] = PAD_IDX

    print(f"Source data shape: {src_data.shape}")
    print(f"Target data shape: {tgt_data.shape}")

    # 创建掩码
    src_mask,