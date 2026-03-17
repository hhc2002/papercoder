# PaperCoder — arxiv:1706.03762

# pip install torch
import torch
import torch.nn.functional as F
from typing import Tuple

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    实现缩放点积注意力机制。

    该函数实现了论文 "Attention Is All You Need" (arXiv:1706.03762) 中
    第 3.2.1 节描述的缩放点积注意力机制。

    Args:
        query (torch.Tensor): 查询张量，形状为 (batch_size, seq_len_q, d_k)。
        key (torch.Tensor): 键张量，形状为 (batch_size, seq_len_k, d_k)。
        value (torch.Tensor): 值张量，形状为 (batch_size, seq_len_v, d_v)。
                           通常 seq_len_k == seq_len_v。
        mask (torch.Tensor | None, optional): 掩码张量，用于屏蔽某些位置。
                                            形状通常为 (batch_size, 1, seq_len_k) 或
                                            (batch_size, seq_len_q, seq_len_k)。
                                            Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - attention_output (torch.Tensor): 注意力机制的输出，形状为 (batch_size, seq_len_q, d_v)。
            - attention_weights (torch.Tensor): 计算得到的注意力权重，形状为 (batch_size, seq_len_q, seq_len_k)。
    """
    # 1. 计算 Q 和 K 的点积
    # 形状: (batch_size, seq_len_q, seq_len_k)
    matmul_qk = torch.matmul(query, key.transpose(-2, -1))

    # 2. 缩放点积
    # d_k 是键向量的维度
    d_k = key.size(-1)
    scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # 3. 应用掩码 (如果存在)
    if mask is not None:
        # 将掩码为 False 的位置设置为一个非常小的负数，以便 softmax 后变为接近 0
        scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 0, float('-1e9'))

    # 4. 应用 softmax 函数计算注意力权重
    # 形状: (batch_size, seq_len_q, seq_len_k)
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)

    # 5. 将注意力权重应用于 V
    # 形状: (batch_size, seq_len_q, d_v)
    attention_output = torch.matmul(attention_weights, value)

    return attention_output, attention_weights

if __name__ == '__main__':
    # 示例用法
    batch_size = 2
    seq_len_q = 4
    seq_len_k = 5
    seq_len_v = 5
    d_k = 32
    d_v = 64

    # 创建示例输入张量
    query = torch.randn(batch_size, seq_len_q, d_k)
    key = torch.randn(batch_size, seq_len_k, d_k)
    value = torch.randn(batch_size, seq_len_v, d_v)

    # 创建一个简单的掩码示例 (例如，用于填充)
    # 假设 seq_len_k 的最后两个位置是填充的，不应被关注
    mask = torch.ones(batch_size, 1, seq_len_k, dtype=torch.bool)
    mask[:, :, -2:] = 0 # 将最后两个位置设置为 False

    # 计算注意力
    output, weights = scaled_dot_product_attention(query, key, value, mask=mask)

    print("Query shape:", query.shape)
    print("Key shape:", key.shape)
    print("Value shape:", value.shape)
    print("Mask shape:", mask.shape)
    print("Attention Output shape:", output.shape)
    print("Attention Weights shape:", weights.shape)

    # 打印一些注意力权重示例
    print("\nExample Attention Weights (first batch, first query token):")
    print(weights[0, 0, :])

    # 验证掩码是否生效 (权重应接近 0)
    print("\nAttention weights for masked positions (should be close to 0):")
    print(weights[0, 0, -2:])