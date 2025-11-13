import torch.nn as nn
import torch


class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim, ffn_dim=2048, dropout=0.1):
        super(PositionalWiseFeedForward, self).__init__()
        # pointwise expansion and contraction layers
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(nn.functional.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))
        output = self.layer_norm(x + output)
        return output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=True, attn_mask=None):
        # standard scaled dot product attention
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            scale = (k.size(2)) ** -0.5
            attention = attention * scale
        if attn_mask:
            attention = attention.masked_fill_(attn_mask, float("-inf"))
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, model_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        # projections prepare packed heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        residual = query
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split heads for parallel attention
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)

        context, attention = self.dot_product_attention(query, key, value, attn_mask=attn_mask)

        context = context.view(batch_size, -1, dim_per_head * num_heads)
        attention = attention.view(batch_size, num_heads, query.size(1), key.size(1))
        attention = attention.sum(dim=1) / num_heads

        output = self.linear_final(context)
        output = self.dropout(output)
        # add residual with layer norm
        output = self.layer_norm(residual + output)

        return output, attention


class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ffn_dim=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(num_heads, model_dim, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, q, k, v, attn_mask=None):
        context, attention = self.attention(q, k, v, attn_mask=attn_mask)
        output = self.feed_forward(context)

        return output, attention


class Encoder(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, ffn_dim=2048, dropout=0.1):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
          [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
           range(num_layers)])

    def forward(self, q, k, v, attn_mask=None):
        output = q
        attention = None
        for encoder in self.encoder_layers:
            # apply each encoder layer in sequence
            output, attention = encoder(output, k, v, attn_mask=attn_mask)

        return output, attention
