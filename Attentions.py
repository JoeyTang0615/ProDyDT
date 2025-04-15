import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, qkv_bias=True):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        # Query, Key, Value projections with optional bias
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.scale = self.head_dim ** -0.5

    def forward(self, q, k, v):
        batch_size, l, _ = q.size()

        # Project inputs with optional bias
        q = self.q_proj(q)  # Shape: [batch, l, hidden_size]
        k = self.k_proj(k)  # Shape: [1, l, hidden_size]
        v = self.v_proj(v)  # Shape: [1, l, hidden_size]

        # Reshape for multi-head attention
        q = q.view(batch_size, l, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: [batch, num_heads, l, head_dim]
        k = k.view(1, l, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: [1, num_heads, l, head_dim]
        v = v.view(1, l, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: [1, num_heads, l, head_dim]

        # Expand key and value to match batch size of query
        k = k.expand(batch_size, -1, -1, -1)  # Shape: [batch, num_heads, l, head_dim]
        v = v.expand(batch_size, -1, -1, -1)  # Shape: [batch, num_heads, l, head_dim]

        # Compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # Shape: [batch, num_heads, l, l]
        attn_probs = F.softmax(attn_scores, dim=-1)  # Shape: [batch, num_heads, l, l]

        # Weighted sum of values
        attn_output = (attn_probs @ v)  # Shape: [batch, num_heads, l, head_dim]

        # Concatenate heads and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, l, self.hidden_size)  # Shape: [batch, l, hidden_size]
        output = self.out_proj(attn_output)  # Shape: [batch, l, hidden_size]

        return output
