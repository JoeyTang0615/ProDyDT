import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from models.Attentions import CrossAttention

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    

class ESMEmbedder(nn.Module):
    """
    Embeds protein ESM features into vector representations. 
    Also handles feature dropout for classifier-free guidance in diffusion training.
    """
    def __init__(self, input_size, hidden_size, dropout_prob):
        """
        Args:
            input_size (int): The dimensionality of the input ESM features (e.g., 1280 from ESM model).
            hidden_size (int): The dimensionality of the output embeddings.
            dropout_prob (float): Dropout probability to apply to the input features for classifier-free guidance.
        """
        super().__init__()
        self.embedding_linear = nn.Linear(input_size, hidden_size)  # Linear projection for ESM features
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

    def feature_drop(self, esm_features, force_drop_ids=None):
        """
        Drops ESM features to enable classifier-free guidance.
        Args:
            esm_features (Tensor): The input ESM features with shape [batch_size, input_size].
            force_drop_ids (Tensor or None): Optional tensor of shape [batch_size] indicating which
                                             features to forcefully drop (for CFG).
        Returns:
            Tensor: ESM features after applying dropout to some features.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(esm_features.shape[0], device=esm_features.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        
        # For dropped features, we can zero out or replace them with some default vector
        dropped_features = torch.zeros_like(esm_features)
        
        # Use torch.where to conditionally replace features
        esm_features = torch.where(drop_ids.unsqueeze(1), dropped_features, esm_features)
        return esm_features

    def forward(self, esm_features, train=True, force_drop_ids=None):
        """
        Args:
            esm_features (Tensor): The input ESM features with shape [batch_size, input_size].
            train (bool): If True, applies dropout during training.
            force_drop_ids (Tensor or None): Optional tensor to manually specify which samples to drop.
        Returns:
            Tensor: Embedded ESM features with shape [batch_size, hidden_size].
        """
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            esm_features = self.feature_drop(esm_features, force_drop_ids)
        
        # Linear projection from ESM features to hidden space
        embeddings = self.embedding_linear(esm_features)
        
        return embeddings



class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.crossattn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.adaLN_modulation_guide = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
    def forward(self, x, c, guide):
        # print(self.adaLN_modulation(c).shape)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        shift_guide, scale_guide, gate_guide = self.adaLN_modulation_guide(guide).chunk(3,dim = 2)

        x = x + gate_guide * self.crossattn(self.norm1(x), shift_guide, scale_guide)
        # print("modulate")
        # print(modulate(self.norm1(x), shift_msa, scale_msa).shape)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm2(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x
    

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, protein_length):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.output_layer = nn.Linear(hidden_size, 3, bias=True)  # Outputs 3D coordinates
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.protein_length = protein_length

    def forward(self, x, c):
        # adaLN conditioning
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)

        # Output coordinate predictions
        coord_output = self.output_layer(x)  # [batch_size, protein_length, 3]

        return coord_output


class DiT(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        depth=6,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        esm_feature_size=1280,
        max_protein_length=150,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.hidden_size = hidden_size
        
        # Embedding layer for 3D coordinates
        self.coord_embedder = nn.Linear(3, hidden_size)
        self.guide_coord_embedder1 = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        )
        self.guide_coord_embedder2 = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        )
        # Timestep embedder for diffusion steps
        self.t_embedder = TimestepEmbedder(hidden_size)

        # ESM Embedder to process the protein sequence features
        self.esm_embedder = ESMEmbedder(esm_feature_size, hidden_size, class_dropout_prob)

        # Position embedding (fixed sin-cos embeddings)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_protein_length, hidden_size), requires_grad=False)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        # Final output layer to predict coordinates
        self.final_layer = FinalLayer(hidden_size, max_protein_length)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize position embeddings
        position = torch.arange(self.pos_embed.shape[1]).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_size, 2).float() * (-math.log(10000.0) / self.hidden_size))
        pos_encoding = torch.zeros(1, self.pos_embed.shape[1], self.hidden_size)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        self.pos_embed.data.copy_(pos_encoding)

    def forward(self, coordinates, esm_features, t, guide_info1 = None, guide_info2 = None,name = None):
        num_conformations, protein_length, _ = coordinates.shape

        # Embed coordinates
        x = self.coord_embedder(coordinates)  # [num_conformations, protein_length, hidden_size]
        # Project guide info to hidden size
        guide_emb1 = self.guide_coord_embedder1(guide_info1)  
        guide_emb2 = self.guide_coord_embedder2(guide_info2) 
        guide_emb = guide_emb1 + guide_emb2
        # Add position embedding
        x = x + self.pos_embed[:, :protein_length]

        # Process timestep embedding
        t_emb = self.t_embedder(t)  # [num_conformations, hidden_size]

        # Embed ESM features
        esm_emb = self.esm_embedder(esm_features)  # [num_conformations, hidden_size]

        # Combine timestep and ESM embeddings (conditioning)
        cond_emb = t_emb + esm_emb  # [num_conformations, hidden_size]

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x, cond_emb, guide_emb)  # [num_conformations, protein_length, hidden_size]

        # Final layer to predict coordinates
        coord_output = self.final_layer(x, cond_emb)  # Outputs coordinates

        return coord_output

    
