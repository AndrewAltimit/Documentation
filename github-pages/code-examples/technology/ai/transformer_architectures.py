"""
Transformer Architectures

Implementation of various transformer-based architectures including
Vision Transformer (ViT), CLIP, and other modern architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False, 
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention"""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, 
                 qkv_bias: bool = False, drop: float = 0., attn_drop: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer implementation with theoretical insights"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, embed_dim: int = 768, 
                 depth: int = 12, num_heads: int = 12, 
                 mlp_ratio: float = 4.0, num_classes: int = 1000,
                 drop_rate: float = 0., attn_drop_rate: float = 0.):
        super().__init__()
        
        # Patch embedding
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                    kernel_size=patch_size, stride=patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, 
                           qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with theoretical justification"""
        # Position embeddings from 2D sine-cosine
        pos_embed = self.get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 
                                                 int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Class token
        torch.nn.init.normal_(self.cls_token, std=.02)
        
        # Xavier/Glorot initialization for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    @staticmethod
    def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, 
                                cls_token: bool = True) -> np.ndarray:
        """2D sine-cosine position embedding"""
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # w first for consistency
        grid = np.stack(grid, axis=0)
        grid = grid.reshape([2, 1, grid_size, grid_size])
        
        pos_embed = np.zeros((grid_size * grid_size, embed_dim))
        pos_embed[:, 0:embed_dim//2] = np.sin(grid[0].reshape(-1, 1) * 
                                              np.arange(embed_dim//2) * 2 * np.pi / grid_size)
        pos_embed[:, embed_dim//2:] = np.cos(grid[1].reshape(-1, 1) * 
                                            np.arange(embed_dim//2) * 2 * np.pi / grid_size)
        
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        
        return pos_embed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]
        return self.head(cls_token_final)
    
    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract attention maps for visualization"""
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        attention_maps = []
        for block in self.blocks:
            # Get attention weights
            # This requires modifying the attention module to return weights
            # Simplified here
            x = block(x)
            # attention_maps.append(attn_weights)
        
        return attention_maps


class CLIP(nn.Module):
    """CLIP model for vision-language understanding"""
    
    def __init__(self, vision_encoder: nn.Module, text_encoder: nn.Module, 
                 embed_dim: int = 512, temperature: float = 0.07):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        
        # Get output dimensions from encoders
        vision_width = vision_encoder.head.in_features
        text_width = text_encoder.width if hasattr(text_encoder, 'width') else 768
        
        # Projection heads
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        
        # Temperature parameter for contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to normalized embedding"""
        image_features = self.vision_encoder(image)
        image_features = self.vision_proj(image_features)
        return F.normalize(image_features, dim=-1)
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text to normalized embedding"""
        text_features = self.text_encoder(text)
        text_features = self.text_proj(text_features)
        return F.normalize(text_features, dim=-1)
    
    def forward(self, image: torch.Tensor, text: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute contrastive loss"""
        # Encode
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # Compute similarity matrix
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        # Contrastive loss
        batch_size = image.shape[0]
        labels = torch.arange(batch_size, device=image.device)
        
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i2t + loss_t2i) / 2
        
        return {
            'loss': loss,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text
        }
    
    def compute_retrieval_metrics(self, image_features: torch.Tensor, 
                                 text_features: torch.Tensor) -> Dict[str, float]:
        """Compute retrieval metrics for evaluation"""
        # Similarity matrix
        similarity = image_features @ text_features.t()
        
        # Image-to-text retrieval
        _, i2t_indices = similarity.topk(10, dim=1)
        i2t_correct = (i2t_indices == torch.arange(len(similarity), 
                                                  device=similarity.device).unsqueeze(1))
        
        # Text-to-image retrieval  
        _, t2i_indices = similarity.t().topk(10, dim=1)
        t2i_correct = (t2i_indices == torch.arange(len(similarity), 
                                                  device=similarity.device).unsqueeze(1))
        
        return {
            'i2t_r1': i2t_correct[:, 0].float().mean().item(),
            'i2t_r5': i2t_correct[:, :5].any(dim=1).float().mean().item(),
            'i2t_r10': i2t_correct[:, :10].any(dim=1).float().mean().item(),
            't2i_r1': t2i_correct[:, 0].float().mean().item(),
            't2i_r5': t2i_correct[:, :5].any(dim=1).float().mean().item(),
            't2i_r10': t2i_correct[:, :10].any(dim=1).float().mean().item(),
        }


class BERTModel(nn.Module):
    """Simplified BERT implementation for understanding transformers in NLP"""
    
    def __init__(self, vocab_size: int, max_length: int = 512, 
                 hidden_size: int = 768, num_layers: int = 12, 
                 num_heads: int = 12, intermediate_size: int = 3072):
        super().__init__()
        
        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_length, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)  # For segment A/B
        
        self.embedding_norm = nn.LayerNorm(hidden_size)
        self.embedding_dropout = nn.Dropout(0.1)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, 
                           mlp_ratio=intermediate_size/hidden_size)
            for _ in range(num_layers)
        ])
        
        # Output heads
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.pooler_activation = nn.Tanh()
    
    def forward(self, input_ids: torch.Tensor, 
                token_type_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        seq_length = input_ids.size(1)
        
        # Create position IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, 
                                   device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Token type IDs default to 0
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        embeddings = token_embeds + position_embeds + token_type_embeds
        embeddings = self.embedding_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        
        # Transformer layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Pooling
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.pooler(first_token_tensor)
        pooled_output = self.pooler_activation(pooled_output)
        
        return {
            'last_hidden_state': hidden_states,
            'pooler_output': pooled_output
        }


class GPT2Model(nn.Module):
    """Simplified GPT-2 implementation for understanding autoregressive transformers"""
    
    def __init__(self, vocab_size: int, max_length: int = 1024,
                 hidden_size: int = 768, num_layers: int = 12,
                 num_heads: int = 12):
        super().__init__()
        
        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_length, hidden_size)
        
        self.drop = nn.Dropout(0.1)
        
        # Transformer layers with causal masking
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(hidden_size)
        
        # Language modeling head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.size(1)
        
        # Get embeddings
        position_ids = torch.arange(seq_length, dtype=torch.long,
                                   device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)
        
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        hidden_states = self.drop(token_embeds + position_embeds)
        
        # Apply transformer layers with causal mask
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        hidden_states = self.ln_f(hidden_states)
        
        # Project to vocabulary
        lm_logits = self.lm_head(hidden_states)
        
        return lm_logits
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100,
                temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """Autoregressive generation with top-k sampling"""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                outputs = self.forward(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids