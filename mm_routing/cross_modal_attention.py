
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from tqdm import tqdm
import os


class CrossModalAttention(nn.Module):
  
  def __init__(self, dim, num_heads=8, dropout=0.1):
    super().__init__()
    self.num_heads = num_heads
    self.dim = dim
    self.head_dim = dim // num_heads
    
    assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
    
    self.q_proj = nn.Linear(dim, dim)
    self.k_proj = nn.Linear(dim, dim)
    self.v_proj = nn.Linear(dim, dim)
    
    self.out_proj = nn.Linear(dim, dim)
    self.dropout = nn.Dropout(dropout)
      
  def forward(self, query, key, value, mask=None):
    """
    Args:
      query: [batch_size, query_len, dim]
      key: [batch_size, key_len, dim]
      value: [batch_size, value_len, dim]
    """
    batch_size = query.shape[0]
    
    Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = self.dropout(attn_weights)
    
    attn_output = torch.matmul(attn_weights, V)
    
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
    output = self.out_proj(attn_output)
    
    return output, attn_weights


class HatefulMemeClassifier(nn.Module):
  """Cross-modal classifier for hateful meme detection."""
  
  def __init__(self, 
    image_dim=768, 
    text_dim=768, 
    hidden_dim=512,
    num_heads=8,
    num_layers=2,
    dropout=0.1):
    super().__init__()
    
    # Project embeddings to hidden dimension
    self.image_proj = nn.Linear(image_dim, hidden_dim)
    self.text_proj = nn.Linear(text_dim, hidden_dim)
    
    # Layer normalization
    self.image_norm = nn.LayerNorm(hidden_dim)
    self.text_norm = nn.LayerNorm(hidden_dim)
    
    # Cross-modal attention layers
    self.img2text_attention = nn.ModuleList([
      CrossModalAttention(hidden_dim, num_heads, dropout) 
      for _ in range(num_layers)
    ])
    self.text2img_attention = nn.ModuleList([
      CrossModalAttention(hidden_dim, num_heads, dropout) 
      for _ in range(num_layers)
    ])
    
    # Feed-forward networks
    self.img_ffn = nn.ModuleList([
      nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim * 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.Dropout(dropout)
      ) for _ in range(num_layers)
    ])
    self.text_ffn = nn.ModuleList([
      nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim * 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.Dropout(dropout)
      ) for _ in range(num_layers)
    ])
    
    # Layer norms for residual connections
    self.img_norm_layers = nn.ModuleList([
      nn.LayerNorm(hidden_dim) for _ in range(num_layers * 2)
    ])
    self.text_norm_layers = nn.ModuleList([
      nn.LayerNorm(hidden_dim) for _ in range(num_layers * 2)
    ])
    
    # Fusion and classification head
    self.fusion = nn.Sequential(
      nn.Linear(hidden_dim * 2, hidden_dim),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim, hidden_dim // 2),
      nn.GELU(),
      nn.Dropout(dropout)
    )
    
    #self.classifier = nn.Linear(hidden_dim // 2, 1)
    self.classifier = nn.Identity()
      
  def forward(self, image_embeds, text_embeds):
    
    img_h = self.image_proj(image_embeds).unsqueeze(1)  # [batch, 1, hidden]
    text_h = self.text_proj(text_embeds).unsqueeze(1)   # [batch, 1, hidden]
    
    img_h = self.image_norm(img_h)
    text_h = self.text_norm(text_h)
    
    for i in range(1):
    #for i in range(len(self.img2text_attention)):
      # Store original features before cross-attention
      img_h_orig = img_h
      text_h_orig = text_h
      
      # Both attentions use the same input features
      img_attn, _ = self.img2text_attention[i](img_h_orig, text_h_orig, text_h_orig)
      text_attn, _ = self.text2img_attention[i](text_h_orig, img_h_orig, img_h_orig)
      
      # Apply both attention outputs
      img_h = self.img_norm_layers[i * 2](img_h_orig + img_attn)
      text_h = self.text_norm_layers[i * 2](text_h_orig + text_attn)
      
      # Feed-forward for image
      img_ffn_out = self.img_ffn[i](img_h)
      img_h = self.img_norm_layers[i * 2 + 1](img_h + img_ffn_out)
      
      # Feed-forward for text
      text_ffn_out = self.text_ffn[i](text_h)
      text_h = self.text_norm_layers[i * 2 + 1](text_h + text_ffn_out) 
      
    # Squeeze and concatenate
    img_h = img_h.squeeze(1)  # [batch, hidden]
    text_h = text_h.squeeze(1)  # [batch, hidden]
    
    # Fusion
    fused = torch.cat([img_h, text_h], dim=-1)  # [batch, hidden*2]
    #fused = self.fusion(fused)  # [batch, hidden//2]
    
    # Classification
    logits = self.classifier(fused)  # [batch, 1]
    
    return logits
