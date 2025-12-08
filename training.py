#!/usr/bin/env python3

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
    
    self.classifier = nn.Linear(hidden_dim // 2, 1)
      
  def forward(self, image_embeds, text_embeds):
    
    img_h = self.image_proj(image_embeds).unsqueeze(1)  # [batch, 1, hidden]
    text_h = self.text_proj(text_embeds).unsqueeze(1)   # [batch, 1, hidden]
    
    img_h = self.image_norm(img_h)
    text_h = self.text_norm(text_h)
    
    for i in range(len(self.img2text_attention)):
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
    fused = self.fusion(fused)  # [batch, hidden//2]
    
    # Classification
    logits = self.classifier(fused)  # [batch, 1]
    
    return logits


class HatefulMemeDataset(Dataset):
  """Dataset for hateful meme classification."""
    
  def __init__(self, image_embeds, text_embeds, labels):
    self.image_embeds = torch.FloatTensor(image_embeds)
    self.text_embeds = torch.FloatTensor(text_embeds)
    self.labels = torch.FloatTensor(labels)
      
  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    return {
      'image': self.image_embeds[idx],
      'text': self.text_embeds[idx],
      'label': self.labels[idx]
    }


def train_epoch(model, dataloader, optimizer, device, criterion):
  """Train for one epoch."""
  model.train()
  total_loss = 0
  all_preds = []
  all_labels = []
  
  for batch in tqdm(dataloader, desc="Training"):
    image_embeds = batch['image'].to(device)
    text_embeds = batch['text'].to(device)
    labels = batch['label'].to(device)
    
    optimizer.zero_grad()
    
    logits = model(image_embeds, text_embeds).squeeze(-1)
    loss = criterion(logits, labels)
    
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item()
    
    # Collect predictions
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    all_preds.extend(probs)
    all_labels.extend(labels.cpu().numpy())
  
  avg_loss = total_loss / len(dataloader)
  auc = roc_auc_score(all_labels, all_preds)
  preds_binary = (np.array(all_preds) > 0.5).astype(int)
  acc = accuracy_score(all_labels, preds_binary)
  
  return avg_loss, auc, acc


def evaluate(model, dataloader, device, criterion):
  """Evaluate the model."""
  model.eval()
  total_loss = 0
  all_preds = []
  all_labels = []
  
  with torch.no_grad():
    for batch in tqdm(dataloader, desc="Evaluating"):
      image_embeds = batch['image'].to(device)
      text_embeds = batch['text'].to(device)
      labels = batch['label'].to(device)
      
      logits = model(image_embeds, text_embeds).squeeze(-1)
      loss = criterion(logits, labels)
      
      total_loss += loss.item()
      
      # Collect predictions
      probs = torch.sigmoid(logits).cpu().numpy()
      all_preds.extend(probs)
      all_labels.extend(labels.cpu().numpy())
  
  avg_loss = total_loss / len(dataloader)
  auc = roc_auc_score(all_labels, all_preds)
  preds_binary = (np.array(all_preds) > 0.5).astype(int)
  acc = accuracy_score(all_labels, preds_binary)
  
  return avg_loss, auc, acc, all_preds, all_labels


def main():
  # Configuration
  config = {
    'hidden_dim': 512,
    'num_heads': 16,
    'num_layers': 2,
    'dropout': 0.3,
    'batch_size': 64,
    'learning_rate': 1e-4,
    'num_epochs': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'random_seed': 42
  }
  
  print("Configuration:")
  for k, v in config.items():
    print(f"  {k}: {v}")
  
  # Set random seed
  torch.manual_seed(config['random_seed'])
  np.random.seed(config['random_seed'])
  
  # Load embeddings
  print("\nLoading embeddings...")
  train_data = np.load('hateful_memes_clip_embeddings_train.npz')
  train_image_embeds = train_data['image_embeddings']
  train_text_embeds = train_data['text_concat_embeddings']
  train_labels = train_data['labels']
  
  val_data = np.load('hateful_memes_clip_embeddings_val.npz')
  val_image_embeds = val_data['image_embeddings']
  val_text_embeds = val_data['text_concat_embeddings']
  val_labels = val_data['labels']

  # Store input dimensions in config for checkpoint/reloading
  config['image_dim'] = train_image_embeds.shape[1]
  config['text_dim'] = train_text_embeds.shape[1]

  print(f"Loaded {len(train_labels)} training samples")
  print(f"  Train Image embeddings shape: {train_image_embeds.shape}")
  print(f"  Train Text embeddings shape: {train_text_embeds.shape}")
  print(f"  Train Positive samples: {train_labels.sum()} ({train_labels.mean()*100:.2f}%)")
  
  print(f"Loaded {len(val_labels)} validation samples")
  print(f"  Validation Image embeddings shape: {val_image_embeds.shape}")
  print(f"  Validation Text embeddings shape: {val_text_embeds.shape}")
  print(f"  Validation Positive samples: {val_labels.sum()} ({val_labels.mean()*100:.2f}%)")
  # Train/val split
  # indices = np.arange(len(labels))
  # train_idx, val_idx = train_test_split(
  #   indices, 
  #   test_size=config['val_split'], 
  #   random_state=config['random_seed'],
  #   stratify=labels
  # )
  
  # print(f"\nTrain samples: {len(train_idx)}")
  # print(f"Val samples: {len(val_idx)}")
  
  # Create datasets
  train_dataset = HatefulMemeDataset(
    train_image_embeds, 
    train_text_embeds, 
    train_labels
  )
  val_dataset = HatefulMemeDataset(
    val_image_embeds, 
    val_text_embeds, 
    val_labels
  )
  
  # Create dataloaders
  train_loader = DataLoader(
    train_dataset, 
    batch_size=config['batch_size'], 
    shuffle=True,
    num_workers=4
  )
  val_loader = DataLoader(
    val_dataset, 
    batch_size=config['batch_size'], 
    shuffle=False,
    num_workers=4
  )
  
  # Create model
  print("\nInitializing model...")
  model = HatefulMemeClassifier(
    image_dim=config['image_dim'],
    text_dim=config['text_dim'],
    hidden_dim=config['hidden_dim'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers'],
    dropout=config['dropout']
  )
  model = model.to(config['device'])
  
  print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
  
  # Loss and optimizer
  criterion = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
  )
  
  # Training loop
  print("\nStarting training...")
  best_val_auc = 0
  
  for epoch in range(config['num_epochs']):
    print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
    
    train_loss, train_auc, train_acc = train_epoch(
      model, train_loader, optimizer, config['device'], criterion
    )
    
    val_loss, val_auc, val_acc, _, _ = evaluate(
      model, val_loader, config['device'], criterion
    )
    
    print(f"Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
    print(f"Val   - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")
    
    # Learning rate scheduling
    scheduler.step(val_auc)
    
    # Save best model
    if val_auc > best_val_auc:
      best_val_auc = val_auc
      torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_auc': val_auc,
        'config': config
      }, 'best_model.pth')
      print(f"✓ Saved best model (AUC: {val_auc:.4f})")
  
  # Load best model and evaluate
  print("\n" + "="*70)
  print("Final evaluation with best model...")
  checkpoint = torch.load('best_model.pth', weights_only=False)
  model.load_state_dict(checkpoint['model_state_dict'])
  
  val_loss, val_auc, val_acc, val_preds, val_labels = evaluate(
    model, val_loader, config['device'], criterion
  )
  
  print(f"\nBest Val AUC: {val_auc:.4f}")
  print(f"Best Val Acc: {val_acc:.4f}")
  
  # Classification report
  val_preds_binary = (np.array(val_preds) > 0.5).astype(int)
  print("\nClassification Report:")
  print(classification_report(val_labels, val_preds_binary, 
                              target_names=['Not Hateful', 'Hateful']))
  
  print("\nModel saved to 'best_model.pth'")


if __name__ == '__main__':
  main()
