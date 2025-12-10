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
import argparse
import matplotlib.pyplot as plt
from datetime import datetime


class BCEWithLabelSmoothing(nn.Module):
  """BCE Loss with label smoothing to prevent overconfidence."""
  
  def __init__(self, smoothing=0.1):
    super().__init__()
    self.smoothing = smoothing
    
  def forward(self, logits, targets):
    # Smooth targets: 0 -> smoothing/2, 1 -> 1 - smoothing/2
    targets = targets * (1 - self.smoothing) + self.smoothing / 2
    return F.binary_cross_entropy_with_logits(logits, targets)


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

class SimpleHatefulMemeClassifier(nn.Module):
  
  def __init__(self, 
    image_dim=768, 
    text_dim=768, 
    hidden_dim=1024,
    dropout=0.1):

    super().__init__()

    self.ffn = nn.Sequential(
      nn.Linear(image_dim + text_dim, hidden_dim * 2),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim * 2, hidden_dim),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim, hidden_dim // 2),
      nn.GELU(),
      nn.Dropout(dropout)
    )

    self.classifier = nn.Linear(hidden_dim // 2, 1)

  def forward(self, image_embeds, text_embeds):

    fused = torch.cat([image_embeds, text_embeds], dim=-1)  # [batch, hidden*2]
    h = self.ffn(fused)  # [batch, hidden//2]
    
    # Classification
    logits = self.classifier(h)  # [batch, 1]
    
    return logits

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


def plot_training_history(history, save_path='training_history.png'):
  """Plot training and validation metrics over epochs."""
  fig, axes = plt.subplots(1, 3, figsize=(15, 4))
  
  epochs = range(1, len(history['train_loss']) + 1)
  
  # Plot Loss
  axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
  axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
  axes[0].set_xlabel('Epoch', fontsize=12)
  axes[0].set_ylabel('Loss', fontsize=12)
  axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
  axes[0].legend(fontsize=10)
  axes[0].grid(True, alpha=0.3)
  
  # Plot AUC
  axes[1].plot(epochs, history['train_auc'], 'b-', label='Train AUC', linewidth=2)
  axes[1].plot(epochs, history['val_auc'], 'r-', label='Val AUC', linewidth=2)
  axes[1].set_xlabel('Epoch', fontsize=12)
  axes[1].set_ylabel('AUC', fontsize=12)
  axes[1].set_ylim(0.0, 1.0)
  axes[1].set_title('Training and Validation AUC', fontsize=14, fontweight='bold')
  axes[1].legend(fontsize=10)
  axes[1].grid(True, alpha=0.3)
  
  # Plot Accuracy
  axes[2].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
  axes[2].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
  axes[2].set_xlabel('Epoch', fontsize=12)
  axes[2].set_ylabel('Accuracy', fontsize=12)
  axes[2].set_ylim(0.0, 1.0)
  axes[2].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
  axes[2].legend(fontsize=10)
  axes[2].grid(True, alpha=0.3)
  
  plt.tight_layout()
  plt.savefig(save_path, dpi=300, bbox_inches='tight')
  print(f"\nTraining history plot saved to '{save_path}'")
  plt.close()


def save_results(output_dir, config, test_auc, test_acc, test_loss, 
                 test_labels, test_preds, best_val_auc, total_epochs):
  """Save training results to a text file."""
  results_path = os.path.join(output_dir, 'results.txt')
  
  test_preds_binary = (np.array(test_preds) > 0.5).astype(int)
  
  with open(results_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("HATEFUL MEME CLASSIFICATION - TRAINING RESULTS\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Output Directory: {output_dir}\n\n")
    
    f.write("-"*70 + "\n")
    f.write("CONFIGURATION\n")
    f.write("-"*70 + "\n")
    for k, v in config.items():
      f.write(f"  {k:30s}: {v}\n")
    
    f.write("\n" + "-"*70 + "\n")
    f.write("TRAINING SUMMARY\n")
    f.write("-"*70 + "\n")
    f.write(f"Total Epochs Trained: {total_epochs}\n")
    f.write(f"Best Validation AUC:  {best_val_auc:.4f}\n\n")
    
    f.write("-"*70 + "\n")
    f.write("TEST SET RESULTS\n")
    f.write("-"*70 + "\n")
    f.write(f"Test AUC:      {test_auc:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Test Loss:     {test_loss:.4f}\n\n")
    
    f.write("-"*70 + "\n")
    f.write("PREDICTION STATISTICS\n")
    f.write("-"*70 + "\n")
    f.write(f"Mean Probability: {np.mean(test_preds):.3f}\n")
    f.write(f"Std Probability:  {np.std(test_preds):.3f}\n")
    f.write(f"Min Probability:  {np.min(test_preds):.3f}\n")
    f.write(f"Max Probability:  {np.max(test_preds):.3f}\n\n")
    
    f.write("-"*70 + "\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("-"*70 + "\n")
    report = classification_report(test_labels, test_preds_binary, 
                                   target_names=['Not Hateful', 'Hateful'])
    f.write(report)
    f.write("\n")
    
    f.write("="*70 + "\n")
  
  print(f"\nResults saved to '{results_path}'")


def main():

  # Configuration
  config = {
    'hidden_dim': 768 * 2,
    'num_heads': 8,
    'num_layers': 2,
    'dropout': 0.4,  
    'batch_size': 64,
    'learning_rate': 1e-4, 
    'weight_decay': 0.01,
    'label_smoothing': 0.1,
    'num_epochs': 50,
    'early_stopping_patience': 20,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'random_seed': 67,
    'simple': False
  }

  parser = argparse.ArgumentParser()
  parser.add_argument('--working_dir', type=str, default="hateful_memes_clip")
  args = parser.parse_args()
  
  # Create timestamped output directory
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  model_type = 'simple' if config['simple'] else 'crossmodal'
  output_dir = os.path.join(args.working_dir, f'run_{timestamp}_{model_type}')
  os.makedirs(output_dir, exist_ok=True)
  
  print("="*70)
  print(f"Output Directory: {output_dir}")
  print("="*70)
  print("\nConfiguration:")
  for k, v in config.items():
    print(f"  {k}: {v}")
  
  # Set random seed
  torch.manual_seed(config['random_seed'])
  np.random.seed(config['random_seed'])
  
  # Load embeddings
  print("\nLoading embeddings...")
  train_data = np.load(os.path.join(args.working_dir, 'embeddings_train.npz'))
  train_image_embeds = train_data['image_embeddings']
<<<<<<< Updated upstream
  train_text_embeds = train_data['text_concat_embeddings']
=======
  train_text_embeds = train_data['text_embeddings'] if "text_concat_embeddings" not in train_data else train_data['text_concat_embeddings'] 
>>>>>>> Stashed changes
  train_labels = train_data['labels']
  
  val_data = np.load(os.path.join(args.working_dir, 'embeddings_val.npz'))
  val_image_embeds = val_data['image_embeddings']
<<<<<<< Updated upstream
  val_text_embeds = val_data['text_concat_embeddings']
=======
  val_text_embeds = val_data['text_embeddings'] if "text_concat_embeddings" not in val_data else val_data['text_concat_embeddings'] 
>>>>>>> Stashed changes
  val_labels = val_data['labels']
  
  test_data = np.load(os.path.join(args.working_dir, 'embeddings_test.npz'))
  test_image_embeds = test_data['image_embeddings']
  test_text_embeds =  test_data['text_embeddings'] if "text_concat_embeddings" not in test_data else test_data['text_concat_embeddings'] 
  test_labels = test_data['labels']

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
  
  print(f"Loaded {len(test_labels)} test samples")
  print(f"  Test Image embeddings shape: {test_image_embeds.shape}")
  print(f"  Test Text embeddings shape: {test_text_embeds.shape}")
  print(f"  Test Positive samples: {test_labels.sum()} ({test_labels.mean()*100:.2f}%)")
  
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
  test_dataset = HatefulMemeDataset(
    test_image_embeds, 
    test_text_embeds, 
    test_labels
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
  test_loader = DataLoader(
    test_dataset, 
    batch_size=config['batch_size'], 
    shuffle=False,
    num_workers=4
  )
  
  # Create model
  print("\nInitializing model...")
<<<<<<< Updated upstream
  model = HatefulMemeClassifier(
    image_dim=config['image_dim'],
    text_dim=config['text_dim'],
    hidden_dim=config['hidden_dim'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers'],
    dropout=config['dropout']
  )
=======

  if config["simple"]:
    model = SimpleHatefulMemeClassifier(
      image_dim=train_image_embeds.shape[1],
      text_dim=train_text_embeds.shape[1],
      hidden_dim=config['hidden_dim'],
      dropout=config['dropout']
    )
    model_name = 'best_model.pth'
  else:
    model = HatefulMemeClassifier(
      image_dim=train_image_embeds.shape[1],
      text_dim=train_text_embeds.shape[1],
      hidden_dim=config['hidden_dim'],
      num_heads=config['num_heads'],
      num_layers=config['num_layers'],
      dropout=config['dropout']
    )
    model_name = 'best_model.pth'
>>>>>>> Stashed changes
  model = model.to(config['device'])
  
  print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
  
  # Loss and optimizer with label smoothing and weight decay
  criterion = BCEWithLabelSmoothing(smoothing=config['label_smoothing'])
  optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=config['learning_rate'],
    weight_decay=config['weight_decay']
  )
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.75, patience=8
  )
  
  # Initialize tracking history
  history = {
    'train_loss': [],
    'train_auc': [],
    'train_acc': [],
    'val_loss': [],
    'val_auc': [],
    'val_acc': []
  }
  
  # Training loop with early stopping
  print("\nStarting training...")
  print("Using label smoothing and early stopping based on validation AUC")
  best_val_auc = 0
  patience_counter = 0
  
  for epoch in range(config['num_epochs']):
    print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
    
    train_loss, train_auc, train_acc = train_epoch(
      model, train_loader, optimizer, config['device'], criterion
    )
    
    val_loss, val_auc, val_acc, _, _ = evaluate(
      model, val_loader, config['device'], criterion
    )
    
    # Track metrics
    history['train_loss'].append(train_loss)
    history['train_auc'].append(train_auc)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_auc'].append(val_auc)
    history['val_acc'].append(val_acc)
    
    print(f"Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
    print(f"Val   - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")
    
    # Learning rate scheduling
    scheduler.step(val_auc)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current LR: {current_lr:.2e}")
    
    # Early stopping based on AUC
    if val_auc > best_val_auc + 0.001:  # Small improvement threshold
      best_val_auc = val_auc
      patience_counter = 0
      torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_auc': val_auc,
        'config': config,
        'history': history
      }, os.path.join(output_dir, model_name))
      print(f"✓ Saved best model (AUC: {val_auc:.4f})")
    else:
      patience_counter += 1
      print(f"No improvement for {patience_counter} epoch(s)")
      
    if patience_counter >= config['early_stopping_patience']:
      print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
      print(f"Best validation AUC: {best_val_auc:.4f}")
      break
  
  total_epochs = epoch + 1
  
  # Save training history
  np.savez(os.path.join(output_dir, 'training_history.npz'), **history)
  print(f"\nTraining history saved to '{os.path.join(output_dir, 'training_history.npz')}'")
  
  # Plot training history
  plot_training_history(history, save_path=os.path.join(output_dir, 'training_history.png'))
  
  # Load best model and evaluate
  print("\n" + "="*70)
  print("Final evaluation with best model...")
  checkpoint = torch.load(os.path.join(output_dir, model_name), weights_only=False)
  model.load_state_dict(checkpoint['model_state_dict'])
  
  test_loss, test_auc, test_acc, test_preds, test_labels = evaluate(
    model, test_loader, config['device'], criterion
  )
  
  print(f"\nTest AUC:  {test_auc:.4f}")
  print(f"Test Acc:  {test_acc:.4f}")
  print(f"Test Loss: {test_loss:.4f}")
  
  # Classification report
  test_preds_binary = (np.array(test_preds) > 0.5).astype(int)
  print("\nClassification Report:")
  print(classification_report(test_labels, test_preds_binary, 
                              target_names=['Not Hateful', 'Hateful']))
  
  # Show prediction distribution
  print("\nPrediction Distribution:")
  print(f"  Mean probability: {np.mean(test_preds):.3f}")
  print(f"  Std probability:  {np.std(test_preds):.3f}")
  print(f"  Min probability:  {np.min(test_preds):.3f}")
  print(f"  Max probability:  {np.max(test_preds):.3f}")
  
  # Save results to text file
  save_results(output_dir, config, test_auc, test_acc, test_loss,
               test_labels, test_preds, best_val_auc, total_epochs)
  
  print(f"\n{'='*70}")
  print(f"All outputs saved to: {output_dir}")
  print(f"{'='*70}")


if __name__ == '__main__':
  main()