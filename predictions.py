#!/usr/bin/env python3

import torch
import numpy as np
from training import HatefulMemeClassifier


def predict_hate_probability(model, image_embed, text_embed, device='cpu'):
    
    model.eval()
    
    # Convert to tensors and add batch dimension
    image_tensor = torch.FloatTensor(image_embed).unsqueeze(0).to(device)
    text_tensor = torch.FloatTensor(text_embed).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(image_tensor, text_tensor).squeeze(-1)
        probability = torch.sigmoid(logits).item()
    
    return probability


def predict_batch(model, image_embeds, text_embeds, device='cpu', batch_size=64):
    
    model.eval()
    
    all_probs = []
    num_samples = len(image_embeds)
    
    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        
        image_batch = torch.FloatTensor(image_embeds[i:batch_end]).to(device)
        text_batch = torch.FloatTensor(text_embeds[i:batch_end]).to(device)
        
        with torch.no_grad():
            logits = model(image_batch, text_batch).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
    
    return np.array(all_probs)


def load_model(checkpoint_path, device='cpu'):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = HatefulMemeClassifier(
        image_dim=768,  # CLIP embedding dim
        text_dim=768,   # CLIP embedding dim
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def main():
    """Example usage of the prediction functions."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model, checkpoint = load_model('best_model.pth', device)
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Validation AUC: {checkpoint['val_auc']:.4f}")
    
    # Load test data (or validation data)
    print("\nLoading embeddings...")
    data = np.load('hateful_memes_clip_embeddings_test.npz')
    image_embeds = data['image_embeddings']
    text_embeds = data['text_embeddings']
    labels = data['labels']
    ids = data['ids']
    
    print("\nMaking predictions...")
    probabilities = predict_batch(model, image_embeds, text_embeds, device)
    
    # Get predictions
    predictions = (probabilities > 0.5).astype(int)
    
    # # Show some examples
    # print("\nExample predictions:")
    # print("-" * 80)
    # for i in range(min(10, len(probabilities))):
    #     print(f"ID: {ids[i]}")
    #     print(f"  True label: {'Hateful' if labels[i] == 1 else 'Not Hateful'}")
    #     print(f"  Predicted: {'Hateful' if predictions[i] == 1 else 'Not Hateful'}")
    #     print(f"  Probability: {probabilities[i]:.4f}")
    #     print()
    
    # Save predictions
    print("Saving predictions...")
    np.savez('predictions.npz',
             probabilities=probabilities,
             predictions=predictions,
             labels=labels,
             ids=ids)
    print("✓ Predictions saved to 'predictions.npz'")
    
    # Summary statistics
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
    
    auc = roc_auc_score(labels, probabilities)
    acc = accuracy_score(labels, predictions)
    cm = confusion_matrix(labels, predictions)
    
    print("\n" + "="*70)
    print("PREDICTION SUMMARY")
    print("="*70)
    print(f"Total samples: {len(probabilities)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Not Hateful  Hateful")
    print(f"Actual Not Hateful    {cm[0,0]:6d}      {cm[0,1]:6d}")
    print(f"Actual Hateful        {cm[1,0]:6d}      {cm[1,1]:6d}")


if __name__ == '__main__':
    main()