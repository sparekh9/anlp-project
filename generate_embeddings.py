#!/usr/bin/env python3

from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
import torch
from tqdm import tqdm
import numpy as np
import os
from PIL import Image


def main():

  # Load the model and processor
  print("Loading CLIP model...")
  model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
  processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
  working_dir = "data"

  # Move model to GPU if available
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = model.to(device)
  print(f"Using device: {device}")

  # Load the Hateful Memes dataset
  print("Loading Hateful Memes dataset...")
  dataset = load_dataset("neuralcatcher/hateful_memes")

  print(f"Dataset splits: {dataset.keys()}")
  print(f"Train split size: {len(dataset['train'])}")

  # Function to check if image file exists and is valid
  def is_valid_image(img_path):
    """Check if image file exists and can be opened."""
    if not os.path.exists(img_path):
      return False
    try:
      with Image.open(img_path) as img:
        img.verify()
      return True
    except Exception:
      return False

  # Function to extract embeddings
  def extract_embeddings(split_name, batch_size=32):
    split = dataset[split_name]
    
    image_embeddings_list = []
    text_embeddings_list = []
    labels_list = []
    ids_list = []
    
    print(f"\nProcessing {split_name} split...")
    
    # First pass: identify valid samples
    print("Checking for valid image files...")
    valid_indices = []
    missing_count = 0
    
    for i in tqdm(range(len(split))):
      img_path = os.path.join(working_dir, split[i]['img'])
      if is_valid_image(img_path):
        valid_indices.append(i)
      else:
        missing_count += 1
        if missing_count <= 10:  # Only print first 10 to avoid spam
          print(f"  Missing/invalid: {split[i]['img']} (ID: {split[i]['id']})")
    
    print(f"Found {len(valid_indices)} valid samples out of {len(split)}")
    if missing_count > 0:
      print(f"Filtered out {missing_count} missing/invalid samples")
    
    # Process in batches (only valid samples)
    print("Extracting embeddings...")
    for batch_start in tqdm(range(0, len(valid_indices), batch_size)):
      batch_end = min(batch_start + batch_size, len(valid_indices))
      batch_idx = valid_indices[batch_start:batch_end]
      
      # Collect batch data
      batch_images = []
      batch_texts = []
      batch_labels = []
      batch_ids = []
      
      for idx in batch_idx:
        img_path = os.path.join(working_dir, split[idx]['img'])
        try:
          img = Image.open(img_path).convert('RGB')
          batch_images.append(img)
          batch_texts.append(split[idx]['text'])
          batch_labels.append(split[idx]['label'])
          batch_ids.append(split[idx]['id'])
        except Exception as e:
          print(f"  Error loading {img_path}: {e}")
          continue
      
      # Skip empty batches
      if len(batch_images) == 0:
        continue
      
      # Process images
      try:
        image_inputs = processor(images=batch_images, return_tensors="pt", padding=True)
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        
        with torch.no_grad():
          image_embeds = model.get_image_features(**image_inputs)
          # Normalize
          image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
          image_embeddings_list.append(image_embeds.cpu().numpy())
      except Exception as e:
        print(f"  Error processing images in batch: {e}")
        continue
      
      # Process texts
      try:
        text_inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
          text_embeds = model.get_text_features(**text_inputs)
          # Normalize
          text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
          text_embeddings_list.append(text_embeds.cpu().numpy())
      except Exception as e:
        print(f"  Error processing texts in batch: {e}")
        continue
      
      labels_list.extend(batch_labels)
      ids_list.extend(batch_ids)
    
    # Concatenate all batches
    if len(image_embeddings_list) == 0:
      raise ValueError(f"No valid samples found in {split_name} split!")
    
    image_embeddings = np.vstack(image_embeddings_list)
    text_embeddings = np.vstack(text_embeddings_list)
    labels_array = np.array(labels_list)
    ids_array = np.array(ids_list)
    
    print(f"Image embeddings shape: {image_embeddings.shape}")
    print(f"Text embeddings shape: {text_embeddings.shape}")
    
    return {
      'image_embeddings': image_embeddings,
      'text_embeddings': text_embeddings,
      'labels': labels_array,
      'ids': ids_array,
      'valid_indices': valid_indices
    }

  # Extract embeddings for each split
  print("\n" + "="*50)
  train_embeddings = extract_embeddings('train', batch_size=32)

  # Save embeddings
  print("\nSaving embeddings...")
  np.savez('hateful_memes_clip_embeddings_train.npz',
    image_embeddings=train_embeddings['image_embeddings'],
    text_embeddings=train_embeddings['text_embeddings'],
    labels=train_embeddings['labels'],
    ids=train_embeddings['ids'])

  print("Embeddings saved to 'hateful_memes_clip_embeddings_train.npz'")

  # Save valid indices for reference
  np.savez('valid_indices_train.npz', 
    valid_indices=np.array(train_embeddings['valid_indices']))
  print("Valid indices saved to 'valid_indices_train.npz'")

  # If you want to process validation/test splits as well:
  if 'validation' in dataset:
    val_embeddings = extract_embeddings('validation', batch_size=32)
    np.savez('hateful_memes_clip_embeddings_val.npz',
      image_embeddings=val_embeddings['image_embeddings'],
      text_embeddings=val_embeddings['text_embeddings'],
      labels=val_embeddings['labels'],
      ids=val_embeddings['ids'])
    np.savez('valid_indices_val.npz',
      valid_indices=np.array(val_embeddings['valid_indices']))
    print("Validation embeddings saved!")

  if 'test' in dataset:
    test_embeddings = extract_embeddings('test', batch_size=32)
    np.savez('hateful_memes_clip_embeddings_test.npz',
      image_embeddings=test_embeddings['image_embeddings'],
      text_embeddings=test_embeddings['text_embeddings'],
      labels=test_embeddings['labels'],
      ids=test_embeddings['ids'])
    np.savez('valid_indices_test.npz',
      valid_indices=np.array(test_embeddings['valid_indices']))
    print("Test embeddings saved!")

  # Example: Load embeddings later
  print("\n" + "="*50)
  print("To load the embeddings later, use:")
  print("data = np.load('hateful_memes_clip_embeddings_train.npz')")
  print("image_emb = data['image_embeddings']")
  print("text_emb = data['text_embeddings']")
  print("labels = data['labels']")
  print("ids = data['ids']")

if __name__ == '__main__':
  main()