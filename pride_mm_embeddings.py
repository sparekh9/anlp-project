#!/usr/bin/env python3

from transformers import AutoModel, AutoProcessor
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import argparse
from PIL import Image


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, default="openai/clip-vit-large-patch14")
  parser.add_argument('--input_dir', type=str, default="data", help="Directory containing the PrideMM images")
  parser.add_argument('--csv_path', type=str, default="PrideMM.csv", help="Path to PrideMM.csv file")
  parser.add_argument('--output_dir', type=str, default="pridemm_clip")
  args = parser.parse_args()

  # Load the model and processor
  print("Loading CLIP model...")
  model = AutoModel.from_pretrained(args.model)
  processor = AutoProcessor.from_pretrained(args.model)

  # Move model to GPU if available
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = model.to(device)
  print(f"Using device: {device}")

  # Load the PrideMM CSV
  print(f"Loading PrideMM dataset from {args.csv_path}...")
  df = pd.read_csv(args.csv_path)
  
  print(f"Total samples in CSV: {len(df)}")
  print(f"Dataset splits: {df['split'].unique()}")
  
  
  # Print split distribution
  for split in df['split'].unique():
    count = len(df[df['split'] == split])
    print(f"  {split}: {count} samples")

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

  # Function to extract embeddings for a specific split
  def extract_embeddings(split_name, batch_size=32):
    # Filter dataframe for this split
    split_df = df[df['split'] == split_name].reset_index(drop=True)
    
    if len(split_df) == 0:
      print(f"\nNo samples found for {split_name} split. Skipping...")
      return None
    
    image_embeddings_list = []
    text_embeddings_list = []
    labels_list = []
    names_list = []
    
    print(f"\nProcessing {split_name} split...")
    
    # First pass: identify valid samples
    print("Checking for valid image files...")
    valid_indices = []
    missing_count = 0
    
    for i in tqdm(range(len(split_df))):
      img_path = os.path.join(args.input_dir, split_df.loc[i, 'name'])
      if is_valid_image(img_path):
        valid_indices.append(i)
      else:
        missing_count += 1
        if missing_count <= 10:  # Only print first 10 to avoid spam
          print(f"  Missing/invalid: {split_df.loc[i, 'name']}")
    
    print(f"Found {len(valid_indices)} valid samples out of {len(split_df)}")
    if missing_count > 0:
      print(f"Filtered out {missing_count} missing/invalid samples")
    
    if len(valid_indices) == 0:
      print(f"No valid samples found in {split_name} split!")
      return None
    
    # Process in batches (only valid samples)
    print("Extracting embeddings...")
    for batch_start in tqdm(range(0, len(valid_indices), batch_size)):
      batch_end = min(batch_start + batch_size, len(valid_indices))
      batch_idx = valid_indices[batch_start:batch_end]
      
      # Collect batch data
      batch_images = []
      batch_texts = []
      batch_labels = []
      batch_names = []
      
      for idx in batch_idx:
        row = split_df.loc[idx]
        img_path = os.path.join(args.input_dir, row['name'])
        try:
          img = Image.open(img_path).convert('RGB')
          batch_images.append(img)
          batch_texts.append(row['text'])
          batch_labels.append(row['hate'])
          batch_names.append(row['name'])
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
          text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
          text_embeddings_list.append(text_embeds.cpu().numpy())
      except Exception as e:
        print(f"  Error processing texts in batch: {e}")
        continue
      
      labels_list.extend(batch_labels)
      names_list.extend(batch_names)
    
    # Concatenate all batches
    if len(image_embeddings_list) == 0:
      print(f"No valid samples processed in {split_name} split!")
      return None
    
    image_embeddings = np.vstack(image_embeddings_list)
    text_embeddings = np.vstack(text_embeddings_list)
    labels_array = np.array(labels_list)
    names_array = np.array(names_list)
    
    print(f"Image embeddings shape: {image_embeddings.shape}")
    print(f"Text embeddings shape: {text_embeddings.shape}")
    
    return {
      'image_embeddings': image_embeddings,
      'text_embeddings': text_embeddings,
      'labels': labels_array,
      'names': names_array,
      'valid_indices': valid_indices
    }

  # Create output directory
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  # Extract embeddings for each split
  print("\n" + "="*50)
  
  splits = df['split'].unique()
  
  for split in ['train', 'val', 'validation', 'test']:
    if split in splits:
      split_embeddings = extract_embeddings(split, batch_size=32)
      
      if split_embeddings is not None:
        # Normalize split name for output files
        output_split = 'val' if split == 'validation' else split
        
        # Save embeddings
        print(f"\nSaving {split} embeddings...")
        np.savez(os.path.join(args.output_dir, f'embeddings_{output_split}.npz'),
          image_embeddings=split_embeddings['image_embeddings'],
          text_embeddings=split_embeddings['text_embeddings'],
          labels=split_embeddings['labels'],
          names=split_embeddings['names'])
        
        print(f"{split.capitalize()} embeddings saved!")

  print("\n" + "="*50)
  print(f"All embeddings saved in '{args.output_dir}'")


if __name__ == '__main__':
  main()