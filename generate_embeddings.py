#!/usr/bin/env python3

from transformers import AutoModel, AutoProcessor
from datasets import load_dataset
import torch
from tqdm import tqdm
import numpy as np
import os
<<<<<<< Updated upstream
import json
=======
import argparse
>>>>>>> Stashed changes
from PIL import Image


def main():

<<<<<<< Updated upstream
  # Load the model and processor
  print("Loading CLIP model...")
  model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
  processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
  working_dir = "data"
  knowledge_dir = "knowledge"
=======
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, default="openai/clip-vit-large-patch14")
  parser.add_argument('--input_dir', type=str, default="data")
  parser.add_argument('--output_dir', type=str, default="hateful_memes_clip")
  args = parser.parse_args()

  print("Loading CLIP model...")
  model = AutoModel.from_pretrained(args.model)
  processor = AutoProcessor.from_pretrained(args.model)
>>>>>>> Stashed changes

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = model.to(device)
  print(f"Using device: {device}")

  print("Loading Hateful Memes dataset...")
  dataset = load_dataset("neuralcatcher/hateful_memes")

  print(f"Dataset splits: {dataset.keys()}")
  print(f"Train split size: {len(dataset['train'])}")

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

<<<<<<< Updated upstream
  def load_split_knowledge(split_name):
    """Load GPT-generated descriptions/emotions for the split."""
    suffix = "val" if split_name == "validation" else split_name
    knowledge_path = os.path.join(knowledge_dir, f"lmm_knowledge_{suffix}.json")
    if not os.path.exists(knowledge_path):
      raise FileNotFoundError(f"Knowledge file not found for {split_name}: {knowledge_path}")
    with open(knowledge_path, "r") as f:
      knowledge = json.load(f)
    # Ensure string keys
    return {str(k): v for k, v in knowledge.items()}, knowledge_path

  def encode_and_average_texts(text_list):
    """Encode a list of texts with CLIP and average their normalized embeddings."""
    text_inputs = processor(text=text_list, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
      text_features = model.get_text_features(**text_inputs)
      text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
      mean_feature = text_features.mean(dim=0, keepdim=True)
      mean_feature = mean_feature / mean_feature.norm(p=2, dim=-1, keepdim=True)
      return mean_feature.squeeze(0).cpu().numpy()

  # Function to extract embeddings
=======
>>>>>>> Stashed changes
  def extract_embeddings(split_name, batch_size=32):
    split = dataset[split_name]

    knowledge, knowledge_path = load_split_knowledge(split_name)
    
    image_embeddings_list = []
    text_embeddings_list = []
    desc_embeddings_list = []
    emotion_embeddings_list = []
    labels_list = []
    ids_list = []
    
    print(f"\nProcessing {split_name} split...")
    print(f"Using knowledge from: {knowledge_path}")
    
    print("Checking for valid image files...")
    valid_indices = []
    missing_count = 0
    missing_knowledge = 0
    
    for i in tqdm(range(len(split))):
<<<<<<< Updated upstream
      img_path = os.path.join(working_dir, split[i]['img'])
      knowledge_entry = knowledge.get(str(split[i]['id']))

      if not is_valid_image(img_path):
=======
      img_path = os.path.join(args.input_dir, split[i]['img'])
      if is_valid_image(img_path):
        valid_indices.append(i)
      else:
>>>>>>> Stashed changes
        missing_count += 1
        if missing_count <= 10: 
          print(f"  Missing/invalid: {split[i]['img']} (ID: {split[i]['id']})")
        continue

      if not knowledge_entry or not knowledge_entry.get("descriptions") or not knowledge_entry.get("emotions"):
        missing_knowledge += 1
        if missing_knowledge <= 10:
          print(f"  Missing knowledge for ID: {split[i]['id']}")
        continue

      valid_indices.append(i)
    
    print(f"Found {len(valid_indices)} valid samples out of {len(split)}")
    if missing_count > 0:
      print(f"Filtered out {missing_count} missing/invalid samples")
    if missing_knowledge > 0:
      print(f"Filtered out {missing_knowledge} samples without knowledge entries")
    
    # Process in batches (only valid samples)
    print("Extracting embeddings...")
    for batch_start in tqdm(range(0, len(valid_indices), batch_size)):
      batch_end = min(batch_start + batch_size, len(valid_indices))
      batch_idx = valid_indices[batch_start:batch_end]
      
      batch_images = []
      batch_texts = []
      batch_desc_texts = []
      batch_emotion_texts = []
      batch_labels = []
      batch_ids = []
      
      for idx in batch_idx:
        img_path = os.path.join(args.input_dir, split[idx]['img'])
        try:
          img = Image.open(img_path).convert('RGB')
          knowledge_entry = knowledge[str(split[idx]['id'])]

          batch_images.append(img)
          batch_texts.append(split[idx]['text'])
          batch_desc_texts.append(knowledge_entry["descriptions"])
          batch_emotion_texts.append(knowledge_entry["emotions"])
          batch_labels.append(split[idx]['label'])
          batch_ids.append(split[idx]['id'])
        except Exception as e:
          print(f"  Error loading {img_path}: {e}")
          continue
      
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
          image_batch_np = image_embeds.cpu().numpy()
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
          text_batch_np = text_embeds.cpu().numpy()
      except Exception as e:
        print(f"  Error processing texts in batch: {e}")
        continue

      # Process description texts
      try:
        desc_batch_embeds = []
        for desc_list in batch_desc_texts:
          desc_embed = encode_and_average_texts(desc_list)
          desc_batch_embeds.append(desc_embed)
        desc_batch_np = np.vstack(desc_batch_embeds)
      except Exception as e:
        print(f"  Error processing descriptions in batch: {e}")
        continue

      # Process emotion texts
      try:
        emotion_batch_embeds = []
        for emotion_list in batch_emotion_texts:
          emotion_embed = encode_and_average_texts(emotion_list)
          emotion_batch_embeds.append(emotion_embed)
        emotion_batch_np = np.vstack(emotion_batch_embeds)
      except Exception as e:
        print(f"  Error processing emotions in batch: {e}")
        continue

      image_embeddings_list.append(image_batch_np)
      text_embeddings_list.append(text_batch_np)
      desc_embeddings_list.append(desc_batch_np)
      emotion_embeddings_list.append(emotion_batch_np)
      
      labels_list.extend(batch_labels)
      ids_list.extend(batch_ids)
    
    if len(image_embeddings_list) == 0:
      raise ValueError(f"No valid samples found in {split_name} split!")
    
    image_embeddings = np.vstack(image_embeddings_list)
    text_embeddings = np.vstack(text_embeddings_list)
    desc_embeddings = np.vstack(desc_embeddings_list)
    emotion_embeddings = np.vstack(emotion_embeddings_list)
    text_concat_embeddings = np.concatenate(
      [text_embeddings, desc_embeddings, emotion_embeddings], axis=1)
    meme_concat_embeddings = np.concatenate(
      [image_embeddings, text_embeddings, desc_embeddings, emotion_embeddings], axis=1)
    labels_array = np.array(labels_list)
    ids_array = np.array(ids_list)
    
    print(f"Image embeddings shape: {image_embeddings.shape}")
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"Description embeddings shape: {desc_embeddings.shape}")
    print(f"Emotion embeddings shape: {emotion_embeddings.shape}")
    print(f"Text concatenated embeddings shape: {text_concat_embeddings.shape}")
    print(f"Meme concatenated embeddings shape: {meme_concat_embeddings.shape}")
    
    return {
      'image_embeddings': image_embeddings,
      'text_embeddings': text_embeddings,
      'desc_embeddings': desc_embeddings,
      'emotion_embeddings': emotion_embeddings,
      'text_concat_embeddings': text_concat_embeddings,
      'meme_concat_embeddings': meme_concat_embeddings,
      'labels': labels_array,
      'ids': ids_array,
      'valid_indices': valid_indices
    }

  if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

  print("\n" + "="*50)
  train_embeddings = extract_embeddings('train', batch_size=32)

  # Save embeddings
  print("\nSaving embeddings...")
  np.savez(os.path.join(args.output_dir, 'embeddings_train.npz'),
    image_embeddings=train_embeddings['image_embeddings'],
    text_embeddings=train_embeddings['text_embeddings'],
    desc_embeddings=train_embeddings['desc_embeddings'],
    emotion_embeddings=train_embeddings['emotion_embeddings'],
    text_concat_embeddings=train_embeddings['text_concat_embeddings'],
    meme_concat_embeddings=train_embeddings['meme_concat_embeddings'],
    labels=train_embeddings['labels'],
    ids=train_embeddings['ids'],
    valid_indices=np.array(train_embeddings['valid_indices']))

  print("Training embeddings saved in", args.output_dir)

  # If you want to process validation/test splits as well:
  if 'validation' in dataset:
    val_embeddings = extract_embeddings('validation', batch_size=32)
    np.savez(os.path.join(args.output_dir, 'embeddings_val.npz'),
      image_embeddings=val_embeddings['image_embeddings'],
      text_embeddings=val_embeddings['text_embeddings'],
      desc_embeddings=val_embeddings['desc_embeddings'],
      emotion_embeddings=val_embeddings['emotion_embeddings'],
      text_concat_embeddings=val_embeddings['text_concat_embeddings'],
      meme_concat_embeddings=val_embeddings['meme_concat_embeddings'],
      labels=val_embeddings['labels'],
<<<<<<< Updated upstream
      ids=val_embeddings['ids'],
      valid_indices=np.array(val_embeddings['valid_indices']))
    np.savez('valid_indices_val.npz',
      valid_indices=np.array(val_embeddings['valid_indices']))
=======
      ids=val_embeddings['ids'])
    # np.savez('cleaned_valid_indices_val.npz',
    #   valid_indices=np.array(val_embeddings['valid_indices']))
>>>>>>> Stashed changes
    print("Validation embeddings saved!")

  if 'test' in dataset:
    test_embeddings = extract_embeddings('test', batch_size=32)
    np.savez(os.path.join(args.output_dir, 'embeddings_test.npz'),
      image_embeddings=test_embeddings['image_embeddings'],
      text_embeddings=test_embeddings['text_embeddings'],
      desc_embeddings=test_embeddings['desc_embeddings'],
      emotion_embeddings=test_embeddings['emotion_embeddings'],
      text_concat_embeddings=test_embeddings['text_concat_embeddings'],
      meme_concat_embeddings=test_embeddings['meme_concat_embeddings'],
      labels=test_embeddings['labels'],
<<<<<<< Updated upstream
      ids=test_embeddings['ids'],
      valid_indices=np.array(test_embeddings['valid_indices']))
    np.savez('valid_indices_test.npz',
      valid_indices=np.array(test_embeddings['valid_indices']))
=======
      ids=test_embeddings['ids'])
    # np.savez('cleaned_valid_indices_test.npz',
    #   valid_indices=np.array(test_embeddings['valid_indices']))
>>>>>>> Stashed changes
    print("Test embeddings saved!")


if __name__ == '__main__':
  main()
