#!/usr/bin/env python3

from transformers import CLIPProcessor, CLIPModel
import torch
from tqdm import tqdm
import numpy as np
import os
import json
from PIL import Image
import argparse
from typing import Dict, List


def is_valid_image(img_path: str) -> bool:
  """Check if image file exists and can be opened."""
  if not os.path.exists(img_path):
    return False
  try:
    with Image.open(img_path) as img:
      img.verify()
    return True
  except Exception:
    return False


def load_pridemm_split(jsonl_path: str) -> List[Dict]:
  """Load a PrideMM split from a local JSONL file."""
  examples: List[Dict] = []
  if not os.path.exists(jsonl_path):
    raise FileNotFoundError(f"PrideMM split not found: {jsonl_path}")

  with open(jsonl_path, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        obj = json.loads(line)
      except json.JSONDecodeError as e:
        print(f"Skipping malformed JSON in {jsonl_path}:{line_num}: {e}")
        continue
      if "id" not in obj or "img" not in obj or "text" not in obj or "label" not in obj:
        print(f"Skipping record without required fields in {jsonl_path}:{line_num}")
        continue
      examples.append(obj)
  return examples


def load_split_knowledge(knowledge_dir: str, split_name: str, prefix: str) -> Dict[str, Dict]:
  """Load GPT-generated descriptions/emotions for the split."""
  suffix = "val" if split_name == "validation" else split_name
  knowledge_path = os.path.join(knowledge_dir, f"{prefix}_{suffix}.json")
  if not os.path.exists(knowledge_path):
    raise FileNotFoundError(f"Knowledge file not found for {split_name}: {knowledge_path}")
  with open(knowledge_path, "r") as f:
    knowledge = json.load(f)
  return {str(k): v for k, v in knowledge.items()}


def encode_and_average_texts(
  texts: List[str],
  processor: CLIPProcessor,
  model: CLIPModel,
  device: str
) -> np.ndarray:
  """Encode a list of texts with CLIP and average their normalized embeddings."""
  text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
  text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
  with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    mean_feature = text_features.mean(dim=0, keepdim=True)
    mean_feature = mean_feature / mean_feature.norm(p=2, dim=-1, keepdim=True)
    return mean_feature.squeeze(0).cpu().numpy()


def extract_embeddings_for_split(
  split_name: str,
  split_examples: List[Dict],
  processor: CLIPProcessor,
  model: CLIPModel,
  device: str,
  image_root: str,
  knowledge_dir: str,
  knowledge_prefix: str,
  batch_size: int = 32,
) -> Dict[str, np.ndarray]:
  """Extract CLIP and knowledge-based embeddings for a PrideMM split."""
  knowledge = load_split_knowledge(knowledge_dir, split_name, knowledge_prefix)

  image_embeddings_list = []
  text_embeddings_list = []
  desc_embeddings_list = []
  emotion_embeddings_list = []
  labels_list = []
  ids_list = []

  print(f"\nProcessing {split_name} split...")
  print(f"Using knowledge prefix: {knowledge_prefix}")

  # First pass: identify valid samples
  print("Checking for valid image files and knowledge entries...")
  valid_indices = []
  missing_count = 0
  missing_knowledge = 0

  for i, example in enumerate(tqdm(split_examples, desc=f"{split_name} scan")):
    img_path = os.path.join(image_root, example["img"])
    knowledge_entry = knowledge.get(str(example["id"]))

    if not is_valid_image(img_path):
      missing_count += 1
      if missing_count <= 10:
        print(f"  Missing/invalid: {example['img']} (ID: {example['id']})")
      continue

    if not knowledge_entry or not knowledge_entry.get("descriptions") or not knowledge_entry.get("emotions"):
      missing_knowledge += 1
      if missing_knowledge <= 10:
        print(f"  Missing knowledge for ID: {example['id']}")
      continue

    valid_indices.append(i)

  print(f"Found {len(valid_indices)} valid samples out of {len(split_examples)}")
  if missing_count > 0:
    print(f"Filtered out {missing_count} missing/invalid samples")
  if missing_knowledge > 0:
    print(f"Filtered out {missing_knowledge} samples without knowledge entries")

  # Process in batches
  print("Extracting embeddings...")
  for batch_start in tqdm(range(0, len(valid_indices), batch_size), desc=f"{split_name} batches"):
    batch_end = min(batch_start + batch_size, len(valid_indices))
    batch_idx = valid_indices[batch_start:batch_end]

    batch_images = []
    batch_texts = []
    batch_desc_texts = []
    batch_emotion_texts = []
    batch_labels = []
    batch_ids = []

    for idx in batch_idx:
      example = split_examples[idx]
      img_path = os.path.join(image_root, example["img"])
      try:
        img = Image.open(img_path).convert("RGB")
        knowledge_entry = knowledge[str(example["id"])]

        batch_images.append(img)
        batch_texts.append(example["text"])
        batch_desc_texts.append(knowledge_entry["descriptions"])
        batch_emotion_texts.append(knowledge_entry["emotions"])
        batch_labels.append(example["label"])
        batch_ids.append(example["id"])
      except Exception as e:
        print(f"  Error loading {img_path}: {e}")
        continue

    if not batch_images:
      continue

    # Images
    try:
      image_inputs = processor(images=batch_images, return_tensors="pt", padding=True)
      image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
      with torch.no_grad():
        image_embeds = model.get_image_features(**image_inputs)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        image_batch_np = image_embeds.cpu().numpy()
    except Exception as e:
      print(f"  Error processing images in batch: {e}")
      continue

    # Meme texts
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

    # Description texts
    try:
      desc_batch_embeds = []
      for desc_list in batch_desc_texts:
        desc_embed = encode_and_average_texts(desc_list, processor, model, device)
        desc_batch_embeds.append(desc_embed)
      desc_batch_np = np.vstack(desc_batch_embeds)
    except Exception as e:
      print(f"  Error processing descriptions in batch: {e}")
      continue

    # Emotion texts
    try:
      emotion_batch_embeds = []
      for emotion_list in batch_emotion_texts:
        emotion_embed = encode_and_average_texts(emotion_list, processor, model, device)
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

  if not image_embeddings_list:
    raise ValueError(f"No valid samples found in {split_name} split!")

  image_embeddings = np.vstack(image_embeddings_list)
  text_embeddings = np.vstack(text_embeddings_list)
  desc_embeddings = np.vstack(desc_embeddings_list)
  emotion_embeddings = np.vstack(emotion_embeddings_list)
  text_concat_embeddings = np.concatenate([text_embeddings, desc_embeddings, emotion_embeddings], axis=1)
  meme_concat_embeddings = np.concatenate(
    [image_embeddings, text_embeddings, desc_embeddings, emotion_embeddings],
    axis=1,
  )
  labels_array = np.array(labels_list)
  ids_array = np.array(ids_list)

  print(f"Image embeddings shape: {image_embeddings.shape}")
  print(f"Text embeddings shape: {text_embeddings.shape}")
  print(f"Description embeddings shape: {desc_embeddings.shape}")
  print(f"Emotion embeddings shape: {emotion_embeddings.shape}")
  print(f"Text concatenated embeddings shape: {text_concat_embeddings.shape}")
  print(f"Meme concatenated embeddings shape: {meme_concat_embeddings.shape}")

  return {
    "image_embeddings": image_embeddings,
    "text_embeddings": text_embeddings,
    "desc_embeddings": desc_embeddings,
    "emotion_embeddings": emotion_embeddings,
    "text_concat_embeddings": text_concat_embeddings,
    "meme_concat_embeddings": meme_concat_embeddings,
    "labels": labels_array,
    "ids": ids_array,
    "valid_indices": valid_indices,
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Generate CLIP embeddings for the PrideMM dataset using local JSONL metadata and LMM knowledge.",
  )
  parser.add_argument(
    "--metadata-dir",
    default=os.path.join("data", "PrideMM"),
    help="Directory containing PrideMM train/validation/test JSONL files.",
  )
  parser.add_argument(
    "--image-root",
    default="data",
    help="Root directory that, joined with example['img'], locates image files.",
  )
  parser.add_argument(
    "--knowledge-dir",
    default="knowledge",
    help="Directory containing pridemm_lmm_knowledge_{train,val,test}.json.",
  )
  parser.add_argument(
    "--knowledge-prefix",
    default="pridemm_lmm_knowledge",
    help="Prefix for knowledge files (default matches generate_knowledge_pridemm.py).",
  )
  parser.add_argument(
    "--output-prefix",
    default="pridemm_clip_embeddings",
    help="Prefix for the output NPZ files.",
  )
  parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size for CLIP encoding.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  print("Loading CLIP model...")
  model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
  processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

  if torch.cuda.is_available():
    device = "cuda"
  elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
  else:
    device = "cpu"
  model = model.to(device)
  print(f"Using device: {device}")

  # Load PrideMM splits from JSONL
  print("Loading PrideMM JSONL splits...")
  train_path = os.path.join(args.metadata_dir, "train.jsonl")
  val_path = os.path.join(args.metadata_dir, "validation.jsonl")
  test_path = os.path.join(args.metadata_dir, "test.jsonl")

  train_split = load_pridemm_split(train_path)
  val_split = load_pridemm_split(val_path) if os.path.exists(val_path) else []
  test_split = load_pridemm_split(test_path) if os.path.exists(test_path) else []

  print(f"Train size: {len(train_split)}")
  print(f"Validation size: {len(val_split)}")
  print(f"Test size: {len(test_split)}")

  print("\n" + "=" * 50)
  train_embeddings = extract_embeddings_for_split(
    "train",
    train_split,
    processor,
    model,
    device,
    args.image_root,
    args.knowledge_dir,
    args.knowledge_prefix,
    batch_size=args.batch_size,
  )

  print("\nSaving train embeddings...")
  np.savez(
    f"{args.output_prefix}_train.npz",
    image_embeddings=train_embeddings["image_embeddings"],
    text_embeddings=train_embeddings["text_embeddings"],
    desc_embeddings=train_embeddings["desc_embeddings"],
    emotion_embeddings=train_embeddings["emotion_embeddings"],
    text_concat_embeddings=train_embeddings["text_concat_embeddings"],
    meme_concat_embeddings=train_embeddings["meme_concat_embeddings"],
    labels=train_embeddings["labels"],
    ids=train_embeddings["ids"],
    valid_indices=np.array(train_embeddings["valid_indices"]),
  )
  np.savez(
    "valid_indices_pridemm_train.npz",
    valid_indices=np.array(train_embeddings["valid_indices"]),
  )
  print(f"Train embeddings saved to '{args.output_prefix}_train.npz'")

  if val_split:
    val_embeddings = extract_embeddings_for_split(
      "validation",
      val_split,
      processor,
      model,
      device,
      args.image_root,
      args.knowledge_dir,
      args.knowledge_prefix,
      batch_size=args.batch_size,
    )
    np.savez(
      f"{args.output_prefix}_val.npz",
      image_embeddings=val_embeddings["image_embeddings"],
      text_embeddings=val_embeddings["text_embeddings"],
      desc_embeddings=val_embeddings["desc_embeddings"],
      emotion_embeddings=val_embeddings["emotion_embeddings"],
      text_concat_embeddings=val_embeddings["text_concat_embeddings"],
      meme_concat_embeddings=val_embeddings["meme_concat_embeddings"],
      labels=val_embeddings["labels"],
      ids=val_embeddings["ids"],
      valid_indices=np.array(val_embeddings["valid_indices"]),
    )
    np.savez(
      "valid_indices_pridemm_val.npz",
      valid_indices=np.array(val_embeddings["valid_indices"]),
    )
    print(f"Validation embeddings saved to '{args.output_prefix}_val.npz'")

  if test_split:
    test_embeddings = extract_embeddings_for_split(
      "test",
      test_split,
      processor,
      model,
      device,
      args.image_root,
      args.knowledge_dir,
      args.knowledge_prefix,
      batch_size=args.batch_size,
    )
    np.savez(
      f"{args.output_prefix}_test.npz",
      image_embeddings=test_embeddings["image_embeddings"],
      text_embeddings=test_embeddings["text_embeddings"],
      desc_embeddings=test_embeddings["desc_embeddings"],
      emotion_embeddings=test_embeddings["emotion_embeddings"],
      text_concat_embeddings=test_embeddings["text_concat_embeddings"],
      meme_concat_embeddings=test_embeddings["meme_concat_embeddings"],
      labels=test_embeddings["labels"],
      ids=test_embeddings["ids"],
      valid_indices=np.array(test_embeddings["valid_indices"]),
    )
    np.savez(
      "valid_indices_pridemm_test.npz",
      valid_indices=np.array(test_embeddings["valid_indices"]),
    )
    print(f"Test embeddings saved to '{args.output_prefix}_test.npz'")


if __name__ == "__main__":
  main()
