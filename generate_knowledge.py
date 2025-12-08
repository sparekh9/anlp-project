#!/usr/bin/env python3

"""Generate per-meme descriptions and emotions using a vision-language GPT model."""

import argparse
import base64
import json
import os
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from together import Together
from PIL import Image
from tqdm import tqdm


def encode_image_to_data_url(image_path: str) -> str:
  """Encode an image as a base64 data URL for GPT vision input."""
  with open(image_path, "rb") as f:
    image_bytes = f.read()
  encoded = base64.b64encode(image_bytes).decode("utf-8")
  ext = os.path.splitext(image_path)[1].lower().lstrip(".") or "jpeg"
  mime = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"
  return f"data:{mime};base64,{encoded}"


def is_valid_image(img_path: str) -> bool:
  """Check if an image exists and can be opened."""
  if not os.path.exists(img_path):
    return False
  try:
    with Image.open(img_path) as img:
      img.verify()
    return True
  except Exception:
    return False


def normalize_list(items, target_len: int, placeholder: str) -> List[str]:
  """Ensure we always return a list of length target_len."""
  if isinstance(items, list):
    cleaned = [str(x).strip() for x in items if str(x).strip()]
  elif isinstance(items, str):
    cleaned = [line.strip(" -") for line in items.splitlines() if line.strip()]
  else:
    cleaned = []

  cleaned = cleaned[:target_len]
  if not cleaned:
    cleaned = [placeholder]

  if len(cleaned) < target_len:
    cleaned.extend([cleaned[-1]] * (target_len - len(cleaned)))

  return cleaned


def extract_knowledge_from_response(
  response_text: str,
  num_descriptions: int,
  num_emotions: int
) -> Optional[Dict[str, List[str]]]:
  """Parse GPT JSON output into lists of descriptions and emotions."""
  clean_text = response_text.strip()
  if clean_text.startswith("```"):
    # Strip Markdown fences (with or without language hints)
    lines = clean_text.splitlines()
    # Drop first fence line
    lines = lines[1:]
    # Drop trailing fence if present
    if lines and lines[-1].strip().startswith("```"):
      lines = lines[:-1]
    clean_text = "\n".join(lines).strip()

  try:
    parsed = json.loads(clean_text)
  except json.JSONDecodeError:
    return None

  descriptions = normalize_list(parsed.get("descriptions"), num_descriptions, "description")
  emotions = normalize_list(parsed.get("emotions"), num_emotions, "emotion")

  return {"descriptions": descriptions, "emotions": emotions}


def request_knowledge(
  client: Together,
  model: str,
  image_path: str,
  meme_text: str,
  num_descriptions: int,
  num_emotions: int
) -> Tuple[Optional[Dict[str, List[str]]], str]:
  """Call GPT with image + text to get descriptions and emotions."""
  data_url = encode_image_to_data_url(image_path)

  system_prompt = (
    "You analyze memes by considering both the image and its embedded text. "
    "Return concise semantic descriptions and emotions the meme could evoke."
  )

  user_prompt = (
    "Look at the meme image and the embedded text provided below. "
    f"Write exactly {num_descriptions} concise descriptions of what the meme communicates "
    f"and {num_emotions} emotions it may evoke. "
    "Return only valid JSON with two keys: "
    '\"descriptions\" (list of strings) and \"emotions\" (list of strings). '
    "Do not add explanations or extra keys."
    f"\n\nMeme text:\n{meme_text}"
  )

  response = client.chat.completions.create(
    model=model,
    messages=[
      {"role": "system", "content": system_prompt},
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": data_url}},
          {"type": "text", "text": user_prompt},
        ],
      },
    ],
    temperature=0.3,
  )

  content = response.choices[0].message.content
  parsed = extract_knowledge_from_response(content, num_descriptions, num_emotions)
  return parsed, content




def save_split_knowledge(
  split_name: str,
  dataset,
  client: Together,
  model: str,
  data_dir: str,
  output_dir: str,
  num_descriptions: int,
  num_emotions: int
) -> None:
  """Generate and save knowledge for a single split."""
  split = dataset[split_name]
  knowledge: Dict[str, Dict[str, List[str]]] = {}
  missing = 0

  print(f"\nProcessing split: {split_name} ({len(split)} samples)")
  for example in tqdm(split, desc=f"{split_name} split"):
    meme_id = str(example["id"])
    img_path = os.path.join(data_dir, example["img"])
    meme_text = example["text"]

    if not is_valid_image(img_path):
      missing += 1
      continue

    try:
      result, raw_response = request_knowledge(
        client,
        model,
        img_path,
        meme_text,
        num_descriptions=num_descriptions,
        num_emotions=num_emotions,
      )
    except Exception as e:
      print(f"  Failed on id {meme_id}: {e}")
      result = None
      raw_response = str(e)

    if result is None:
      print(f"  Could not parse response for id {meme_id}. Raw response:\n{raw_response}")
      continue

    knowledge[meme_id] = result

  split_suffix = "val" if split_name == "validation" else split_name
  output_path = os.path.join(output_dir, f"lmm_knowledge_{split_suffix}.json")
  os.makedirs(output_dir, exist_ok=True)
  with open(output_path, "w") as f:
    json.dump(knowledge, f, indent=2)

  print(
    f"Saved {len(knowledge)} entries for {split_name} to {output_path}. "
    f"Skipped {missing} missing/invalid images."
  )


def parse_args():
  parser = argparse.ArgumentParser(description="Generate meme descriptions and emotions via GPT.")
  parser.add_argument("--model", default="yiwei_64f1/Qwen/Qwen2.5-VL-72B-Instruct-7f41fb31", help="Together model to use for generation.")
  parser.add_argument("--data-dir", default="data", help="Directory containing meme images.")
  parser.add_argument("--output-dir", default="knowledge", help="Where to write JSON knowledge files.")
  parser.add_argument("--num-descriptions", type=int, default=10, help="Number of descriptions per meme.")
  parser.add_argument("--num-emotions", type=int, default=10, help="Number of emotions per meme.")
  return parser.parse_args()


def main():
  args = parse_args()

  print("Loading hateful_memes dataset...")
  dataset = load_dataset("neuralcatcher/hateful_memes")

  client = Together()
  print(f"Using model: {args.model}")

  for split_name in ["train", "validation", "test"]:
    if split_name not in dataset:
      continue
    save_split_knowledge(
      split_name=split_name,
      dataset=dataset,
      client=client,
      model=args.model,
      data_dir=args.data_dir,
      output_dir=args.output_dir,
      num_descriptions=args.num_descriptions,
      num_emotions=args.num_emotions,
    )


if __name__ == "__main__":
  main()
