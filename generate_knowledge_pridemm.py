#!/usr/bin/env python3

"""Generate per-meme descriptions and emotions for the PrideMM dataset.

This mirrors `generate_knowledge.py` but reads local JSONL metadata
(`data/PrideMM/{train,validation,test}.jsonl`) instead of loading a
Hugging Face dataset.
"""

import argparse
import base64
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

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
    lines = lines[1:]
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


def generate_knowledge_for_example(
  client: Together,
  model: str,
  data_dir: str,
  example,
  num_descriptions: int,
  num_emotions: int,
  max_retries: int = 3,
  base_delay: float = 1.0,
  backoff_factor: float = 2.0,
) -> Tuple[str, Optional[Dict[str, List[str]]], str, Optional[str]]:
  """Generate knowledge for a single PrideMM example with retries.

  Returns (meme_id, result, status, debug_message) where:
    - status is one of {"ok", "missing_image", "error"}
    - result is the parsed knowledge dict when status == "ok"
    - debug_message contains an error or raw response when status == "error"
  """
  meme_id = str(example["id"])
  img_path = os.path.join(data_dir, example["img"])
  meme_text = example["text"]

  if not is_valid_image(img_path):
    return meme_id, None, "missing_image", None

  result: Optional[Dict[str, List[str]]] = None
  raw_response: Optional[str] = None

  for attempt in range(max_retries + 1):
    try:
      result, raw_response = request_knowledge(
        client,
        model,
        img_path,
        meme_text,
        num_descriptions=num_descriptions,
        num_emotions=num_emotions,
      )
      break
    except Exception as e:
      if attempt >= max_retries:
        return meme_id, None, "error", f"{e}"
      delay = base_delay * (backoff_factor ** attempt)
      delay *= 0.5 + random.random()
      time.sleep(delay)

  if result is None:
    return meme_id, None, "error", raw_response

  return meme_id, result, "ok", None


def load_jsonl_split(jsonl_path: str) -> List[Dict]:
  """Load a PrideMM split from a JSONL file into a list of examples."""
  examples: List[Dict] = []
  if not os.path.exists(jsonl_path):
    print(f"JSONL file not found: {jsonl_path}")
    return examples

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

      if "id" not in obj or "img" not in obj or "text" not in obj:
        print(f"Skipping record without required fields in {jsonl_path}:{line_num}")
        continue

      examples.append(obj)

  return examples


def load_pridemm_dataset(metadata_dir: str) -> Dict[str, List[Dict]]:
  """Load PrideMM train/validation/test splits from local JSONL files."""
  dataset: Dict[str, List[Dict]] = {}
  for split_name in ["train", "validation", "test"]:
    filename = f"{split_name}.jsonl"
    jsonl_path = os.path.join(metadata_dir, filename)
    split_examples = load_jsonl_split(jsonl_path)
    if not split_examples:
      print(f"Warning: no examples loaded for split '{split_name}' from {jsonl_path}")
      continue
    dataset[split_name] = split_examples
    print(f"Loaded {len(split_examples)} {split_name} examples from {jsonl_path}")
  return dataset


def save_split_knowledge(
  split_name: str,
  dataset,
  client: Together,
  model: str,
  data_dir: str,
  output_dir: str,
  num_descriptions: int,
  num_emotions: int,
  output_prefix: str,
  max_concurrent_requests: int,
  max_retries: int,
) -> None:
  """Generate and save knowledge for a single split."""
  split = dataset[split_name]
  knowledge: Dict[str, Dict[str, List[str]]] = {}
  missing = 0
  errors = 0

  print(f"\nProcessing split: {split_name} ({len(split)} samples)")
  with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
    futures = [
      executor.submit(
        generate_knowledge_for_example,
        client,
        model,
        data_dir,
        example,
        num_descriptions,
        num_emotions,
        max_retries,
      )
      for example in split
    ]

    for future in tqdm(as_completed(futures), total=len(futures), desc=f"{split_name} split"):
      meme_id, result, status, debug_message = future.result()

      if status == "missing_image":
        missing += 1
        continue

      if status != "ok":
        errors += 1
        if debug_message:
          print(f"  Error for id {meme_id}: {debug_message}")
        continue

      if result is not None:
        knowledge[meme_id] = result

  split_suffix = "val" if split_name == "validation" else split_name
  os.makedirs(output_dir, exist_ok=True)
  output_path = os.path.join(output_dir, f"{output_prefix}_{split_suffix}.json")
  with open(output_path, "w") as f:
    json.dump(knowledge, f, indent=2)

  print(
    f"Saved {len(knowledge)} entries for {split_name} to {output_path}. "
    f"Skipped {missing} missing/invalid images and {errors} failed generations."
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Generate PrideMM meme descriptions and emotions via GPT using local JSONL metadata."
  )
  parser.add_argument(
    "--model",
    default="yiwei_64f1/Qwen/Qwen2.5-VL-72B-Instruct-7f41fb31",
    help="Together model to use for generation.",
  )
  parser.add_argument(
    "--data-dir",
    default="data",
    help="Root directory containing meme images (PrideMM paths are joined to this).",
  )
  parser.add_argument(
    "--metadata-dir",
    default=os.path.join("data", "PrideMM"),
    help="Directory containing PrideMM JSONL metadata files.",
  )
  parser.add_argument(
    "--output-dir",
    default="knowledge",
    help="Where to write JSON knowledge files.",
  )
  parser.add_argument(
    "--output-prefix",
    default="pridemm_lmm_knowledge",
    help="Filename prefix for knowledge JSON (e.g. 'pridemm_lmm_knowledge').",
  )
  parser.add_argument(
    "--num-descriptions",
    type=int,
    default=10,
    help="Number of descriptions per meme.",
  )
  parser.add_argument(
    "--num-emotions",
    type=int,
    default=10,
    help="Number of emotions per meme.",
  )
  parser.add_argument(
    "--max-concurrent-requests",
    type=int,
    default=4,
    help="Maximum number of concurrent Together API requests.",
  )
  parser.add_argument(
    "--max-retries",
    type=int,
    default=3,
    help="Maximum number of retries per meme on API errors.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  print("Loading PrideMM metadata from JSONL files...")
  dataset = load_pridemm_dataset(args.metadata_dir)

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
      output_prefix=args.output_prefix,
      max_concurrent_requests=args.max_concurrent_requests,
      max_retries=args.max_retries,
    )


if __name__ == "__main__":
  main()
