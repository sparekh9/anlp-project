#!/usr/bin/env python3

import argparse
import csv
import json
import os
from typing import Dict, List


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Convert the PrideMM CSV/Images layout into a Hateful Memes–style JSONL "
      "format with fields: id, img, label, text."
    )
  )
  parser.add_argument(
    "--csv-path",
    default=os.path.join("data", "PrideMM", "PrideMM.csv"),
    help="Path to PrideMM.csv.",
  )
  parser.add_argument(
    "--images-dir",
    default=os.path.join("data", "PrideMM", "Images"),
    help="Directory containing PrideMM images.",
  )
  parser.add_argument(
    "--output-dir",
    default=os.path.join("data", "PrideMM"),
    help="Directory to write train/validation/test JSONL files.",
  )
  parser.add_argument(
    "--id-from-name",
    action="store_true",
    help=(
      "If set, derive numeric ids from the image filename (e.g., img_123.png -> id 123). "
      "Otherwise, assign incremental ids per row."
    ),
  )
  return parser.parse_args()


def derive_id_from_name(name: str) -> int:
  """Extract an integer id from filenames like 'img_123.png'."""
  stem, _ = os.path.splitext(name)
  parts = stem.split("_")
  for part in reversed(parts):
    if part.isdigit():
      return int(part)
  raise ValueError(f"Could not derive integer id from name: {name}")


def convert_pridemm(csv_path: str, images_dir: str, output_dir: str, id_from_name: bool) -> None:
  """Convert PrideMM CSV to Hateful Memes–style JSONL splits."""
  splits: Dict[str, List[Dict]] = {"train": [], "validation": [], "test": []}

  with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    next_auto_id = 1

    for row in reader:
      split_raw = row.get("split", "").strip().lower()
      if split_raw not in {"train", "val", "validation", "test"}:
        continue

      if split_raw == "val":
        split = "validation"
      elif split_raw == "validation":
        split = "validation"
      else:
        split = split_raw

      name = row["name"].strip()
      text = row.get("text", "").strip()
      hate_str = row.get("hate", "").strip()

      if not name or not text or hate_str == "":
        continue

      try:
        label = int(hate_str)
      except ValueError:
        continue

      if id_from_name:
        try:
          example_id = derive_id_from_name(name)
        except ValueError:
          example_id = next_auto_id
          next_auto_id += 1
      else:
        example_id = next_auto_id
        next_auto_id += 1

      img_rel = os.path.join("PrideMM", "Images", name).replace(os.sep, "/")

      example = {
        "id": example_id,
        "img": img_rel,
        "label": label,
        "text": text,
      }

      splits[split].append(example)

  os.makedirs(output_dir, exist_ok=True)
  for split_name, records in splits.items():
    if not records:
      continue
    out_path = os.path.join(output_dir, f"{split_name}.jsonl")
    with open(out_path, "w", encoding="utf-8") as out_f:
      for record in records:
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} examples to {out_path}")


def main() -> None:
  args = parse_args()
  convert_pridemm(
    csv_path=args.csv_path,
    images_dir=args.images_dir,
    output_dir=args.output_dir,
    id_from_name=args.id_from_name,
  )


if __name__ == "__main__":
  main()
