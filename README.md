# ANLP Project

## Intro
- Purpose: Detect hateful memes by combining CLIP image/text embeddings with a lightweight cross‑modal attention classifier.
- Pipeline: generate CLIP embeddings → train classifier on embeddings → run inference to produce probabilities/predictions.

## Key Modules
- `generate_embeddings.py`: Downloads `neuralcatcher/hateful_memes`, reads images from `data/`, verifies each image, batches through `openai/clip-vit-large-patch14` (via `transformers.CLIPModel/CLIPProcessor`), L2‑normalizes image/text features, and saves `.npz` per split (`hateful_memes_clip_embeddings_{train|val|test}.npz`) plus `valid_indices_{split}.npz`. Uses GPU if available.
- `training.py`: Defines the model (`HatefulMemeClassifier`) with dual cross‑modal attention stacks (`CrossModalAttention`), GELU MLPs, residual + LayerNorm, fusion, and a binary classifier head. Trains on precomputed embeddings (train + val NPZs), reports AUC/Acc, uses AdamW + ReduceLROnPlateau, saves best checkpoint to `best_model.pth` containing epoch, state_dicts, val AUC, and config. Final block reloads the best checkpoint for a last val evaluation.
- `predictions.py`: Reloads `best_model.pth` (image/text dims assumed 768), runs sigmoid probabilities on test (or other) embeddings, saves `predictions.npz` with probs/preds/labels/ids, and prints accuracy/AUC/confusion matrix.

## Repro/usage quickstart
1) Generate embeddings: `python generate_embeddings.py` (ensure GPU if available for speed).
2) Train: `python training.py` (reads `hateful_memes_clip_embeddings_train.npz` and `_val.npz`, writes `best_model.pth`).
3) Predict: `python predictions.py` (reads `best_model.pth` and `hateful_memes_clip_embeddings_test.npz`, writes `predictions.npz` and prints metrics).
