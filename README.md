# ANLP Project

# Data

Refer to this [Google Drive](https://drive.google.com/drive/folders/1Xi7SrF3ZS8Kxb_NKKk_n-aM2irU1Z53e?usp=sharing) for data, models, embeddings, and results. 

## Intro
- Purpose: Detect hateful memes by combining CLIP image/text embeddings with a lightweight cross-modal attention classifier.
- Pipeline: generate LMM knowledge → build enriched CLIP embeddings → train classifier → run inference.

## Workflow
1) Generate LMM knowledge  
   - `python generate_knowledge.py`  
   - Writes `knowledge/lmm_knowledge_{train,val,test}.json` mapping meme id → `descriptions` (10) and `emotions` (10).
2) Generate embeddings  
   - `python generate_embeddings.py`  
   - Uses dataset + knowledge to emit `hateful_memes_clip_embeddings_{train,val,test}.npz` containing `image_embeddings`, `text_embeddings`, `desc_embeddings`, `emotion_embeddings`, `text_concat_embeddings`, `meme_concat_embeddings`, `labels`, `ids`, `valid_indices`.
3) Train  
   - `python training.py`  
   - Consumes `image_embeddings` + `text_concat_embeddings` for train/val and saves `best_model.pth` (includes `image_dim` and `text_dim` in `config`).
4) Predict  
   - `python predictions.py`  
   - Loads `best_model.pth` and test `image_embeddings` + `text_concat_embeddings`, writes `predictions.npz`, and prints accuracy/AUC/confusion matrix.
