import numpy as np
import torch
from torch.utils.data import Dataset

class HatefulMemesClipDataset(Dataset):
    def __init__(self, model, npz_path, normalize=True, device='cpu'):
        """
        npz_path : str
            Path to hateful_memes_clip_embeddings_*.npz
        normalize : bool
            Whether to L2 normalize the embeddings
        device : 'cpu' or 'cuda'
            Device to store tensors on
        """
        data = np.load(npz_path)

        img = torch.tensor(data['image_embeddings'], dtype=torch.float32)
        txt = torch.tensor(data['text_embeddings'], dtype=torch.float32)
        lbl = torch.tensor(data['labels'], dtype=torch.float32)
        self.ids = data['ids']
        # Move to target device (cpu or cuda)
        self.image_emb = img.to(device)
        self.text_emb  = txt.to(device)

        model.eval()
        with torch.no_grad():
            combined_embs = model(self.image_emb, self.text_emb)
            combined_embs = combined_embs / (combined_embs.norm(dim = 1, keepdim=True) + 1e-8)
        model.train()
        self.image_text_emb = combined_embs.to(device)
        self.labels    = lbl.to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns:
            image_emb[idx], text_emb[idx], labels[idx], ids[idx]
        """
        return (
            self.image_emb[idx],
            self.text_emb[idx],
            self.image_text_emb[idx],
            self.labels[idx],
            self.ids[idx]
        )
