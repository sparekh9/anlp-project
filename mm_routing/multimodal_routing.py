import torch
import torch.nn as nn

class MultimodalRouting(nn.Module):
    """
    Implements equations (1)–(3) from the paper.
    fi: (B, N, D)
    pi: (B, N)
    Wij: (N, D, C, Dc)
    """
    def __init__(self, num_concepts=4, num_iters=3):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_iters = num_iters

    def forward(self, fi, pi, W):
        """
        fi:   (B, N, D)
        pi:   (B, N)
        W:    (N, D, C, Dc)
        """
        B, N, D = fi.shape
        C = self.num_concepts

        # 1. Initialize concepts uniformly
        cj = torch.randn(B, C, W.shape[-1], device=fi.device)
        cj = cj / (cj.norm(dim=-1, keepdim=True) + 1e-6)

        for _ in range(self.num_iters):
            # project features:  (B,N,D) x (N,D,C,Dc) -> (B,N,C,Dc)
            votes = torch.einsum("bnd,ndcd->bncd", fi, W)

            # agreement scores: <vote_ij , c_j>
            sij = (votes * cj[:,None,:,:]).sum(dim=-1)  # (B,N,C)

            # routing coeffs r_ij
            rij = torch.softmax(sij, dim=-1)           # normalize over concepts C

            # concept update eq (2)
            weighted = pi[...,None] * rij              # (B,N,C)
            cj_new = torch.einsum("bnc,bncd->bcd", weighted, votes)

            cj = cj_new / (cj_new.norm(dim=-1, keepdim=True) + 1e-6)

        return cj, rij
