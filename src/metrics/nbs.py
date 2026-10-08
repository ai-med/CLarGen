"""Normal Bias Similarity (NBS, paper Appendix A.2).

The normal reference e_N is the L2-normalised mean embedding of the ground-truth reports
whose 18 labels are all absent; NBS(r) = cos(phi(r), e_N), averaged over the test set.
phi is a mean-pooled, L2-normalised text encoder.
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class TextEncoder:
    def __init__(self, name, device=None, batch_size=32, max_length=512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(name, trust_remote_code=True).to(self.device).eval()
        self.batch_size = batch_size
        self.max_length = max_length

    @torch.no_grad()
    def __call__(self, texts):
        embeddings = []
        for start in tqdm(range(0, len(texts), self.batch_size), desc="NBS embeddings", leave=False):
            enc = self.tokenizer(texts[start:start + self.batch_size], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            embeddings.append(F.normalize(pooled, dim=1).cpu())
        return torch.cat(embeddings).numpy()


def normal_reference(reference_texts, reference_labels, encoder):
    normal = [t for t, y in zip(reference_texts, reference_labels) if not np.any(y)]
    if not normal:
        raise ValueError("No fully normal reference report found; cannot build the NBS reference.")
    mean = encoder(normal).mean(axis=0)
    return mean / np.linalg.norm(mean), len(normal)


def nbs(texts, reference_embedding, encoder):
    return float((encoder(texts) @ reference_embedding).mean())
