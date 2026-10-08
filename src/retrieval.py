"""Stage 2: retrieval of exemplar reports from the training-set database."""
import numpy as np
import torch
import torch.nn.functional as F

from .data import patient_id
from .metrics.crg import crg_against_database

STRATEGIES = ("label_match", "pathology_guided", "vol2text", "vol2vol")


def _cosine(query, bank):
    """Cosine similarity between one query embedding (D,) and each row of ``bank`` (n, D)."""
    return torch.cosine_similarity(query.unsqueeze(0).expand(bank.size(0), -1), bank, dim=1).numpy()


class Retriever:
    """Selects k exemplar reports for a query CT.

    Strategies
    ----------
    label_match       Used for all reported CLarGen results. Candidates whose label vector
                      equals the predicted labels are ranked by image-to-text cosine similarity;
                      if fewer than k exist, the whole database is ranked by CRG agreement with
                      the predicted labels and the top-k are ordered by (CRG, cosine similarity).
    pathology_guided  Clinical alignment score s_find (Eq. 2) combined with cosine similarity,
                      s_i = alpha * cos(v, t_i) + (1 - alpha) * s_find(i), followed by
                      coverage-aware greedy selection over findings with p_c >= 0.5 (Eq. 3).
    vol2text          k nearest reports by image-to-text cosine similarity (ablation "Vol<->Text").
    vol2vol           k nearest volumes by image-to-image cosine similarity (ablation "Vol<->Vol").

    Database entries from the query's own patient are never retrieved when
    ``exclude_same_patient`` is set (the CT-RATE train/valid splits are already disjoint).
    """

    def __init__(self, database, strategy="label_match", k=1, alpha=0.5, exclude_same_patient=True):
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown retrieval strategy '{strategy}', choose from {STRATEGIES}")
        self.strategy = strategy
        self.k = int(k)
        self.alpha = float(alpha)
        self.labels = database["labels"]
        self.texts = database["texts_latent"]
        self.images = database["images_latent"]
        self.patients = np.array([patient_id(n) for n in database["volume_names"]])
        self.exclude_same_patient = exclude_same_patient

    def _allowed(self, query_name):
        """Boolean mask of retrievable entries, or None when every entry is allowed."""
        if not self.exclude_same_patient or query_name is None:
            return None
        allowed = self.patients != patient_id(query_name)
        return None if allowed.all() else allowed

    def __call__(self, image_latent, probs, pred_labels, query_name=None):
        """Return the database indices of the k exemplars, best first."""
        allowed = self._allowed(query_name)
        image_latent = image_latent.detach().cpu().float()
        if self.strategy == "label_match":
            return self._label_match(image_latent, np.asarray(pred_labels, dtype=int), allowed)
        if self.strategy == "pathology_guided":
            return self._pathology_guided(image_latent, np.asarray(probs, dtype=np.float64), allowed)
        bank = self.texts if self.strategy == "vol2text" else self.images
        sims = F.normalize(bank, dim=1) @ F.normalize(image_latent, dim=0)
        sims = sims.numpy().astype(np.float64)
        if allowed is not None:
            sims[~allowed] = -np.inf
        return np.argsort(-sims, kind="stable")[:self.k].tolist()

    def _label_match(self, v, pred, allowed):
        k = self.k
        match = (self.labels == pred[None, :]).all(axis=1)
        if allowed is not None:
            match &= allowed
        exact_idx = np.flatnonzero(match)

        if len(exact_idx) >= k:
            sims = _cosine(v, self.texts[exact_idx])
            best = np.argsort(-sims)[:k]
            return [int(exact_idx[i]) for i in best]

        # Not enough exact matches: score the whole database with CRG, tie-break by cosine.
        crg = crg_against_database(pred, self.labels)
        if allowed is not None:
            crg = np.where(allowed, crg, -np.inf)
        k_eff = min(k, len(crg))
        top = np.argsort(-crg)[:k_eff]
        sims = _cosine(v, self.texts[top.tolist()])
        ranked = [(int(idx), float(crg[int(idx)]), float(sims[i])) for i, idx in enumerate(top)]
        ranked.sort(key=lambda t: (-t[1], -t[2]))
        return [t[0] for t in ranked][:k_eff]

    def _pathology_guided(self, v, p, allowed):
        L = self.labels.astype(np.float64)
        eps = 1e-8
        s_find = (L @ p) / (p.sum() + eps) - (L @ (1 - p)) / ((1 - p).sum() + eps)  # Eq. (2)
        cos = (F.normalize(self.texts, dim=1) @ F.normalize(v, dim=0)).numpy().astype(np.float64)
        score = self.alpha * cos + (1 - self.alpha) * s_find
        if allowed is not None:
            score[~allowed] = -np.inf

        targets = np.flatnonzero(p >= 0.5)
        weights = p[targets] / p[targets].sum() if len(targets) else np.zeros(0)
        covered = np.zeros(len(targets), dtype=bool)
        selected = []
        n_candidates = int(np.isfinite(score).sum())
        for _ in range(min(self.k, n_candidates)):
            gain = score + L[:, targets[~covered]] @ weights[~covered]  # Eq. (3)
            gain[selected] = -np.inf
            best = int(np.argmax(gain))
            selected.append(best)
            covered |= L[best, targets] == 1
        return selected
