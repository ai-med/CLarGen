"""Stage 1: multi-label pathology detection from frozen HLIP features."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .labels import NUM_CLASSES


class LinearHead(nn.Module):
    """Detection head used for all reported CLarGen results (``data/hlip_classifier_with_focal.pt``).

    ReLU -> Dropout -> Linear on the pooled 512-d HLIP image embedding (9,234 parameters).
    """

    def __init__(self, latent_dim=512, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        if x.dim() == 3:  # (B, N, D) token features -> pooled embedding
            x = x.mean(dim=1)
        return self.classifier(self.dropout(self.relu(x)))


class LatentQueryTransformer(nn.Module):
    """Latent Query Transformer (paper, Sec. 3, Stage 1).

    C learnable pathology queries attend to the frozen encoder output Z (B, N, D) through
    cross-attention and to each other through self-attention; each refined query is
    projected to the logit of its pathology. A pooled embedding (B, D) is treated as N = 1.
    """

    def __init__(self, latent_dim=512, num_classes=NUM_CLASSES, dim=256, heads=8, layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, dim)
        self.queries = nn.Parameter(torch.randn(num_classes, dim) * 0.02)
        layer = nn.TransformerDecoderLayer(d_model=dim, nhead=heads, dim_feedforward=4 * dim,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.head_weight = nn.Parameter(torch.randn(num_classes, dim) * 0.02)
        self.head_bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, z):
        if z.dim() == 2:
            z = z.unsqueeze(1)
        memory = self.input_proj(z)
        queries = self.queries.unsqueeze(0).expand(z.size(0), -1, -1)
        refined = self.norm(self.decoder(queries, memory))
        return (refined * self.head_weight).sum(dim=-1) + self.head_bias


def build_detector(cfg):
    """Instantiate the detector described by the ``detector`` config section."""
    if cfg["type"] == "linear":
        return LinearHead(cfg["latent_dim"], cfg["num_classes"], cfg["dropout"])
    if cfg["type"] == "lqt":
        lqt = cfg["lqt"]
        return LatentQueryTransformer(cfg["latent_dim"], cfg["num_classes"], dim=lqt["dim"],
                                      heads=lqt["heads"], layers=lqt["layers"], dropout=lqt["dropout"])
    raise ValueError(f"Unknown detector type '{cfg['type']}' (expected 'linear' or 'lqt')")


def load_detector(cfg, checkpoint, device="cpu"):
    model = build_detector(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device).eval()


def load_thresholds(path, num_classes=NUM_CLASSES):
    if path is None:
        return np.full(num_classes, 0.5)
    return np.asarray(np.load(path), dtype=np.float64)


@torch.no_grad()
def predict_proba(model, features, device="cpu", batch_size=1024):
    """Sigmoid probabilities (M, C) for a tensor of HLIP features (M, D) or (M, N, D)."""
    model.eval()
    probs = []
    for start in range(0, len(features), batch_size):
        batch = features[start:start + batch_size].float().to(device)
        probs.append(torch.sigmoid(model(batch)).cpu())
    return torch.cat(probs).numpy()


def binarize(probs, thresholds):
    """A finding is predicted present when its probability is strictly above the class threshold."""
    return (np.asarray(probs) > np.asarray(thresholds)[None, :]).astype(np.int64)


def focal_loss(logits, targets, alpha, gamma=2.0):
    """Class-weighted focal loss, Eq. (1) of the paper.

    ``alpha`` (C,) weights the positive term of each class and ``1 - alpha`` its negative
    term. Using ``alpha_c = 1 - prevalence_c`` gives rarer findings a larger weight.
    """
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # -log(p_t)
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * ce).sum(dim=1).mean()


def tune_thresholds(probs, targets, grid=None):
    """Per-class decision threshold maximising Youden's J (sensitivity + specificity - 1)."""
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2) if grid is None else np.asarray(grid)
    thresholds = np.full(probs.shape[1], 0.5)
    for c in range(probs.shape[1]):
        y = targets[:, c].astype(bool)
        if y.all() or not y.any():
            continue
        pred = probs[:, c][None, :] > grid[:, None]
        sens = (pred & y).sum(1) / y.sum()
        spec = (~pred & ~y).sum(1) / (~y).sum()
        thresholds[c] = grid[int(np.argmax(sens + spec - 1))]
    return thresholds
