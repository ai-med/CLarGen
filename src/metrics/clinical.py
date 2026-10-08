"""Clinical correctness: labels extracted from report text with CT-RATE's RadBERT classifier.

The same extractor labels the reference and the generated report; per-class precision,
recall (sensitivity), specificity, F1 and accuracy are macro-averaged, and CRG is computed
over the full label matrix.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ..labels import LABEL_NAMES
from .crg import crg_score

RADBERT = "zzxslp/RadBERT-RoBERTa-4m"


class RadBertClassifier(nn.Module):
    def __init__(self, n_classes=18):
        super().__init__()
        self.config = AutoConfig.from_pretrained(RADBERT)
        self.model = AutoModel.from_pretrained(RADBERT, config=self.config)
        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attn_mask):
        output = self.model(input_ids=input_ids, attention_mask=attn_mask)
        return self.classifier(output.pooler_output)


class RadBertLabeler:
    """Text -> 18 binary labels (sigmoid >= 0.5), reports truncated to 512 tokens."""

    def __init__(self, checkpoint, device=None, batch_size=32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(RADBERT, do_lower_case=True)
        self.model = RadBertClassifier(len(LABEL_NAMES))
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)
        self.model.to(self.device).eval()
        self.batch_size = batch_size
        self._cache = {}

    @torch.no_grad()
    def __call__(self, texts):
        new_texts = sorted({t for t in texts if t not in self._cache})
        for start in tqdm(range(0, len(new_texts), self.batch_size), desc="RadBERT labels", leave=False):
            batch = new_texts[start:start + self.batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", max_length=512,
                                 truncation=True, padding="max_length")
            logits = self.model(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
            preds = (torch.sigmoid(logits).cpu().numpy() >= 0.5).astype(np.int64)
            self._cache.update(zip(batch, preds))
        return np.stack([self._cache[t] for t in texts]) if texts else np.zeros((0, len(LABEL_NAMES)), int)


def per_class_metrics(pred, true, label_names=LABEL_NAMES):
    """Per-class metrics; classes with a single ground-truth value are skipped."""
    rows = {}
    for c, name in enumerate(label_names):
        y_pred, y_true = pred[:, c], true[:, c]
        if len(np.unique(y_true)) < 2:
            continue
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows[name] = {
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Prevalence": float(y_true.mean()),
            "GT+": int(y_true.sum()),
        }
    return rows


def clinical_metrics(pred, true):
    """Macro-averaged clinical metrics, CRG and prevalence-stratified survival."""
    per_class = per_class_metrics(pred, true)
    macro = {m: float(np.mean([row[m] for row in per_class.values()]))
             for m in ("F1", "Precision", "Recall", "Specificity", "Accuracy")}
    macro["CRG"] = float(crg_score(true, pred))
    macro.update(survival(per_class))
    return macro, per_class


def survival(per_class):
    """Surv_rare / Surv_common: mean recall over the bottom / top prevalence quartile of findings."""
    prevalence = np.array([row["Prevalence"] for row in per_class.values()])
    recall = np.array([row["Recall"] for row in per_class.values()])
    q1, q3 = np.percentile(prevalence, [25, 75])
    rare = float(recall[prevalence <= q1].mean())
    common = float(recall[prevalence >= q3].mean())
    return {"Surv_rare": rare, "Surv_common": common, "Delta": common - rare}
