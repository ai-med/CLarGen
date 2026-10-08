"""Train the Stage-1 pathology detector on frozen HLIP embeddings (paper, Sec. 4).

Focal loss (gamma = 2) with per-class weights alpha_c = 1 - prevalence_c, AdamW
(lr 6e-5, weight decay 0.008), cosine schedule with linear warm-up, 20 epochs, batch 64,
bf16 autocast on GPU. A patient-level hold-out of the training set is used to tune the
per-class thresholds; the model is then evaluated on the query set (CT-RATE validation).

The paper's released detector is ``data/hlip_classifier_with_focal.pt`` with
``data/thresholds_auc.npy``; this script retrains a detector from scratch.
"""
import json
import math
import os

import numpy as np
import torch

from src.config import base_parser, resolve_config, resolve_path
from src.data import load_database, patient_id, select_queries
from src.detector import binarize, build_detector, focal_loss, predict_proba, tune_thresholds
from src.labels import LABEL_NAMES
from src.metrics.clinical import per_class_metrics

SHORTCUTS = {"detector": "detector.type", "epochs": "training.epochs", "output_dir": "training.output_dir"}


def parse_args():
    parser = base_parser(__doc__)
    parser.add_argument("--detector", choices=["linear", "lqt"], help="Detector architecture.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output_dir", help="Where to save <type>_detector.pt and <type>_thresholds.npy.")
    return parser.parse_args()


def patient_holdout(volume_names, fraction, seed):
    patients = sorted({patient_id(n) for n in volume_names})
    rng = np.random.default_rng(seed)
    held = set(rng.choice(patients, size=int(round(fraction * len(patients))), replace=False))
    mask = np.array([patient_id(n) in held for n in volume_names])
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def macro_f1(pred, true):
    return float(np.mean([row["F1"] for row in per_class_metrics(pred, true).values()]))


def main():
    args = parse_args()
    cfg = resolve_config(args, SHORTCUTS)
    det_cfg, train_cfg = cfg["detector"], cfg["training"]
    torch.manual_seed(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_db = load_database(resolve_path(train_cfg["database"]))
    train_idx, holdout_idx = patient_holdout(train_db["volume_names"], train_cfg["holdout_fraction"],
                                             train_cfg["seed"])
    X = train_db["images_latent"]
    Y = torch.as_tensor(train_db["labels"], dtype=torch.float32)
    print(f"Train: {len(train_idx)} volumes | threshold hold-out: {len(holdout_idx)} volumes")

    model = build_detector(det_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Detector '{det_cfg['type']}': {n_params:,} trainable parameters")

    prevalence = Y[train_idx].mean(0).clamp(1e-3, 1 - 1e-3)
    alpha = (1 - prevalence).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    steps_per_epoch = math.ceil(len(train_idx) / train_cfg["batch_size"])
    total_steps = steps_per_epoch * train_cfg["epochs"]
    warmup = max(1, int(train_cfg["warmup_fraction"] * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: s / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))))

    train_idx_t = torch.as_tensor(train_idx)
    for epoch in range(train_cfg["epochs"]):
        model.train()
        order = train_idx_t[torch.randperm(len(train_idx_t))]
        running = 0.0
        for start in range(0, len(order), train_cfg["batch_size"]):
            batch = order[start:start + train_cfg["batch_size"]]
            x, y = X[batch].to(device), Y[batch].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
            loss = focal_loss(logits.float(), y, alpha, train_cfg["gamma"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(batch)
        print(f"epoch {epoch + 1:2d}/{train_cfg['epochs']}  focal loss {running / len(order):.4f}")

    holdout_probs = predict_proba(model, X[holdout_idx], device)
    thresholds = tune_thresholds(holdout_probs, train_db["labels"][holdout_idx])

    eval_db = load_database(resolve_path(train_cfg["eval_database"]))
    q = select_queries(eval_db["volume_names"], cfg["data"]["skip_suffixes"])
    probs = predict_proba(model, eval_db["images_latent"][q], device)
    true = eval_db["labels"][q]
    results = {
        "trainable_params": n_params,
        "macro_F1@0.5": macro_f1(binarize(probs, np.full(len(LABEL_NAMES), 0.5)), true),
        "macro_F1@tuned": macro_f1(binarize(probs, thresholds), true),
        "thresholds": dict(zip(LABEL_NAMES, thresholds.round(2).tolist())),
    }
    print(json.dumps(results, indent=2))

    out_dir = resolve_path(train_cfg["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, f"{det_cfg['type']}_detector.pt"))
    np.save(os.path.join(out_dir, f"{det_cfg['type']}_thresholds.npy"), thresholds)
    with open(os.path.join(out_dir, f"{det_cfg['type']}_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved detector and thresholds to {out_dir}")


if __name__ == "__main__":
    main()
