"""Evaluate generated reports (paper, Sec. 4 "Evaluation Metrics").

  (a) NLP fidelity         ROUGE-L, BLEU (mean of BLEU-1..4), BLEU-4, METEOR
  (b) clinical correctness macro F1 / precision / recall, CRG (RadBERT-extracted labels)
  (c) diversity            Distinct-1/2, effV1/effV2, template concentration, NBS (--nbs)
  (d) survival             Surv_rare, Surv_common, Delta (prevalence quartiles of the findings)

Writes ``metrics_<subset>.csv`` (one row per model), ``per_class_<subset>.csv`` and
``templates_<subset>.csv`` to the output directory.
"""
import os

import pandas as pd

from src.config import base_parser, resolve_config, resolve_path
from src.data import load_reference_reports, load_reports
from src.labels import LABEL_NAMES
from src.metrics.clinical import RadBertLabeler, clinical_metrics
from src.metrics.diversity import diversity_metrics, self_bleu
from src.metrics.nlp import compute_nlp_metrics
from src.metrics.templates import template_metrics

SHORTCUTS = {"subset": "evaluation.subset", "output_dir": "evaluation.output_dir",
             "radbert": "evaluation.radbert_checkpoint"}

COLUMNS = ["n", "ROUGE-L", "BLEU", "BLEU-4", "METEOR",
           "F1", "Precision", "Recall", "CRG", "Specificity", "Accuracy",
           "Distinct-1", "effV1", "Distinct-2", "effV2", "NBS", "Self-BLEU-4",
           "Surv_rare", "Surv_common", "Delta",
           "Unique", "Top-1", "Top-5", "T80"]


def parse_args():
    parser = base_parser(__doc__)
    parser.add_argument("--reports", nargs="+", metavar="NAME=PATH",
                        help="Report files to evaluate (default: evaluation.reports in the config).")
    parser.add_argument("--subset", choices=["all", "common"],
                        help="all: every volume of each file; common: volumes shared by all files.")
    parser.add_argument("--radbert", help="Path to CT-RATE's RadBertClassifier.pth.")
    parser.add_argument("--nbs", action="store_true", help="Also compute the Normal Bias Similarity.")
    parser.add_argument("--self_bleu", action="store_true", help="Also compute Self-BLEU-4 (slow).")
    parser.add_argument("--output_dir", help="Where to write the result tables.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = resolve_config(args, SHORTCUTS)["evaluation"]
    files = dict(item.split("=", 1) for item in args.reports) if args.reports else cfg["reports"]

    references = load_reference_reports(resolve_path(cfg["references"]))
    outputs = {name: load_reports(resolve_path(path)) for name, path in files.items()}
    if cfg["subset"] == "common":
        shared = set(references).intersection(*outputs.values())
        outputs = {name: {v: r for v, r in reports.items() if v in shared} for name, reports in outputs.items()}
        print(f"Common subset: {len(shared)} volumes")

    labeler = RadBertLabeler(resolve_path(cfg["radbert_checkpoint"]), batch_size=cfg["radbert_batch_size"])
    if args.nbs:
        from src.metrics.nbs import TextEncoder, nbs, normal_reference

        encoder = TextEncoder(cfg["nbs_encoder"])
        volumes = sorted(set().union(*outputs.values()) & set(references))
        ref_texts = [references[v] for v in volumes if references[v]]
        normal_emb, n_normal = normal_reference(ref_texts, labeler(ref_texts), encoder)
        print(f"NBS reference: mean of {n_normal} fully normal reference reports")

    rows, per_class_rows, template_rows = {}, [], []
    for name, reports in outputs.items():
        print(f"\n=== {name} ({len(reports)} reports) ===")
        pairs = [(references[v], r) for v, r in reports.items() if references.get(v)]
        refs, hyps = [p[0] for p in pairs], [p[1] for p in pairs]
        texts = [r for r in reports.values() if r]

        row = {"n": len(pairs)}
        row.update(compute_nlp_metrics(refs, hyps))
        macro, per_class = clinical_metrics(labeler(hyps), labeler(refs))
        row.update(macro)
        row.update(diversity_metrics(texts))
        templates = template_metrics(texts)
        row.update({k: v for k, v in templates.items() if k in COLUMNS})
        if args.nbs:
            row["NBS"] = nbs(texts, normal_emb, encoder)
        if args.self_bleu:
            row["Self-BLEU-4"] = self_bleu(texts)
        rows[name] = row

        for finding, values in per_class.items():
            per_class_rows.append({"model": name, "finding": finding, **values})
        template_rows.append({"model": name, "count": templates["Top-1 count"],
                              "share": templates["Top-1"], "template": templates["Top-1 template"]})
        print(", ".join(f"{k}={row[k]:.3f}" for k in ("ROUGE-L", "BLEU", "F1", "Recall", "CRG", "effV2")))

    out_dir = resolve_path(cfg["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    table = pd.DataFrame.from_dict(rows, orient="index")
    table = table.reindex(columns=[c for c in COLUMNS if c in table.columns])
    table.to_csv(os.path.join(out_dir, f"metrics_{cfg['subset']}.csv"), float_format="%.6f")
    per_class = pd.DataFrame(per_class_rows)
    per_class["finding"] = pd.Categorical(per_class["finding"], LABEL_NAMES, ordered=True)
    per_class.sort_values(["model", "finding"]).to_csv(
        os.path.join(out_dir, f"per_class_{cfg['subset']}.csv"), index=False, float_format="%.6f")
    pd.DataFrame(template_rows).to_csv(os.path.join(out_dir, f"templates_{cfg['subset']}.csv"), index=False)

    print("\n" + table.drop(columns=["n"]).round(3).to_string())
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
