"""CLarGen inference: pathology detection -> pathology-guided retrieval -> LLM report synthesis.

With the default configuration this regenerates ``generated_reports/clargen_outputs.json``
(Table 1 of the paper). Use ``--dry_run`` to run Stages 1-2 on CPU and inspect the prompts.
"""
import json
import os

import torch
from tqdm import tqdm

from src.config import base_parser, resolve_config, resolve_path
from src.data import load_database, load_metadata, save_reports, select_queries
from src.detector import binarize, load_detector, load_thresholds, predict_proba
from src.labels import LABEL_NAMES
from src.prompts import build_user_prompt
from src.retrieval import Retriever

SHORTCUTS = {
    "k": "retrieval.k",
    "retrieval": "retrieval.strategy",
    "detector": "detector.type",
    "checkpoint": "detector.checkpoint",
    "thresholds": "detector.thresholds",
    "model": "generation.model",
    "output": "generation.output",
}


def parse_args():
    parser = base_parser(__doc__)
    parser.add_argument("--k", type=int, help="Number of retrieved exemplar reports.")
    parser.add_argument("--retrieval", choices=["label_match", "pathology_guided", "vol2text", "vol2vol"],
                        help="Stage-2 retrieval strategy.")
    parser.add_argument("--detector", choices=["linear", "lqt"], help="Stage-1 detector architecture.")
    parser.add_argument("--checkpoint", help="Stage-1 detector weights.")
    parser.add_argument("--thresholds", help="Per-class decision thresholds (.npy).")
    parser.add_argument("--model", help="HuggingFace id of the frozen LLM.")
    parser.add_argument("--output", help="Output JSON file.")
    parser.add_argument("--limit", type=int, help="Only process the first N query volumes.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run Stages 1-2 and write the prompts, without loading the LLM.")
    parser.add_argument("--save_prompts", help="Also write predicted labels, exemplars and prompts to this JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = resolve_config(args, SHORTCUTS)
    data_cfg, det_cfg, ret_cfg, prompt_cfg, gen_cfg = (
        cfg["data"], cfg["detector"], cfg["retrieval"], cfg["prompt"], cfg["generation"])

    # ---- data ------------------------------------------------------------------
    database = load_database(resolve_path(data_cfg["database"]))
    queries = load_database(resolve_path(data_cfg["queries"]))
    query_idx = select_queries(queries["volume_names"], data_cfg["skip_suffixes"])
    if args.limit:
        query_idx = query_idx[:args.limit]
    names = [queries["volume_names"][i] for i in query_idx]
    print(f"Database: {len(database['volume_names'])} reports | queries: {len(names)} volumes")

    metadata = {}
    if prompt_cfg["use_metadata"]:
        if not data_cfg["metadata_csv"]:
            raise ValueError("prompt.use_metadata requires data.metadata_csv")
        metadata = load_metadata(resolve_path(data_cfg["metadata_csv"]))

    # ---- Stage 1: pathology detection -------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = load_detector(det_cfg, resolve_path(det_cfg["checkpoint"]), device)
    thresholds = load_thresholds(resolve_path(det_cfg["thresholds"]), det_cfg["num_classes"])
    features = queries["images_latent"][query_idx]
    probs = predict_proba(detector, features, device)
    pred_labels = binarize(probs, thresholds)

    # ---- Stage 2: retrieval + prompt construction --------------------------------
    retriever = Retriever(database, ret_cfg["strategy"], ret_cfg["k"], ret_cfg["alpha"],
                          ret_cfg["exclude_same_patient"])
    prompts, records = [], []
    for j, name in enumerate(tqdm(names, desc="Retrieval")):
        idx = retriever(features[j], probs[j], pred_labels[j], query_name=name)
        exemplars = [database["reports"][i] for i in idx]
        prompt = build_user_prompt(
            labels=pred_labels[j] if prompt_cfg["use_labels"] else None,
            reports=exemplars if prompt_cfg["use_exemplars"] else None,
            metadata=metadata.get(name) if prompt_cfg["use_metadata"] else None,
        )
        prompts.append(prompt)
        records.append({
            "volume_name": name,
            "probabilities": {c: round(float(p), 4) for c, p in zip(LABEL_NAMES, probs[j])},
            "predicted_labels": [c for c, y in zip(LABEL_NAMES, pred_labels[j]) if y],
            "retrieved": [database["volume_names"][i] for i in idx],
            "prompt": prompt,
        })

    prompts_file = args.save_prompts or (os.path.splitext(resolve_path(gen_cfg["output"]))[0] + "_prompts.json"
                                         if args.dry_run else None)
    if prompts_file:
        os.makedirs(os.path.dirname(os.path.abspath(prompts_file)), exist_ok=True)
        with open(prompts_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"Saved prompts: {prompts_file}")
    if args.dry_run:
        return

    # ---- Stage 3: report synthesis -----------------------------------------------
    from src.generation import generate, load_llm

    print(f"Loading {gen_cfg['model']} with vLLM...")
    llm = load_llm(gen_cfg)
    reports = generate(llm, prompts, gen_cfg)

    output = resolve_path(gen_cfg["output"])
    os.makedirs(os.path.dirname(output), exist_ok=True)
    save_reports(output, names, reports, extra={"k": ret_cfg["k"]})
    print(f"Saved {len(reports)} reports: {output}")


if __name__ == "__main__":
    main()
