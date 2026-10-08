"""Build a CLarGen database (``rag_database_<split>.pt``) from precomputed embeddings.

Needed only to use a different visual/text encoder (e.g. the encoder comparison of Table 2);
the HLIP databases used in the paper are provided (see README).

Inputs
  --embeddings  .pt file holding a dict with ``volume_names`` (list[str]), ``images_latent``
                (M, d) and ``texts_latent`` (M, d): image embeddings of the CT volumes and text
                embeddings of their reports, in a shared (contrastive) space
  --reports     CT-RATE ``<split>_reports.csv`` (VolumeName, Findings_EN, Impressions_EN)
  --labels      CT-RATE ``<split>_predicted_labels.csv`` (VolumeName + the 18 findings)
"""
import argparse

import numpy as np
import pandas as pd
import torch

from src.labels import LABEL_NAMES


def clean(x):
    return "" if pd.isna(x) or str(x) == "Not given." else str(x).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--reports", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    emb = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    names = [str(n).replace(".nii.gz", "") for n in emb["volume_names"]]

    reports = pd.read_csv(args.reports)
    reports.index = reports["VolumeName"].astype(str).str.replace(".nii.gz", "", regex=False)
    labels = pd.read_csv(args.labels)
    labels.index = labels["VolumeName"].astype(str).str.replace(".nii.gz", "", regex=False)

    missing = [n for n in names if n not in reports.index or n not in labels.index]
    if missing:
        raise KeyError(f"{len(missing)} volumes have no report or labels, e.g. {missing[:5]}")

    database = {
        "volume_names": names,
        "images_latent": torch.as_tensor(emb["images_latent"]).half(),
        "texts_latent": torch.as_tensor(emb["texts_latent"]).half(),
        "labels": labels.loc[names, LABEL_NAMES].to_numpy(dtype=np.int64),
        "reports": [f"Findings: {clean(reports.at[n, 'Findings_EN'])} "
                    f"Impressions : {clean(reports.at[n, 'Impressions_EN'])}" for n in names],
    }
    torch.save(database, args.output)
    print(f"Saved {len(names)} entries to {args.output}")


if __name__ == "__main__":
    main()
