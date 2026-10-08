"""Loading of the precomputed HLIP databases, CT-RATE reports and generated-report files."""
import json
import re

import numpy as np
import pandas as pd
import torch

from .labels import NUM_CLASSES

DATABASE_KEYS = ("volume_names", "images_latent", "texts_latent", "labels", "reports")


def load_database(path):
    """Load a CLarGen database (``rag_database_{train,valid}.pt``).

    The file is a dict with, for M volumes:
      - ``volume_names``  list[str]            CT-RATE volume ids, e.g. ``train_42_a_1``
      - ``images_latent`` (M, 512) tensor      HLIP image embeddings (joint image-text space)
      - ``texts_latent``  (M, 512) tensor      HLIP text embeddings of the reports
      - ``labels``        (M, 18) int array     CT-RATE multi-abnormality labels
      - ``reports``       list[str]            report text ("Findings: ... Impressions : ...")
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)
    missing = [k for k in DATABASE_KEYS if k not in raw]
    if missing:
        raise KeyError(f"{path} is missing keys {missing}")
    db = {
        "volume_names": list(raw["volume_names"]),
        "images_latent": torch.as_tensor(raw["images_latent"]).float(),
        "texts_latent": torch.as_tensor(raw["texts_latent"]).float(),
        "labels": np.stack([np.asarray(x, dtype=np.int64) for x in raw["labels"]]),
        "reports": list(raw["reports"]),
    }
    if db["labels"].shape[1] != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} labels per volume, got {db['labels'].shape[1]}")
    return db


def select_queries(volume_names, skip_suffixes=("_2",)):
    """Indices of the volumes to generate reports for.

    CT-RATE stores several reconstructions of the same scan (``..._1``, ``..._2``, ...),
    all sharing one report. Skipping the ``_2`` reconstructions leaves the 1,607
    validation volumes evaluated in the paper.
    """
    skip = tuple(skip_suffixes or ())
    return [i for i, name in enumerate(volume_names) if not (skip and name.endswith(skip))]


def patient_id(volume_name):
    """``valid_12_b_1`` -> ``valid_12`` (CT-RATE: split, patient, scan, reconstruction)."""
    parts = volume_name.replace(".nii.gz", "").split("_")
    return "_".join(parts[:2])


def load_metadata(csv_path):
    """Map volume name -> short patient description from CT-RATE's ``*_metadata.csv``."""
    df = pd.read_csv(csv_path)
    df["VolumeName"] = df["VolumeName"].astype(str).str.replace(".nii.gz", "", regex=False)
    sex_names = {"M": "Male", "F": "Female"}
    out = {}
    for _, row in df.iterrows():
        lines = []
        age = str(row.get("PatientAge", "")).strip()
        if age and age.lower() != "nan":
            digits = re.sub(r"\D", "", age)
            lines.append(f"- Age: {int(digits)} years" if digits else f"- Age: {age}")
        sex = str(row.get("PatientSex", "")).strip()
        if sex and sex.lower() != "nan":
            lines.append(f"- Sex: {sex_names.get(sex, sex)}")
        if lines:
            out[row["VolumeName"]] = "\n".join(lines)
    return out


def clean_volume_name(name):
    return str(name).strip().replace(".nii.gz", "")


def clean_report(text):
    return str(text).replace("<|eot_id|>", "").strip()


def load_reports(json_path):
    """Read a ``[{"volume_name", "prediction"}, ...]`` file into an ordered dict volume -> report."""
    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return {clean_volume_name(r["volume_name"]): clean_report(r.get("prediction", "")) for r in rows}


def save_reports(json_path, volume_names, reports, extra=None):
    rows = []
    for i, (name, report) in enumerate(zip(volume_names, reports)):
        row = {"volume_name": name, "prediction": report}
        if extra:
            row.update({key: values[i] if isinstance(values, list) else values
                        for key, values in extra.items()})
        rows.append(row)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=4)


def load_reference_reports(csv_path):
    """Reference reports used for all paper metrics, built from CT-RATE's ``valid_reports.csv``.

    Findings and impressions are concatenated as ``"Findings: <F> Impressions: <I>"``;
    missing fields ("Not given.") are left empty.
    """
    df = pd.read_csv(csv_path)
    df["VolumeName"] = df["VolumeName"].astype(str).str.replace(".nii.gz", "", regex=False)

    def clean(x):
        if pd.isna(x) or str(x) == "Not given.":
            return ""
        return str(x).strip()

    findings = df["Findings_EN"].apply(clean)
    impressions = df["Impressions_EN"].apply(clean)
    reference = ("Findings: " + findings + " " + "Impressions: " + impressions).str.strip()
    return dict(zip(df["VolumeName"], reference))
