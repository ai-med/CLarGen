"""The 18 CT-RATE thoracic findings predicted by CLarGen (paper, Appendix A.1).

The order matches the columns of CT-RATE's ``*_predicted_labels.csv`` files, the
``labels`` arrays of the retrieval databases, and the RadBERT label extractor.
"""

LABEL_NAMES = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]

NUM_CLASSES = len(LABEL_NAMES)
