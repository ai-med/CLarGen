"""Report-level template concentration (paper, Appendix A.5.1, Tables A.1 and A.3)."""
import re
from collections import Counter


def normalize_report(text):
    """Lower-case and collapse whitespace; two reports are the same template if they are then identical."""
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def template_metrics(texts):
    """Unique: fraction of distinct normalised reports; Top-1/Top-5: share of outputs covered by
    the 1/5 most frequent reports; T80: number of distinct reports needed to cover 80%."""
    normalized = [normalize_report(t) for t in texts]
    counts = Counter(t for t in normalized if t).most_common()
    total = sum(c for _, c in counts)
    if total == 0:
        return {"Unique": 0.0, "Top-1": 0.0, "Top-5": 0.0, "T80": 0, "Top-1 count": 0, "Top-1 template": ""}
    cumulative, t80 = 0, len(counts)
    for rank, (_, count) in enumerate(counts, start=1):
        cumulative += count
        if cumulative / total >= 0.8:
            t80 = rank
            break
    return {
        "Unique": len(counts) / total,
        "Top-1": counts[0][1] / total,
        "Top-5": sum(c for _, c in counts[:5]) / total,
        "T80": t80,
        "Top-1 count": counts[0][1],
        "Top-1 template": counts[0][0],
    }
