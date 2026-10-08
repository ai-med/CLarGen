"""Corpus-level lexical diversity: Distinct-n, effective n-gram vocabulary effV_n = 2^H_n, Self-BLEU-4."""
from collections import Counter

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from tqdm import tqdm


def _ngrams(text, n):
    tokens = text.split()
    return list(zip(*[tokens[i:] for i in range(n)])) if len(tokens) >= n else []


def ngram_counts(texts, n):
    counts = Counter()
    for text in texts:
        counts.update(_ngrams(text, n))
    return counts


def distinct_n(texts, n):
    """Unique n-grams / total n-grams over the corpus (Li et al., 2016)."""
    counts = ngram_counts(texts, n)
    total = sum(counts.values())
    return len(counts) / total if total else 0.0


def effective_vocabulary(texts, n):
    """effV_n = 2^{H_n}, H_n the Shannon entropy (bits) of the corpus n-gram distribution."""
    counts = ngram_counts(texts, n)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    p = np.array(list(counts.values()), dtype=np.float64) / total
    return float(2 ** (-(p * np.log2(p)).sum()))


def self_bleu(texts, n_samples=None, seed=0):
    """Mean BLEU-4 of each report against all others (O(N^2); optionally on a random subset)."""
    tokenized = [t.split() for t in texts if t.strip()]
    if n_samples and n_samples < len(tokenized):
        rng = np.random.default_rng(seed)
        tokenized = [tokenized[i] for i in rng.choice(len(tokenized), n_samples, replace=False)]
    smooth = SmoothingFunction().method1
    scores = []
    for i in tqdm(range(len(tokenized)), desc="Self-BLEU-4", leave=False):
        refs = tokenized[:i] + tokenized[i + 1:]
        if tokenized[i] and refs:
            scores.append(sentence_bleu(refs, tokenized[i], weights=(0.25,) * 4, smoothing_function=smooth))
    return float(np.mean(scores)) if scores else 0.0


def diversity_metrics(texts):
    texts = [t for t in texts if t.strip()]
    return {
        "Distinct-1": distinct_n(texts, 1),
        "effV1": effective_vocabulary(texts, 1),
        "Distinct-2": distinct_n(texts, 2),
        "effV2": effective_vocabulary(texts, 2),
    }
