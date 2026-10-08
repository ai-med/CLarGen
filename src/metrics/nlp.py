"""NLP fidelity: ROUGE-L, BLEU-1..4 (and their mean, reported as "BLEU" in the paper), METEOR."""
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer

BLEU_WEIGHTS = {
    "BLEU-1": (1.0, 0, 0, 0),
    "BLEU-2": (0.5, 0.5, 0, 0),
    "BLEU-3": (1 / 3, 1 / 3, 1 / 3, 0),
    "BLEU-4": (0.25, 0.25, 0.25, 0.25),
}


def _ensure_nltk_data():
    for resource in ("corpora/wordnet", "corpora/omw-1.4", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[1], quiet=True)


def compute_nlp_metrics(references, hypotheses):
    """Corpus averages of sentence-level scores over (reference, hypothesis) pairs.

    BLEU uses whitespace tokens, METEOR NLTK word tokens; pairs with an empty reference are skipped.
    """
    _ensure_nltk_data()
    smooth = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    totals = {name: 0.0 for name in ["ROUGE-L", "METEOR", *BLEU_WEIGHTS]}
    n = 0
    for ref, hyp in zip(references, hypotheses):
        if not ref:
            continue
        n += 1
        ref_tok, hyp_tok = ref.split(), hyp.split()
        for name, weights in BLEU_WEIGHTS.items():
            totals[name] += sentence_bleu([ref_tok], hyp_tok, weights=weights, smoothing_function=smooth)
        totals["ROUGE-L"] += rouge.score(ref, hyp)["rougeL"].fmeasure
        totals["METEOR"] += single_meteor_score(word_tokenize(ref), word_tokenize(hyp))
    scores = {name: value / max(n, 1) for name, value in totals.items()}
    scores["BLEU"] = sum(scores[name] for name in BLEU_WEIGHTS) / len(BLEU_WEIGHTS)
    return scores
