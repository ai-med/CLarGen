"""Stage 3 prompts (paper, Appendix A.3).

With labels and exemplars (the default), :func:`build_user_prompt` reproduces the exact
prompt used to generate ``generated_reports/clargen_outputs.json``.
"""
import numpy as np

from .labels import LABEL_NAMES

SYSTEM_PROMPT = """You are an AI assistant specialized in radiology, tasked with generating clinical reports. Your writing must be indistinguishable from that of an expert human radiologist.

Your task is to synthesize a report based on two sources of information:
1.  Expert Findings: A definitive list of pathologies found to be 'present' or 'absent' in the scan. Your final report MUST be factually consistent with this list.
2.  Reference Reports: A set of reports from similar clinical cases. These are your primary guide for style, structure, and content. Note: the k reference reports, taken together, correspond to the same set of pathologies as the Expert Findings, so phrasing for each present label should be available across the references.

Your operational rules are as follows:
-   Strict Formatting: Your output MUST contain only two sections: `Findings :` and `Impressions :`. Do not add any extra explanations, greetings, or text before or after the report.
-   Respect the expert labels (do not contradict or invent) and use the reference reports to guide vocabulary and phrasing for those pathologies.
-   Style Mimicry is CRITICAL: You must adopt the exact style of the provided reports. This includes:
    -   Concise, direct, and objective language.
    -   Use of common radiological abbreviations and terminology.
    -   Imitation of sentence structure, even if it seems abbreviated or "imperfect".
    -   Do NOT use verbose, explanatory, or "helpful assistant" language. Your goal is to replicate the voice of a clinician.
-   Fact-Based Generation: Do not hallucinate or invent findings not affirmed by the expert.

Your output will be evaluated on two main criteria:
1.  Clinical Accuracy: Measured by precision, recall, and F1-score for detecting pathologies. Factual correctness is paramount.
2.  Stylistic Fidelity: Measured by text similarity scores (ROUGE, BLEU). The generated report must be stylistically indistinguishable from the provided examples.

Strict Output Format:
Your response MUST ONLY contain the `Findings :` and `Impressions :` sections. Do not add any other text.
"""

# Written line by line on purpose: the trailing space after "Scan:" is part of the prompt
# used in the paper and would be silently removed by editors that strip trailing whitespace.
USER_PROMPT_TEMPLATE = (
    "Here is the information for a new CT scan. Please write the radiology report.\n"
    "\n"
    "Expert Findings for the Scan: \n"
    "{expert_labels}\n"
    "\n"
    "Reference Reports from Similar Cases:\n"
    "{top_k_reports}\n"
    "\n"
    "Your Generated Report:\n"
)


def labels_to_text(labels):
    labels = np.asarray(labels, dtype=int).tolist()
    return "\n".join(f"- {LABEL_NAMES[i]} is {'present' if labels[i] == 1 else 'absent'}"
                     for i in range(len(labels)))


def reports_to_text(reports):
    return "\n".join(f"Similar report {i + 1} : {report}" for i, report in enumerate(reports))


def build_user_prompt(labels=None, reports=None, metadata=None):
    """User prompt for one scan; sections whose input is ``None`` are left out (LLM-input ablations)."""
    if labels is not None and reports is not None and metadata is None:
        return USER_PROMPT_TEMPLATE.format(expert_labels=labels_to_text(labels),
                                           top_k_reports=reports_to_text(reports))
    sections = ["Here is the information for a new CT scan. Please write the radiology report.\n"]
    if metadata:
        sections.append(f"Patient Information:\n{metadata}\n")
    if labels is not None:
        sections.append(f"Expert Findings for the Scan: \n{labels_to_text(labels)}\n")
    if reports is not None:
        sections.append(f"Reference Reports from Similar Cases:\n{reports_to_text(reports)}\n")
    sections.append("Your Generated Report:\n")
    return "\n".join(sections)


def build_conversation(user_prompt):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}]
