"""Rebuild styles/config/vocabularies/Thesis/accept.txt for Vale.

Collects technical tokens (digits, internal capitals, non-ASCII) from the
chapter sources under thesis/chapters plus author surnames from
references.bib, converts every entry to a case-insensitive pattern
([Tt]aglish) so Vale.Terms does not enforce a single capitalization, and adds
a curated domain list. Paths are anchored to the repository root, so the
script runs from any directory.
"""

import glob
import os
import re
from pathlib import Path

MANUAL = """Taglish Tagalog Cebuano Filipino Sinhala Odia Ulysses Vamco Mistral SeaLLM Kalahi
Cityscapes Xplore Mixup Spearman's Nemenyi Friedman Iman Davenport Nash multitask
scalarization subword tokenization tokenizer tokenizers deduplication pretraining triage
debiasing homoscedastic hyperparameters hoc feedforward nonurgent benchmarked overfitting
multilinguality hatespeech subconcept multilabel chatbots pytest psutil pynvml resample
resamples resampled masterlist codebook operationalizes delimitations tweet_id tagset
subwords subsampling regenerable lowercased backpropagation Fleiss bio_tags language_tags
Rai Paeng Haiyan Profiler Studentized postcondition Batayan unbatched Hiligaynon
microblogs uncentered subproblems hyperparameter ang ng mga sa na""".split()


def case_insensitive(word: str) -> str:
    # Documented vocabulary idiom: the (?i) flag makes the whole entry
    # case-insensitive (docs.vale.sh, "Vocabularies" — case-aware by default).
    return f"(?i){re.escape(word)}"


ROOT = Path(__file__).resolve().parent.parent

# Only genuinely unknown terms belong here: Vale masks vocabulary matches out
# of the text before other rules run, so ordinary words (even capitalized at
# sentence start) must stay out or phrase-level rules silently stop matching.
tokens = set()
for path in glob.glob(str(ROOT / "thesis" / "chapters" / "**" / "*.qmd"), recursive=True):
    text = open(path, encoding="utf-8").read()
    for tok in re.findall(r"[\w’'-]+", text):
        if tok.startswith("_") or len(tok) < 2 or not re.search(r"[A-Za-z]", tok):
            continue
        technical = (
            re.search(r"[0-9]", tok)
            or re.search(r"[^\x00-\x7F]", tok)
            or re.search(r"[A-Z]", tok[1:])  # internal capital: RoBERTa, fastText
            or re.search(r"^[A-Z]{2,}$", tok)  # acronym: GLUE, XNLI
        )
        if technical:
            tokens.add(tok)
            if tok.endswith(("'s", "’s")):
                tokens.add(tok[:-2])

bib = (ROOT / "thesis" / "references.bib").read_text(encoding="utf-8")
for block in re.findall(r"@\w+\{[^,]+,(.*?)\n\}", bib, re.S):
    m = re.search(r"author\s*=\s*\{([^}]+)\}", block)
    if m:
        for name in m.group(1).split(" and "):
            family = re.sub(r"[{}\\]", "", name.split(",")[0].strip())
            if len(family) > 1 and " " not in family:
                tokens.add(family)

entries = sorted({case_insensitive(t) for t in tokens if len(t) > 1} | {case_insensitive(m) for m in MANUAL})
out = str(ROOT / "styles" / "config" / "vocabularies" / "Thesis" / "accept.txt")
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, "w", encoding="utf-8").write("\n".join(entries) + "\n")
print(len(entries), "vocab entries")
print("samples:", entries[:2], entries[-2:])
