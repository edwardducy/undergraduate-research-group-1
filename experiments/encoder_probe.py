from pathlib import Path

from huggingface_hub import hf_hub_download
from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer

MODELS = {
    "mBERT cased (bert-base-multilingual-cased)": "bert-base-multilingual-cased",
    "mBERT uncased (bert-base-multilingual-uncased)": "bert-base-multilingual-uncased",
    "XLM-R base (xlm-roberta-base)": "xlm-roberta-base",
    "XLM-R large (xlm-roberta-large)": "xlm-roberta-large",
    "XLM-R XL (facebook/xlm-roberta-xl)": "facebook/xlm-roberta-xl",
    "XLM-R XXL (facebook/xlm-roberta-xxl)": "facebook/xlm-roberta-xxl",
    "DistilmBERT (distilbert-base-multilingual-cased)": "distilbert-base-multilingual-cased",
    "SEA-LION ModernBERT 300M (aisingapore/SEA-LION-ModernBERT-300M)": "aisingapore/SEA-LION-ModernBERT-300M",
    "SEA-LION ModernBERT 600M (aisingapore/SEA-LION-ModernBERT-600M)": "aisingapore/SEA-LION-ModernBERT-600M",
    "mmBERT small (jhu-clsp/mmBERT-small)": "jhu-clsp/mmBERT-small",
    "mmBERT base (jhu-clsp/mmBERT-base)": "jhu-clsp/mmBERT-base",
    "mDeBERTa-v3 base (microsoft/mdeberta-v3-base)": "microsoft/mdeberta-v3-base",
    "XLM-V base (facebook/xlm-v-base)": "facebook/xlm-v-base",
    "XLM-100 (xlm-mlm-100-1280)": "xlm-mlm-100-1280",
    "Glot500 base (cis-lmu/glot500-base)": "cis-lmu/glot500-base",
    "LaBSE (sentence-transformers/LaBSE)": "sentence-transformers/LaBSE",
}

SPM_ONLY = {"microsoft/mdeberta-v3-base"}

TEXTS = [
    line.strip()
    for path in Path("data/raw/yolanda").glob("*.txt")
    for line in path.read_text().splitlines()
    if line.strip()
]

def main() -> None:
    print(f"tweets loaded: {len(TEXTS)}")
    for name, repo in MODELS.items():
        if repo in SPM_ONLY:
            spm_path = hf_hub_download(repo, "spm.model")
            spm = SentencePieceProcessor(model_file=spm_path)
            print(name)
            total_words = sum(len(text.split()) for text in TEXTS)
            total_tokens = sum(len(spm.encode(text)) for text in TEXTS)
            print(f"  tokens per word: {total_tokens / total_words:.2f} ({total_tokens} tokens, {total_words} words)")
            print(f"  ma-evacuate -> {spm.encode('ma-evacuate', out_type=str)} ({len(spm.encode('ma-evacuate'))} pieces)")
            print()
            continue
        tokenizer = AutoTokenizer.from_pretrained(repo, use_fast=False)
        print(name)
        total_words = 0
        total_tokens = 0
        for text in TEXTS:
            total_words += len(text.split())
            total_tokens += len(tokenizer.encode(text, add_special_tokens=False))
        fertility = total_tokens / total_words
        print(f"  tokens per word: {fertility:.2f} ({total_tokens} tokens, {total_words} words)")
        segmented = tokenizer.tokenize("ma-evacuate")
        print(f"  ma-evacuate -> {segmented} ({len(segmented)} pieces)")
        print()

if __name__ == "__main__":
    main()
