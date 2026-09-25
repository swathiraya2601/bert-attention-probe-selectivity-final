from conllu import parse_incr
from pathlib import Path 

def load_sentences(filepath):
    """Yields one parsed sentence (list of token dicts) at a time."""
    with open(filepath, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            yield tokenlist


sentences = list(load_sentences("data/raw/ud_english_ewt_r2.18/en_ewt-ud-train.conllu"))
print(len(sentences))
print(sentences[0])

def clean_tokens(tokenlist):
    """Keep only tokens with a plain integer ID (drop multiword-token ranges)."""
    return [tok for tok in tokenlist if isinstance(tok["id"], int)]

    