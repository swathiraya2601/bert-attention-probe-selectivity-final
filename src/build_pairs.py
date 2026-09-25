import json 
from parse_treebank import load_sentences, clean_tokens
from build_pairs_helpers import extract_negative_pairs, extract_positive_pairs
from config import SEEDS

def build_dataset(conllu_path, out_path, seed):
    records = []
    for sent_idx, tokenlist in enumerate(load_sentences(conllu_path)):
        tokens = clean_tokens(tokenlist)
        words = [tok["form"] for tok in tokens]
        pos_pairs = extract_positive_pairs(tokenlist)
        neg_pairs = extract_negative_pairs(tokenlist, pos_pairs, seed = seed + sent_idx)

        for dep, head, deprel in pos_pairs:
            records.append({"sent_idx": sent_idx, "words": words, "i": dep, "j": head, "label": 1, "deprel": deprel})
        for i, j in neg_pairs:
            records.append({"sent_idx": sent_idx, "words": words, "i": i, "j": j, "label": 0, "deprel": None})
    
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} pairs ({sum(r['label'] for r in records)} positive) to {out_path}")

if __name__ == "__main__":
    splits = {
        "train": "data/raw/ud_english_ewt_r2.18/en_ewt-ud-train.conllu",
        "dev": "data/raw/ud_english_ewt_r2.18/en_ewt-ud-dev.conllu",
        "test": "data/raw/ud_english_ewt_r2.18/en_ewt-ud-test.conllu",
    }
    for split_name, conllu_path in splits.items():
        out_path = f"data/processed/{split_name}_pairs.jsonl"
        build_dataset(conllu_path, out_path, seed=SEEDS[0])