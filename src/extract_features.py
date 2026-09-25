# src/extract_features.py
import json
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModel

from config import MODEL_NAME, MODEL_REVISION

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, use_fast=True)
model = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, output_attentions=True)
model.eval()


def tokenize_with_alignment(words, tok=None):
    """words: list of UD tokens for one sentence.
    Returns: encoding, word_to_subword (dict: word idx -> list of subword indices)"""
    tok = tok or tokenizer
    encoding = tok(words, is_split_into_words=True, return_tensors="pt", truncation=True)
    word_ids = encoding.word_ids(batch_index=0)

    word_to_subword = {}
    for subword_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        word_to_subword.setdefault(word_idx, []).append(subword_idx)
    return encoding, word_to_subword


def extract_all_features(pairs_path, out_path, model=model, tokenizer=tokenizer):
    """Groups pairs by sentence, runs BERT once per sentence, writes out
    one JSON record per pair with its 144-dim attention feature vector."""
    pairs_by_sent = defaultdict(list)
    with open(pairs_path) as f:
        for line in f:
            r = json.loads(line)
            pairs_by_sent[r["sent_idx"]].append(r)

    n_written = 0
    with open(out_path, "w") as out_f:
        for i, (sent_idx, records) in enumerate(pairs_by_sent.items()):
            words = records[0]["words"]
            encoding, word_to_subword = tokenize_with_alignment(words, tokenizer)

            with torch.no_grad():
                outputs = model(**encoding)
            attentions = outputs.attentions  # tuple of 12 tensors [1, 12heads, seq, seq]

            for r in records:
                wi, wj = r["i"], r["j"]
                subwords_i = word_to_subword.get(wi, [])
                subwords_j = word_to_subword.get(wj, [])
                if not subwords_i or not subwords_j:
                    continue  # word got truncated out of a very long sentence

                vec = []
                for layer_attn in attentions:
                    layer_attn = layer_attn[0]  # [12heads, seq, seq]
                    for head in range(layer_attn.shape[0]):
                        vals = layer_attn[head, subwords_i, :][:, subwords_j]
                        vec.append(vals.mean().item())

                out_f.write(json.dumps({
                    "sent_idx": sent_idx, "i": wi, "j": wj,
                    "label": r["label"], "deprel": r["deprel"],
                    "features": vec
                }) + "\n")
                n_written += 1

            if i % 500 == 0:
                print(f"Processed sentence {i} / {len(pairs_by_sent)}")

    print(f"Done. Wrote {n_written} feature records to {out_path}")


if __name__ == "__main__":
    # extract_all_features("data/processed/dev_pairs.jsonl", "data/processed/dev_features.jsonl")
    extract_all_features("data/processed/train_pairs.jsonl", "data/processed/train_features.jsonl")
    extract_all_features("data/processed/test_pairs.jsonl", "data/processed/test_features.jsonl")