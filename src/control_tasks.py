# src/control_tasks.py
import json
import random
from collections import Counter, defaultdict
import numpy as np


def build_words_lookup(pairs_path):
    """sent_idx -> words, needed to look up which word is at position i/j."""
    lookup = {}
    with open(pairs_path) as f:
        for line in f:
            r = json.loads(line)
            if r["sent_idx"] not in lookup:
                lookup[r["sent_idx"]] = r["words"]
    return lookup


def build_vocab(words_lookup):
    """All unique word types (lowercased) seen in this lookup."""
    vocab = set()
    for words in words_lookup.values():
        vocab.update(w.lower() for w in words)
    return vocab


# ---------- Existence control task ----------

def build_existence_control_vocab(vocab, seed=42, n_classes=2):
    """Assigns each word type a random arbitrary class (0 or 1), unrelated to position."""
    rng = random.Random(seed)
    return {word: rng.randrange(n_classes) for word in vocab}


def get_control_existence_label(words, i, j, control_vocab):
    class_i = control_vocab.get(words[i].lower())
    class_j = control_vocab.get(words[j].lower())
    if class_i is None or class_j is None:
        return 0
    return 1 if class_i == class_j else 0


# ---------- Type control task ----------

def build_type_control_vocab(vocab, deprel_distribution, seed=42):
    """deprel_distribution: list of (label, probability) pairs summing to 1."""
    rng = random.Random(seed)
    labels, probs = zip(*deprel_distribution)
    return {word: rng.choices(labels, weights=probs, k=1)[0] for word in vocab}


def get_control_type_label(words, i, control_vocab):
    return control_vocab.get(words[i].lower(), "other")


def compute_deprel_distribution(pairs_path, label_set):
    """Empirical distribution over the same top-20+other label set used for the real type probe."""
    counts = Counter()
    with open(pairs_path) as f:
        for line in f:
            r = json.loads(line)
            if r["label"] == 1:
                for dep, head, deprel in [(r["i"], r["j"], r.get("deprel"))]:
                    pass
    return counts  # placeholder, replaced below by a simpler direct approach


def compute_deprel_distribution_from_type_labels(mapped_labels):
    """mapped_labels: the already-top-N-mapped deprel labels (with 'other') from build_type_labels."""
    counts = Counter(l for l in mapped_labels if l is not None)
    total = sum(counts.values())
    return [(label, count / total) for label, count in counts.items()]


# ---------- Relabel a features file with control labels ----------

def relabel_with_control(features_path, out_path, words_lookup,
                          existence_control_vocab, type_control_vocab):
    with open(features_path) as f, open(out_path, "w") as out_f:
        for line in f:
            r = json.loads(line)
            words = words_lookup[r["sent_idx"]]
            i, j = r["i"], r["j"]

            control_exist = get_control_existence_label(words, i, j, existence_control_vocab)
            control_type = get_control_type_label(words, i, type_control_vocab)

            r["control_label"] = control_exist
            r["control_deprel"] = control_type
            out_f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    from probes import load_features, build_type_labels

    # 1. Build word lookup and vocab from TRAIN pairs only
    train_words_lookup = build_words_lookup("data/processed/train_pairs.jsonl")
    test_words_lookup = build_words_lookup("data/processed/test_pairs.jsonl")
    vocab = build_vocab(train_words_lookup)

    # 2. Build existence control vocab (train vocab only, applied everywhere)
    existence_control_vocab = build_existence_control_vocab(vocab, seed=42)

    # 3. Build type control vocab — need the real label distribution first
    _, _, y_type_train = load_features("data/processed/train_features.jsonl")
    mapped_train, label_set = build_type_labels(y_type_train, top_n=20)
    deprel_dist = compute_deprel_distribution_from_type_labels(mapped_train)
    type_control_vocab = build_type_control_vocab(vocab, deprel_dist, seed=42)

    # 4. Relabel train and test features files with control labels
    relabel_with_control(
        "data/processed/train_features.jsonl", "data/processed/train_features_control.jsonl",
        train_words_lookup, existence_control_vocab, type_control_vocab
    )
    relabel_with_control(
        "data/processed/test_features.jsonl", "data/processed/test_features_control.jsonl",
        test_words_lookup, existence_control_vocab, type_control_vocab
    )

    print("Control-relabeled files written.")
    print("Vocab size:", len(vocab))
    print("Existence control vocab size:", len(existence_control_vocab))
    print("Type control vocab size:", len(type_control_vocab))