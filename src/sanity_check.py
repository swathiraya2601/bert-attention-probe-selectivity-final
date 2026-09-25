import json
from collections import Counter

def check_balance(path):
    labels = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            labels.append(r["label"])
    counts = Counter(labels)
    total = len(labels)
    print(f"Total pairs: {total}")
    print(f"Positive (label=1): {counts[1]} ({counts[1]/total:.1%})")
    print(f"Negative (label=0): {counts[0]} ({counts[0]/total:.1%})")

check_balance("data/processed/train_pairs.jsonl")

def spot_check(path, n=10):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))

    positives = [r for r in records if r["label"] == 1]
    import random
    sample = random.sample(positives, n)

    for r in sample:
        words = r["words"]
        dep_word = words[r["i"]]
        head_word = words[r["j"]]
        print(f"'{dep_word}' --[{r['deprel']}]--> '{head_word}'   (full sentence: {' '.join(words)})")

spot_check("data/processed/train_pairs.jsonl", n=10)

def deprel_distribution(path):
    counts = Counter()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["label"] == 1:
                counts[r["deprel"]] += 1

    total = sum(counts.values())
    print(f"{'deprel':<15} {'count':>8} {'pct':>8}")
    for label, count in counts.most_common():
        print(f"{label:<15} {count:>8} {count/total:>7.2%}")
    print(f"\nTotal unique deprels: {len(counts)}")

deprel_distribution("data/processed/train_pairs.jsonl")