import random 
from parse_treebank import clean_tokens

def extract_positive_pairs(tokenlist):
    """Returns list of (dependent_idx, head_idx, deprel) using 0-based indices."""
    tokens = clean_tokens(tokenlist)
    id_to_pos = {tok["id"]: i for i, tok in enumerate(tokens)}
    pairs = []
    for tok in tokens:
        if tok["head"] == 0 or tok["head"] is None:
            continue #skip root
        if tok["head"] not in id_to_pos:
            continue # skip if head was a filtered multiword token
        dep_idx = id_to_pos[tok["id"]]
        head_idx = id_to_pos[tok["head"]]
        pairs.append((dep_idx, head_idx, tok["deprel"]))

    return pairs

def extract_negative_pairs(tokenlist, positive_pairs, seed):
    tokens = clean_tokens(tokenlist)
    n = len(tokens)
    pos_set = {(d, h) for d, h, _ in positive_pairs}
    rng = random.Random(seed)
    negatives = []
    attempts = 0

    while len(negatives) < len(positive_pairs) and attempts < len(positive_pairs) * 20:
        i, j = rng.randrange(n), rng.randrange(n)
        attempts += 1
        if i == j:
            continue
        if (i,j) in pos_set or (j,i) in pos_set:
            continue
        negatives.append((i,j))
    return negatives