# metrics.py  (Step 7)
# The three retrieval metrics, written BY HAND (no library), so we can unit-test them.
#
# Every function receives `relevances` = a list of 0/1 in RANK ORDER.
#   relevances = [0, 1, 0, 1]  means: rank 1 wrong, rank 2 RIGHT, rank 3 wrong, rank 4 RIGHT.
#   (1 = the chunk contains the gold phrase, 0 = it does not)

import math


def recall_at_k(relevances, k):
    """Recall@k: did at least ONE relevant chunk appear in the top k?  1 = yes, 0 = no.
    (Each question has ONE gold passage, so 'found it' is the right meaning of recall.)
    Example: [0, 1, 0, 1], k=3  ->  1.0        [0, 0, 0, 1], k=3  ->  0.0
    """
    if sum(relevances[:k]) > 0:
        return 1.0
    return 0.0


def reciprocal_rank(relevances, k=None):
    """1 / (rank of the FIRST relevant chunk). 0 if none is found.
    Example: [0, 1, 0, 1] -> 1/2 = 0.5       [1, 0, 0] -> 1.0       [0, 0, 0] -> 0.0
    The average of this over all questions is called MRR.
    """
    top = relevances if k is None else relevances[:k]
    position = 1
    for rel in top:
        if rel > 0:
            return 1.0 / position
        position = position + 1
    return 0.0


def dcg_at_k(relevances, k):
    """Discounted Cumulative Gain: a relevant chunk counts less the lower it is ranked.
    DCG = sum of  rel / log2(position + 1)
    """
    total = 0.0
    position = 1
    for rel in relevances[:k]:
        total = total + rel / math.log2(position + 1)
        position = position + 1
    return total


def ndcg_at_k(relevances, total_relevant, k):
    """Normalised DCG = DCG / best possible DCG, so the result is between 0 and 1.
    total_relevant = how many relevant chunks exist in the whole knowledge base.
    Example: [0, 1, 0, 1], total_relevant=2, k=4
        DCG   = 1/log2(3) + 1/log2(5)  = 0.6309 + 0.4307 = 1.0616
        ideal = [1, 1]: 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309
        nDCG  = 1.0616 / 1.6309 = 0.651
    """
    best_list = [1] * min(total_relevant, k)
    best_dcg = dcg_at_k(best_list, k)
    if best_dcg == 0:
        return 0.0
    return dcg_at_k(relevances, k) / best_dcg


def mean(values):
    """Average of a list (0.0 if the list is empty)."""
    if len(values) == 0:
        return 0.0
    return sum(values) / len(values)

# ---------------------------------------------------------------------------
# Agreement between two labellers (used to compare the LLM judge with a human)
# ---------------------------------------------------------------------------

def percent_agreement(labels_a, labels_b):
    """Share of items where both labellers gave the SAME label.
    Example: [1,1,0,0], [1,0,0,0] -> 3 of 4 agree -> 0.75"""
    if len(labels_a) == 0:
        return None
    same = 0
    for a, b in zip(labels_a, labels_b):
        if a == b:
            same = same + 1
    return same / len(labels_a)


def cohen_kappa(labels_a, labels_b):
    """Cohen's kappa = agreement AFTER removing the agreement expected by pure luck.
        kappa = (observed - expected) / (1 - expected)
    1.0 = perfect, 0 = no better than chance, below 0 = worse than chance.
    Example: a=[1,1,0,0,1,0]  b=[1,0,0,0,1,1]
        observed = 4/6 = 0.667
        expected = P(both say 1) + P(both say 0) = 0.5*0.5 + 0.5*0.5 = 0.5
        kappa    = (0.667 - 0.5) / (1 - 0.5) = 0.333
    If expected agreement is 1 (everybody always says the same single label) kappa
    is undefined; by convention we return 1.0 when they agree fully, else 0.0."""
    n = len(labels_a)
    if n == 0:
        return None
    observed = percent_agreement(labels_a, labels_b)
    expected = 0.0
    for label in set(labels_a) | set(labels_b):
        expected = expected + (labels_a.count(label) / n) * (labels_b.count(label) / n)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)