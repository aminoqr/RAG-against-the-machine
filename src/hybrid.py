"""Reciprocal Rank Fusion: combine multiple ranked lists into one
without needing their scores to be on the same scale. BM25 scores and
cosine similarities live in completely different ranges, so averaging
them directly is meaningless -- but each list's *rank order* is always
comparable, which is what RRF uses instead."""

from src.models import MinimalSource

RRF_K = 60

# Equal-weight fusion (1:1) measurably *hurt* recall on this corpus --
# see README "Retrieval method": tested against the labeled datasets,
# 1:1 dropped docs Recall@5 from 86.0% (BM25 alone) to 69.0%. BM25 here
# is unusually strong (tuned k1/b, boosted with file-path tokens), so
# a general-purpose semantic model votes with a much weaker signal.
# Weighting BM25 higher recovers most of that loss (10:1 -> 83.0%
# docs, 80.8% code, matching BM25 alone on code) without dropping
# semantic search's actual value: catching a paraphrase BM25 misses
# entirely. Not a substitute for BM25 here -- a corrective on top.
BM25_WEIGHT = 10.0
SEMANTIC_WEIGHT = 1.0


def reciprocal_rank_fusion(
    *ranked_lists: list[MinimalSource],
    weights: tuple[float, ...] | None = None,
    k: int = RRF_K,
) -> list[MinimalSource]:
    """Fuse any number of already-ranked (best first) source lists into
    one, using each item's rank in each list rather than its raw score.

    fused_score(item) = sum over lists containing item of
                         weight[list] / (k + rank)

    Args:
        *ranked_lists: any number of ranked lists (e.g. BM25's and the
            semantic index's, best first).
        weights: one weight per list, same order as ranked_lists.
            Defaults to equal weight (1.0 each) if not given. Unequal
            weighting matters when the rankers aren't equally
            trustworthy -- see BM25_WEIGHT/SEMANTIC_WEIGHT above.
        k: RRF's smoothing constant -- higher values shrink the gap
            between a rank-1 hit and a rank-20 hit.

    Returns:
        A single de-duplicated list, ranked by fused score descending.
        An item present in multiple input lists keeps its combined
        (weighted sum) score, so agreement between rankers is rewarded.
    """
    if weights is None:
        weights = tuple(1.0 for _ in ranked_lists)

    scores: dict[tuple[str, int, int], float] = {}
    items: dict[tuple[str, int, int], MinimalSource] = {}

    for weight, ranked in zip(weights, ranked_lists):
        for rank, source in enumerate(ranked, start=1):
            key = (
                source.file_path,
                source.first_character_index,
                source.last_character_index,
            )
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            items[key] = source

    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [items[key] for key in ranked_keys]
