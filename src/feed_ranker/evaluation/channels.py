"""Label-based channel diagnosis, strictly downstream of recommendation execution."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from feed_ranker.evaluation.funnel import GainPool
from feed_ranker.recommendation.contracts import Recommendation


def candidate_recall(
    pool: GainPool, relevant: np.ndarray, results: list[Recommendation]
) -> pd.DataFrame:
    """Binary Recall for executed candidate sets, after FunnelEvaluator validation.

    Evaluate each budget's actual IDs: anchored allocations are not nested prefixes
    of one full-scope score order. Zero-positive users retain NaN and are counted.
    """
    if relevant.shape != pool.items.shape or len(results) != len(pool.users):
        raise ValueError("Relevance/results must align with the complete gain pool")
    rows = []
    for u, result in enumerate(results):
        lo, hi = pool.offsets[u : u + 2]
        positions = pd.Index(pool.items[lo:hi]).get_indexer(result.retrieved.items)
        if result.request.user_id != pool.users[u] or (positions < 0).any():
            raise ValueError("Candidate recall requires aligned users and eligible items")
        positives = int(relevant[lo:hi].sum())
        rows.append(
            {
                "user_id": int(pool.users[u]),
                "positives": positives,
                "binary_recall": float(relevant[lo + positions].sum() / positives)
                if positives
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class ChannelDiagnostics:
    """Join added/displaced candidates to outcomes and frozen channel rankings."""

    pool: GainPool
    relevant: np.ndarray
    predictions: np.ndarray
    source_ranks: dict[str, np.ndarray]

    def compare(
        self, baseline: list[Recommendation], results: list[Recommendation]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows, changed = [], []
        for u, (base, result) in enumerate(zip(baseline, results, strict=True)):
            lo, hi = self.pool.offsets[u : u + 2]
            index = pd.Index(self.pool.items[lo:hi])
            b = lo + index.get_indexer(base.retrieved.items)
            s = lo + index.get_indexer(result.retrieved.items)
            bf, sf = b[base.final_positions], s[result.final_positions]
            added, lost = np.setdiff1d(s, b), np.setdiff1d(b, s)
            intersection, union = np.intersect1d(b, s), np.union1d(b, s)
            ideal = self.pool.ideals[u]

            def utility(positions: np.ndarray) -> float:
                final = self.pool.order(self.predictions, positions)[: self.pool.k]
                return self.pool.dcg(self.pool.gains[final]) / ideal if ideal else np.nan

            fb, fs, fi, fu = (utility(p) for p in (b, s, intersection, union))
            positives = int(self.relevant[lo:hi].sum())
            row = {
                "user_id": int(self.pool.users[u]),
                "binary_recall": float(self.relevant[s].sum() / positives) if positives else np.nan,
                "positives": positives,
                "added": len(added),
                "lost": len(lost),
                "positive_added": int(self.relevant[added].sum()),
                "positive_lost": int(self.relevant[lost].sum()),
                "gain_added": float(self.pool.gains[added].sum()),
                "gain_lost": float(self.pool.gains[lost].sum()),
                "mf_final_items_removed": int(np.isin(bf, lost).sum()),
                "added_items_in_final": int(np.isin(sf, added).sum()),
                "final_items_changed": int((~np.isin(bf, sf)).sum()),
                # Exact two-order average decomposition, not a causal estimate.
                "removal_contribution": 0.5 * ((fi - fb) + (fs - fu)),
                "addition_contribution": 0.5 * ((fu - fb) + (fs - fi)),
                "union_ndcg_diagnostic": fu,
                "counterfactual_delta": fs - fb,
            }
            rows.append(row)
            for action, positions in (("added", added), ("lost", lost)):
                n = len(positions)
                if not n:
                    continue
                before_ranks = {int(p): i + 1 for i, p in enumerate(bf)}
                after_ranks = {int(p): i + 1 for i, p in enumerate(sf)}
                changed.append(
                    pd.DataFrame(
                        {
                            "user_id": np.full(n, self.pool.users[u]),
                            "video_id": self.pool.items[positions],
                            "action": action,
                            "gain": self.pool.gains[positions],
                            "positive": self.relevant[positions],
                            "duration_ms": self.pool.duration_ms[positions],
                            "prediction": self.predictions[positions],
                            "mf_final_rank": [before_ranks.get(int(p), 0) for p in positions],
                            "merged_final_rank": [after_ranks.get(int(p), 0) for p in positions],
                            **{
                                f"rank_{name}": ranks[positions]
                                for name, ranks in self.source_ranks.items()
                            },
                        }
                    )
                )
        return pd.DataFrame(rows), pd.concat(
            changed, ignore_index=True
        ) if changed else pd.DataFrame()
