"""
Synthetic RecSys Data Generator

Shape validation and early testing (no dataset download required).

Usage:
    gen = SyntheticRecSysGenerator()
    batch = gen.get_batch(batch_size=32, task="ctr")
    gen.get_shapes()  # print all shapes
"""

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SyntheticConfig:
    """Scaled-down settings vs. production-scale dimensions."""
    num_users: int = 10_000
    num_items: int = 50_000
    num_creators: int = 5_000
    num_categories: int = 100
    max_seq_len: int = 50
    user_dense_dim: int = 16
    item_dense_dim: int = 8
    context_dim: int = 4
    max_num_neighbors: int = 20
    ctr_rate: float = 0.05
    watch_ratio_mean: float = 0.40
    watch_ratio_std: float = 0.25

    # Uplift task
    uplift_latent_dim: int = 16
    uplift_effect_scale: float = 0.3   # max synthetic uplift magnitude

    # Graph relation task (source-target via middle nodes)
    num_relation_bits: int = 8         # bitwise relation encoding dim
    max_middle_len: int = 10           # path length between source & target

    seed: int = 42


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class SyntheticRecSysGenerator:
    """
    Synthetic batch generator for RecSys model shape tests.

    Supported tasks:
        ctr            : CTR binary label       -> label [B]
        watch          : watch ratio             -> label [B]
        multitask      : CTR + watch             -> ctr_label [B], watch_label [B]
                          (reuse this batch for teacher-student distillation)
        seq            : next-item prediction     -> input_seq [B,T], target_id [B]
        retrieval      : two-tower                -> pos_item_id [B], neg_item_id [B,K]
        graph          : GraphSAGE / PinSage       -> neighbor_ids [B,K], neighbor_feats [B,K,D]
        uplift         : candidate exposure uplift -> label_control/treat, true_uplift (for self-validation)
        graph_relation : source-target relation    -> middle_relation_id [B,T], label (GPM-inspired)
    """

    def __init__(self, cfg: SyntheticConfig | None = None):
        self.cfg = cfg or SyntheticConfig()
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

    def get_batch(
        self,
        batch_size: int = 32,
        task: Literal[
            "ctr", "watch", "multitask", "seq",
            "retrieval", "graph", "uplift", "graph_relation"
        ] = "ctr",
    ) -> dict[str, torch.Tensor]:
        fn = {
            "ctr": self._ctr_batch,
            "watch": self._watch_batch,
            "multitask": self._multitask_batch,
            "seq": self._seq_batch,
            "retrieval": self._retrieval_batch,
            "graph": self._graph_batch,
            "uplift": self._uplift_batch,
            "graph_relation": self._graph_relation_batch,
        }[task]
        return fn(batch_size)

    # ------------------------------------------------------------------
    # Base batch (shared features)
    # ------------------------------------------------------------------

    def _base_batch(self, B: int) -> dict[str, torch.Tensor]:
        cfg = self.cfg

        user_id = torch.randint(0, cfg.num_users, (B,))
        item_id = torch.randint(0, cfg.num_items, (B,))
        category_id = torch.randint(0, cfg.num_categories, (B,))

        user_dense = torch.randn(B, cfg.user_dense_dim)
        item_dense = torch.randn(B, cfg.item_dense_dim)
        context = torch.randn(B, cfg.context_dim)

        # Heavy-tail history length (geometric distribution)
        raw_lens = np.clip(
            np.random.geometric(p=0.05, size=B),
            1, cfg.max_seq_len
        ).astype(np.int64)
        history_len = torch.from_numpy(raw_lens)

        history_ids = torch.zeros(B, cfg.max_seq_len, dtype=torch.long)
        for i, L in enumerate(raw_lens):
            history_ids[i, :L] = torch.randint(0, cfg.num_items, (int(L),))

        return {
            "user_id": user_id,          # [B]
            "item_id": item_id,          # [B]
            "category_id": category_id,  # [B]
            "user_dense": user_dense,    # [B, Ud]
            "item_dense": item_dense,    # [B, Id]
            "context": context,          # [B, Cd]
            "history_ids": history_ids,  # [B, T]
            "history_len": history_len,  # [B]
        }

    # ------------------------------------------------------------------
    # Core tasks
    # ------------------------------------------------------------------

    def _ctr_batch(self, B: int) -> dict[str, torch.Tensor]:
        batch = self._base_batch(B)
        batch["label"] = torch.bernoulli(torch.full((B,), self.cfg.ctr_rate))
        return batch

    def _watch_batch(self, B: int) -> dict[str, torch.Tensor]:
        batch = self._base_batch(B)
        raw = torch.normal(self.cfg.watch_ratio_mean, self.cfg.watch_ratio_std, (B,))
        batch["label"] = raw.clamp(0.0, 1.0)
        return batch

    def _multitask_batch(self, B: int) -> dict[str, torch.Tensor]:
        """
        Multi-task models (MMoE, PLE, etc.).
        Teacher-student distillation can reuse this batch:
          train teacher (MMoE) -> save teacher logits -> train student (SharedBottom)
          with labels + teacher logits jointly.
        """
        batch = self._base_batch(B)
        batch["ctr_label"] = torch.bernoulli(torch.full((B,), self.cfg.ctr_rate))
        raw = torch.normal(self.cfg.watch_ratio_mean, self.cfg.watch_ratio_std, (B,))
        batch["watch_label"] = raw.clamp(0.0, 1.0)
        return batch

    def _seq_batch(self, B: int) -> dict[str, torch.Tensor]:
        """SASRec, BERT4Rec, SIM."""
        batch = self._base_batch(B)
        T = self.cfg.max_seq_len
        full_seq = torch.randint(0, self.cfg.num_items, (B, T + 1))
        batch["input_seq"] = full_seq[:, :-1]
        batch["target_id"] = full_seq[:, -1]
        batch["neg_id"] = torch.randint(0, self.cfg.num_items, (B,))
        return batch

    def _retrieval_batch(self, B: int) -> dict[str, torch.Tensor]:
        """Two-tower / DSSM."""
        batch = self._base_batch(B)
        K = 4
        batch["pos_item_id"] = batch.pop("item_id")
        batch["neg_item_id"] = torch.randint(0, self.cfg.num_items, (B, K))
        return batch

    def _graph_batch(self, B: int) -> dict[str, torch.Tensor]:
        """GraphSAGE / PinSage."""
        batch = self._base_batch(B)
        K = self.cfg.max_num_neighbors
        batch["node_id"] = batch["item_id"]
        batch["neighbor_ids"] = torch.randint(0, self.cfg.num_items, (B, K))
        batch["neighbor_feats"] = torch.randn(B, K, self.cfg.item_dense_dim)
        raw_w = torch.rand(B, K)
        batch["edge_weights"] = raw_w / raw_w.sum(dim=-1, keepdim=True)
        return batch

    # ------------------------------------------------------------------
    # Applied research idea tasks
    # ------------------------------------------------------------------

    def _uplift_batch(self, B: int) -> dict[str, torch.Tensor]:
        """
        Dual-Path Attention Uplift.

        Core design: embed synthetic ground-truth uplift so you can self-validate
        whether the model actually recovers the signal.

        Generation logic:
          - user_latent, candidate_latent: random latent vectors
          - affinity = <user_latent, candidate_latent>  (source of true uplift)
          - true_uplift = sigmoid(affinity) * effect_scale  (range 0 .. effect_scale)
          - label_control = Bernoulli(base_prob)                    # no candidate exposure
          - label_treat   = Bernoulli(base_prob + true_uplift)       # with candidate exposure

        Validation (experiment):
          model_uplift = logit_treat - logit_control   (model prediction)
          true_uplift  = value generated above
          -> check correlation(model_uplift, true_uplift)
          -> higher correlation means the architecture recovers uplift well
        """
        batch = self._base_batch(B)
        cfg = self.cfg
        D = cfg.uplift_latent_dim

        candidate_id = torch.randint(0, cfg.num_items, (B,))
        candidate_dense = torch.randn(B, cfg.item_dense_dim)

        user_latent = torch.randn(B, D)
        candidate_latent = torch.randn(B, D)
        affinity = (user_latent * candidate_latent).sum(dim=-1) / (D ** 0.5)

        true_uplift = torch.sigmoid(affinity) * cfg.uplift_effect_scale
        base_prob = torch.sigmoid(torch.randn(B) * 0.5)

        label_control = torch.bernoulli(base_prob.clamp(0.0, 1.0))
        label_treat = torch.bernoulli((base_prob + true_uplift).clamp(0.0, 1.0))

        batch["candidate_id"] = candidate_id                # [B]
        batch["candidate_dense"] = candidate_dense           # [B, Id]
        batch["label_control"] = label_control               # [B]  no candidate exposure
        batch["label_treat"] = label_treat                   # [B]  with candidate exposure
        batch["true_uplift"] = true_uplift                   # [B]  ground truth (validation only, not for training)
        return batch

    def _graph_relation_batch(self, B: int) -> dict[str, torch.Tensor]:
        """
        Graph Relation Sequence + Cross-Attention (GPM-inspired).

        Encode relations on middle nodes connecting source_user and target_user
        as bitwise vectors, then pack into integers for embedding lookup.

        middle_relation_bits: [B, T, num_bits]  0/1 bitwise feature
        middle_relation_id:   [B, T]            packed int from bits
                                                 (usable with nn.Embedding(2**num_bits, dim))
        label: relation strength between source and target (synthetic, scales with path length)
        """
        cfg = self.cfg
        T = cfg.max_middle_len
        n_bits = cfg.num_relation_bits

        source_user_id = torch.randint(0, cfg.num_users, (B,))
        target_user_id = torch.randint(0, cfg.num_users, (B,))

        # Heavy-tail path length (shorter paths more common)
        raw_lens = np.clip(
            np.random.geometric(p=0.3, size=B), 1, T
        ).astype(np.int64)
        middle_len = torch.from_numpy(raw_lens)

        # Bitwise relation encoding: interaction type per middle edge
        # (e.g. follow / like / comment / co-view as multi-hot bits)
        middle_relation_bits = torch.zeros(B, T, n_bits, dtype=torch.float32)
        middle_relation_id = torch.zeros(B, T, dtype=torch.long)

        powers = torch.tensor([2 ** i for i in range(n_bits)], dtype=torch.long)
        for i, L in enumerate(raw_lens):
            L = int(L)
            bits = torch.randint(0, 2, (L, n_bits)).float()
            middle_relation_bits[i, :L] = bits
            middle_relation_id[i, :L] = (bits.long() * powers).sum(dim=-1)

        # Synthetic label: shorter paths and more relation bits -> stronger tie
        len_tensor = middle_len.float()
        strength = middle_relation_bits.sum(dim=(1, 2)) / (len_tensor + 1e-6)
        relation_strength = torch.sigmoid(strength - strength.mean())
        is_connected = torch.bernoulli(relation_strength.clamp(0.0, 1.0))

        return {
            "source_user_id": source_user_id,             # [B]
            "target_user_id": target_user_id,              # [B]
            "middle_relation_bits": middle_relation_bits,   # [B, T, n_bits]
            "middle_relation_id": middle_relation_id,       # [B, T]  (for embedding lookup)
            "middle_len": middle_len,                       # [B]
            "relation_strength": relation_strength,         # [B]  continuous label
            "is_connected": is_connected,                   # [B]  binary label
        }

    # ------------------------------------------------------------------
    # Debug helper
    # ------------------------------------------------------------------

    def get_shapes(self, batch_size: int = 4) -> None:
        """Print tensor shapes for every task (use before implementing models)."""
        tasks = [
            "ctr", "watch", "multitask", "seq",
            "retrieval", "graph", "uplift", "graph_relation",
        ]
        for task in tasks:
            batch = self.get_batch(batch_size, task)  # type: ignore
            print(f"\n=== {task} ===")
            for k, v in batch.items():
                print(f"  {k:24s}: {str(tuple(v.shape)):25s}  {v.dtype}")


if __name__ == "__main__":
    gen = SyntheticRecSysGenerator()
    gen.get_shapes(batch_size=4)

    # Sanity: multitask
    b = gen.get_batch(8, "multitask")
    assert b["history_ids"].shape == (8, 50)
    assert b["ctr_label"].shape == (8,)

    # Sanity: uplift — true_uplift should correlate with (label_treat - label_control) on average
    b = gen.get_batch(2000, "uplift")
    assert b["true_uplift"].shape == (2000,)
    assert (b["true_uplift"] >= 0).all()
    empirical_diff = b["label_treat"].mean() - b["label_control"].mean()
    print(f"\n[uplift sanity] mean true_uplift={b['true_uplift'].mean():.4f}  "
          f"empirical (treat - control) rate diff={empirical_diff:.4f}")

    # Sanity: graph_relation
    b = gen.get_batch(8, "graph_relation")
    assert b["middle_relation_id"].shape == (8, 10)
    assert b["middle_relation_id"].max() < 2 ** gen.cfg.num_relation_bits

    print("\nAll assertions passed.")
