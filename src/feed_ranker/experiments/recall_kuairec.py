"""Validation-only exact recall runs; test and DNN intentionally have no CLI path."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from feed_ranker.artifacts import collect_run_provenance, save_checkpoint, sha256, write_json
from feed_ranker.data.recall import RecallData
from feed_ranker.models.recall import BPRMF, ObservedSampler, baseline_scores
from feed_ranker.training.recall_evaluation import evaluate


def export(data: RecallData, scores: np.ndarray, output: Path) -> dict[str, Any]:
    summary, per_user, candidates = evaluate(data, scores)
    np.save(output / "scores.npy", scores)
    np.save(output / "candidates.npy", candidates)
    per_user.to_csv(output / "per_user.csv", index=False)
    write_json(output / "summary.json", summary)
    return summary


def run(
    data_path: Path,
    output: Path,
    method: str,
    epochs: int,
    resume: bool,
    negative_policy: str = "negative_only",
    threads: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    if threads < 1 or seed < 0:
        raise ValueError("Require positive thread count and nonnegative seed")
    if method != "mf" and negative_policy != "negative_only":
        raise ValueError("Comparison policy applies only to MF")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    start = time.monotonic()
    data = RecallData(data_path)
    if output.exists() and not resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    provenance = collect_run_provenance()
    config = {
        "method": method,
        "dimension": 32,
        "batch_size": 4096,
        "seed": seed,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "selection": "macro_recall@50",
        "data_manifest_sha256": sha256(data_path / "meta.json"),
        "threads": threads,
        "negative_policy": negative_policy,
    }
    if not resume:
        write_json(output / "config.json", config)
        write_json(output / "provenance.json", provenance)
        shutil.copytree(
            Path(__file__).parents[1],
            output / "source" / "feed_ranker",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    elif json.loads((output / "config.json").read_text()) != config:
        raise ValueError("Resume configuration/data mismatch")
    write_json(output / "support.json", data.audit)
    write_json(output / "status.json", {"stage": "running", "method": method})
    if method != "mf":
        if resume:
            raise ValueError("Only MF supports resume")
        scores, support = baseline_scores(data, method)
        summary = export(data, scores, output)
        summary["support"] = support
    else:
        model = BPRMF(len(data.users), len(data.items))
        # reasonable defaults, not from any specific production config
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        sampler = ObservedSampler(data, negative_policy)
        history = []
        first_epoch, best, best_epoch = 0, -1.0, 0
        if resume:
            saved = torch.load(output / "last.pt", weights_only=True)
            if saved["source_sha256"] != provenance["source_sha256"]:
                raise ValueError("Training source changed; cannot silently resume")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            torch.set_rng_state(saved["torch_rng"])
            first_epoch, best, best_epoch = saved["epoch"], saved["best"], saved["best_epoch"]
            history = json.loads((output / "history.json").read_text())
        for epoch in range(first_epoch, epochs):
            tick = time.monotonic()
            model.train()
            users, positive, negative = sampler.sample(np.random.default_rng(seed + epoch))
            total, margin_total, gradient_total, steps = 0.0, 0.0, 0.0, 0
            for lo in range(0, len(users), 4096):
                u, p, n = users[lo : lo + 4096], positive[lo : lo + 4096], negative[lo : lo + 4096]
                optimizer.zero_grad(set_to_none=True)
                margin = model(u, p, n)
                loss = F.softplus(-margin).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite BPR loss")
                loss.backward()
                grad = sum(float(param.grad.square().sum()) for param in model.parameters()) ** 0.5
                if not np.isfinite(grad):
                    raise ValueError("Nonfinite BPR gradient")
                optimizer.step()
                model.item_updated[p] = True
                model.item_updated[n] = True
                total += float(loss.detach()) * len(u)
                margin_total += float(margin.detach().sum())
                gradient_total += grad
                steps += 1
            training_seconds = time.monotonic() - tick
            scores = model.pool_scores(data)
            measured, _, _ = evaluate(data, scores)
            value = measured["metrics"]["recall@50"]
            if value is None:
                raise ValueError("No eligible positive validation users")
            if value > best:
                best, best_epoch = value, epoch + 1
                save_checkpoint(
                    output / "best.pt",
                    {
                        "model": model.state_dict(),
                        "epoch": best_epoch,
                        "metric": best,
                        "config": config,
                    },
                )
            row = {
                "epoch": epoch + 1,
                "loss": total / len(users),
                "mean_margin": margin_total / len(users),
                "mean_gradient_norm": gradient_total / steps,
                "triples": len(users),
                "negative_items_used": int(torch.unique(negative).numel()),
                "neutral_comparison_draws": sampler.last_neutral_draws,
                "supported_items": int(model.item_updated.sum()),
                "train_seconds": training_seconds,
                "epoch_seconds": time.monotonic() - tick,
                **measured["metrics"],
            }
            history.append(row)
            write_json(output / "history.json", history)
            save_checkpoint(
                output / "last.pt",
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best": best,
                    "best_epoch": best_epoch,
                    "torch_rng": torch.get_rng_state(),
                    "source_sha256": provenance["source_sha256"],
                },
            )
            write_json(
                output / "status.json",
                {
                    "stage": "training",
                    "epoch": epoch + 1,
                    "best_epoch": best_epoch,
                    "best_recall@50": best,
                },
            )
            print(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "loss": row["loss"],
                        "recall@50": value,
                        "seconds": row["epoch_seconds"],
                        "best_epoch": best_epoch,
                    }
                ),
                flush=True,
            )
        selected = torch.load(output / "best.pt", weights_only=True)
        model.load_state_dict(selected["model"])
        summary = export(data, model.pool_scores(data), output)
        summary.update(
            {
                "selected_epoch": selected["epoch"],
                "epochs_completed": len(history),
                "selected_checkpoint_sha256": sha256(output / "best.pt"),
            }
        )
    summary["run_seconds"] = time.monotonic() - start
    write_json(output / "summary.json", summary)
    write_json(output / "status.json", {"stage": "complete", "summary": summary})
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=["popularity", "cooccurrence", "mf"], required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--negative-policy",
        choices=["negative_only", "negative_and_neutral"],
        default="negative_only",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    run(
        args.data,
        args.output,
        args.method,
        args.epochs,
        args.resume,
        args.negative_policy,
        args.threads,
        args.seed,
    )


if __name__ == "__main__":
    main()
