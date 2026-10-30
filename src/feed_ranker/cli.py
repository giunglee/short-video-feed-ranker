"""Thin command dispatch for prepared data, single jobs and fitted scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feed_ranker.data.parquet import ParquetReader
from feed_ranker.evaluation.funnel import evaluate_pipeline, open_benchmark
from feed_ranker.preprocessing.kuairand import TRAIN_END, VAL_END, prepare_kuairand
from feed_ranker.recommendation.contracts import CandidateScope, Request, Retriever
from feed_ranker.recommendation.merged import MergedRetriever
from feed_ranker.recommendation.pipeline import RecommendationPipeline
from feed_ranker.recommendation.rankers import load_ranker
from feed_ranker.recommendation.retrievers import load_retriever
from feed_ranker.recommendation.traces import RecommendationTrace
from feed_ranker.training.jobs import fit_recall_job, job_config, train_job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--raw", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--users", type=int, default=32)
    prep.add_argument("--train-end", type=int, default=TRAIN_END)
    prep.add_argument("--val-end", type=int, default=VAL_END)
    prep.add_argument("--basic", type=Path)
    prep.add_argument("--profiles", type=Path)
    validation = commands.add_parser("validate-data")
    validation.add_argument("--data", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--resume-from", type=Path)
    recall = commands.add_parser("fit-recall")
    recall.add_argument("--data", type=Path, required=True)
    recall.add_argument("--out", type=Path, required=True)
    recall.add_argument("--epochs", type=int, required=True)
    recall.add_argument("--seed", type=int, required=True)
    for name in ("recommend", "evaluate"):
        operation = commands.add_parser(name)
        operation.add_argument("--ranker", type=Path, required=True)
        operation.add_argument("--recall", type=Path, nargs="+", required=True)
        operation.add_argument("--policy", choices=("rr_all", "rrf_all"), default="rrf_all")
        operation.add_argument("--budget", type=int, default=50)
        operation.add_argument("--out", type=Path, required=True)
        operation.add_argument(
            "--requests" if name == "recommend" else "--data", type=Path, required=True
        )
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_kuairand(
            args.raw,
            args.out,
            users=args.users,
            train_end=args.train_end,
            val_end=args.val_end,
            basic=args.basic,
            profiles=args.profiles,
        )
    elif args.command == "validate-data":
        ParquetReader(args.data).validate()
    elif args.command == "train":
        train_job(
            job_config(json.loads(args.config.read_text())), args.out, resume_from=args.resume_from
        )
    elif args.command == "fit-recall":
        fit_recall_job(args.data, args.out, epochs=args.epochs, seed=args.seed)
    else:
        ranker = load_ranker(args.ranker)
        sources = [load_retriever(p) for p in args.recall]
        retriever: Retriever = (
            sources[0]
            if len(sources) == 1
            else MergedRetriever(
                sources, {"rr_all": "round_robin", "rrf_all": "rrf"}[args.policy], source_depth=50
            )
        )
        if args.command == "evaluate":
            evaluate_pipeline(
                open_benchmark(args.data), ranker, retriever, args.out, budget=args.budget
            )
        else:
            pipeline = RecommendationPipeline(retriever, ranker, args.budget, 10)
            rows = json.loads(args.requests.read_text())
            results = []
            for row in rows:
                items = np.asarray(row["items"])
                if items.size == 0:
                    items = items.astype(np.int64)
                results.append(
                    pipeline.recommend(
                        Request(row["query_id"], row["user_id"]), CandidateScope(items)
                    )
                )
            args.out.mkdir(parents=True, exist_ok=False)
            RecommendationTrace.from_results(results).write(args.out / "recommendation")


if __name__ == "__main__":
    main()
