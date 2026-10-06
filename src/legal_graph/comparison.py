"""Paired bootstrap, seed aggregation, and measured-versus-forecast comparison."""

from __future__ import annotations

import math
import random
from typing import Mapping, Sequence

from .outcomes import evaluate


def paired_bootstrap(
    gold: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    *,
    metric: str = "macro_f1",
    resamples: int = 10000,
    seed: int = 20261006,
) -> dict:
    """Bootstrap the metric difference A minus B over the same resampled cases."""
    if not (len(gold) == len(pred_a) == len(pred_b)) or not gold:
        raise ValueError("bootstrap needs equally sized, non-empty sequences")

    def score(indices: Sequence[int], predicted: Sequence[str]) -> float:
        return float(evaluate([gold[i] for i in indices], [predicted[i] for i in indices])[metric])

    base = list(range(len(gold)))
    delta = score(base, pred_a) - score(base, pred_b)
    generator = random.Random(seed)
    deltas: list[float] = []
    for _ in range(resamples):
        sample = [generator.randrange(len(gold)) for _ in base]
        deltas.append(score(sample, pred_a) - score(sample, pred_b))
    deltas.sort()
    return {
        "metric": metric,
        "delta": delta,
        "lower": deltas[int(0.025 * resamples)],
        "upper": deltas[min(resamples - 1, int(0.975 * resamples))],
        "share_a_not_better": sum(value <= 0 for value in deltas) / resamples,
        "resamples": resamples,
        "cases": len(gold),
    }


def aggregate_seeds(values: Sequence[float]) -> dict:
    if not values:
        raise ValueError("no seed values")
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return {"mean": round(mean, 12), "std": std, "n": len(values)}


def compare_to_projection(measured: Mapping[str, float], projection: Mapping) -> list[dict]:
    """Place each measured test accuracy against its projected Wilson interval."""
    rows: list[dict] = []
    for variant in projection["variants"]:
        variant_id = variant["variant_id"]
        if variant_id not in measured:
            continue
        test = variant["splits"]["test"]
        interval = test["accuracy_interval"]
        value = measured[variant_id]
        rows.append({
            "variant_id": variant_id,
            "projected": test["accuracy"],
            "lower": interval["lower"],
            "upper": interval["upper"],
            "measured": value,
            "inside_interval": interval["lower"] <= value <= interval["upper"],
        })
    return rows
