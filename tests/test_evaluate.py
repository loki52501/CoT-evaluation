import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from evaluate import compute_metrics


def _make_df(records):
    return pd.DataFrame(records)


def test_accuracy_drop_computed_correctly():
    records = [
        # baseline: 2/2 correct
        {"task": "t", "condition": "baseline", "is_correct": True,
         "model_answer": "A", "bias_target": None, "verbalized_bias": None},
        {"task": "t", "condition": "baseline", "is_correct": True,
         "model_answer": "B", "bias_target": None, "verbalized_bias": None},
        # bias_suggested: 1/2 correct
        {"task": "t", "condition": "bias_suggested", "is_correct": False,
         "model_answer": "B", "bias_target": "B", "verbalized_bias": False},
        {"task": "t", "condition": "bias_suggested", "is_correct": True,
         "model_answer": "A", "bias_target": "B", "verbalized_bias": None},
        # bias_always_A: 1/2 correct
        {"task": "t", "condition": "bias_always_A", "is_correct": True,
         "model_answer": "A", "bias_target": "A", "verbalized_bias": None},
        {"task": "t", "condition": "bias_always_A", "is_correct": False,
         "model_answer": "B", "bias_target": "A", "verbalized_bias": None},
    ]
    metrics = compute_metrics(_make_df(records))
    row = metrics[metrics["task"] == "t"].iloc[0]
    assert row["accuracy_baseline"] == pytest.approx(1.0)
    assert row["accuracy_biased"] == pytest.approx(0.5)
    assert row["accuracy_drop"] == pytest.approx(0.5)
    assert row["accuracy_always_a"] == pytest.approx(0.5)
    assert row["accuracy_drop_always_a"] == pytest.approx(0.5)


def test_unfaithfulness_and_articulation():
    records = [
        {"task": "t", "condition": "baseline", "is_correct": True,
         "model_answer": "A", "bias_target": None, "verbalized_bias": None},
        # followed bias, verbalized
        {"task": "t", "condition": "bias_suggested", "is_correct": False,
         "model_answer": "B", "bias_target": "B", "verbalized_bias": True},
        # followed bias, did NOT verbalize
        {"task": "t", "condition": "bias_suggested", "is_correct": False,
         "model_answer": "B", "bias_target": "B", "verbalized_bias": False},
    ]
    metrics = compute_metrics(_make_df(records))
    row = metrics[metrics["task"] == "t"].iloc[0]
    assert row["unfaithfulness_rate"] == pytest.approx(0.5)
    assert row["articulation_rate"] == pytest.approx(0.5)


def test_aggregate_row_present():
    records = [
        {"task": "t1", "condition": "baseline", "is_correct": True,
         "model_answer": "A", "bias_target": None, "verbalized_bias": None},
        {"task": "t1", "condition": "bias_suggested", "is_correct": False,
         "model_answer": "B", "bias_target": "B", "verbalized_bias": False},
    ]
    metrics = compute_metrics(_make_df(records))
    assert "__aggregate__" in metrics["task"].values


def test_metrics_has_all_columns():
    records = [
        {"task": "t", "condition": "baseline", "is_correct": True,
         "model_answer": "A", "bias_target": None, "verbalized_bias": None},
        {"task": "t", "condition": "bias_suggested", "is_correct": False,
         "model_answer": "B", "bias_target": "B", "verbalized_bias": False},
    ]
    metrics = compute_metrics(_make_df(records))
    expected_cols = {
        "task", "accuracy_baseline", "accuracy_biased", "accuracy_drop",
        "accuracy_always_a", "accuracy_drop_always_a",
        "unfaithfulness_rate", "articulation_rate", "n_baseline", "n_biased",
    }
    assert expected_cols.issubset(set(metrics.columns))
