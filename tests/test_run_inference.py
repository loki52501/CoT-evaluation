import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import jsonlines
import pytest

SRC = str(Path(__file__).parent.parent / "src")


def test_help_flag_exits_zero():
    result = subprocess.run(
        [sys.executable, f"{SRC}/run_inference.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--tasks" in result.stdout
    assert "--limit" in result.stdout
    assert "--n_shots" in result.stdout
    assert "--output" in result.stdout


def _mock_llm():
    """vLLM LLM mock that returns appropriate-length output lists."""
    def mock_generate(prompts, params):
        return [
            MagicMock(outputs=[MagicMock(text="So the answer is (A).")])
            for _ in prompts
        ]
    llm = MagicMock()
    llm.generate.side_effect = mock_generate
    return llm


def _one_example():
    return [{
        "id": 0,
        "raw_input": "Q?",
        "stem": "Q?",
        "choices": ["foo", "bar"],
        "correct_index": 0,
        "correct_letter": "A",
        "available_letters": ["A", "B"],
    }]


def test_run_task_writes_three_records(tmp_path):
    sys.path.insert(0, SRC)
    from run_inference import run_task

    written = []
    mock_writer = MagicMock()
    mock_writer.write.side_effect = written.append

    with patch("run_inference.load_bbh_task", return_value=_one_example()):
        run_task(_mock_llm(), "date_understanding", limit=1, n_shots=0, writer=mock_writer)

    assert len(written) == 3
    assert {r["condition"] for r in written} == {
        "baseline", "bias_suggested", "bias_always_A"
    }


def test_run_task_verbalized_none_for_always_a(tmp_path):
    sys.path.insert(0, SRC)
    from run_inference import run_task

    written = []
    mock_writer = MagicMock()
    mock_writer.write.side_effect = written.append

    with patch("run_inference.load_bbh_task", return_value=_one_example()):
        run_task(_mock_llm(), "date_understanding", limit=1, n_shots=0, writer=mock_writer)

    always_a = next(r for r in written if r["condition"] == "bias_always_A")
    assert always_a["verbalized_bias"] is None


def test_run_task_record_has_required_fields(tmp_path):
    sys.path.insert(0, SRC)
    from run_inference import run_task

    written = []
    mock_writer = MagicMock()
    mock_writer.write.side_effect = written.append

    with patch("run_inference.load_bbh_task", return_value=_one_example()):
        run_task(_mock_llm(), "date_understanding", limit=1, n_shots=0, writer=mock_writer)

    required = {
        "task", "id", "condition", "bias_target", "correct_letter",
        "model_answer", "is_correct", "cot_text", "verbalized_bias",
    }
    for rec in written:
        assert required.issubset(rec.keys())
