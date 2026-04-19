"""
BBH dataset loading, prompt construction for CoT faithfulness evaluation.
"""

import re
from pathlib import Path
from typing import Optional, TypedDict

from datasets import load_dataset

from parse_answers import index_to_letter, letter_to_index, pick_wrong_letter

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
TURPIN_EXEMPLARS_DIR = PROMPTS_DIR / "turpin_exemplars"

# Tasks with no Options block in input — synthesize choices from known answer space
_SYNTHETIC_CHOICES: dict[str, list[str]] = {
    "boolean_expressions": ["True", "False"],
    "formal_fallacies": ["valid", "invalid"],
}

_OPT_LINE_RE = re.compile(r"^\(([A-F])\)\s+(.+)$", re.IGNORECASE)
_LETTER_TARGET_RE = re.compile(r"^\(([A-F])\)$")


class PromptRecord(TypedDict):
    task: str
    id: int
    condition: str           # "baseline" | "bias_suggested" | "bias_always_A"
    bias_target: Optional[str]
    correct_letter: str
    prompt_text: str


def _parse_options(raw_input: str, task: str) -> tuple[str, list[str]]:
    """
    Extract (question_stem, choices) from a BBH input string.
    Returns stem without Options block, and list of choice strings.
    """
    lines = raw_input.split("\n")
    stem_lines: list[str] = []
    choices: list[str] = []
    in_options = False

    for line in lines:
        stripped = line.strip()
        if stripped.lower() in ("options:", "choices:"):
            in_options = True
            continue
        if in_options:
            m = _OPT_LINE_RE.match(stripped)
            if m:
                choices.append(m.group(2).strip())
            elif stripped:
                in_options = False
                stem_lines.append(line)
        else:
            stem_lines.append(line)

    if choices:
        return "\n".join(stem_lines).strip(), choices

    if task in _SYNTHETIC_CHOICES:
        return raw_input.strip(), _SYNTHETIC_CHOICES[task]

    return raw_input.strip(), []


def _get_correct_index(target: str, choices: list[str]) -> int:
    """Map a BBH target string ('(B)' or 'True') to a 0-based choice index."""
    m = _LETTER_TARGET_RE.match(target.strip())
    if m:
        return letter_to_index(m.group(1))
    target_clean = target.strip().lower()
    for i, c in enumerate(choices):
        if c.strip().lower() == target_clean:
            return i
    return 0


def _format_options(choices: list[str]) -> str:
    """Format ['foo', 'bar'] → '(A) foo\n(B) bar'"""
    return "\n".join(f"({index_to_letter(i)}) {c}" for i, c in enumerate(choices))


def load_bbh_task(task_name: str) -> list[dict]:
    """
    Load a BBH task from HuggingFace and return parsed examples.
    Each dict: {id, raw_input, stem, choices, correct_index, correct_letter, available_letters}
    """
    ds = load_dataset(
        "lighteval/big_bench_hard", task_name, trust_remote_code=True, split="test"
    )
    examples = []
    for i, row in enumerate(ds):
        stem, choices = _parse_options(row["input"], task_name)
        correct_index = _get_correct_index(row["target"], choices)
        available_letters = [index_to_letter(j) for j in range(len(choices))]
        examples.append({
            "id": i,
            "raw_input": row["input"],
            "stem": stem,
            "choices": choices,
            "correct_index": correct_index,
            "correct_letter": index_to_letter(correct_index),
            "available_letters": available_letters,
        })
    return examples
