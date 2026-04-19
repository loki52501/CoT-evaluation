import pytest
from dataset import _parse_options, _get_correct_index, _format_options


def test_parse_options_with_options_block():
    raw = "What is the date?\nOptions:\n(A) 01/01/2002\n(B) 02/01/2002\n(C) 03/01/2002"
    stem, choices = _parse_options(raw, "date_understanding")
    assert stem == "What is the date?"
    assert choices == ["01/01/2002", "02/01/2002", "03/01/2002"]


def test_parse_options_synthetic_boolean():
    raw = "not ( True or False )"
    stem, choices = _parse_options(raw, "boolean_expressions")
    assert stem == "not ( True or False )"
    assert choices == ["True", "False"]


def test_parse_options_synthetic_formal():
    raw = "Some As are Bs. All Bs are Cs. Therefore some As are Cs."
    stem, choices = _parse_options(raw, "formal_fallacies")
    assert choices == ["valid", "invalid"]


def test_get_correct_index_letter_target():
    choices = ["foo", "bar", "baz"]
    assert _get_correct_index("(B)", choices) == 1


def test_get_correct_index_letter_target_a():
    choices = ["foo", "bar"]
    assert _get_correct_index("(A)", choices) == 0


def test_get_correct_index_word_target_true():
    choices = ["True", "False"]
    assert _get_correct_index("True", choices) == 0


def test_get_correct_index_word_target_false():
    choices = ["True", "False"]
    assert _get_correct_index("False", choices) == 1


def test_format_options_two_choices():
    assert _format_options(["alpha", "beta"]) == "(A) alpha\n(B) beta"


def test_format_options_three_choices():
    result = _format_options(["x", "y", "z"])
    assert result == "(A) x\n(B) y\n(C) z"


from dataset import build_prompts
from parse_answers import index_to_letter


def _make_example(
    id_=0,
    stem="Is it raining?",
    choices=("Yes", "No"),
    correct_index=0,
):
    choices = list(choices)
    letters = [index_to_letter(i) for i in range(len(choices))]
    return {
        "id": id_,
        "raw_input": stem,
        "stem": stem,
        "choices": choices,
        "correct_index": correct_index,
        "correct_letter": index_to_letter(correct_index),
        "available_letters": letters,
    }


def test_build_prompts_returns_three_per_example():
    examples = [_make_example()]
    records = build_prompts("navigate", examples, limit=1, n_shots=0)
    assert len(records) == 3


def test_build_prompts_all_conditions_present():
    examples = [_make_example()]
    records = build_prompts("navigate", examples, limit=1, n_shots=0)
    conditions = {r["condition"] for r in records}
    assert conditions == {"baseline", "bias_suggested", "bias_always_A"}


def test_baseline_prompt_has_correct_format():
    examples = [_make_example(stem="Who is president?", choices=["Alice", "Bob"])]
    records = build_prompts("navigate", examples, limit=1, n_shots=0)
    baseline = next(r for r in records if r["condition"] == "baseline")
    assert "Let's think step by step" in baseline["prompt_text"]
    assert baseline["bias_target"] is None
    assert baseline["correct_letter"] == "A"


def test_bias_suggested_uses_wrong_letter():
    # correct=A, so wrong should be B
    examples = [_make_example(choices=["Yes", "No"], correct_index=0)]
    records = build_prompts("navigate", examples, limit=1, n_shots=0)
    suggested = next(r for r in records if r["condition"] == "bias_suggested")
    assert suggested["bias_target"] == "B"
    assert "(B)" in suggested["prompt_text"]
    assert "I think the answer is" in suggested["prompt_text"]


def test_bias_always_a_target_is_a():
    examples = [_make_example()]
    records = build_prompts("navigate", examples, limit=1, n_shots=0)
    always_a = next(r for r in records if r["condition"] == "bias_always_A")
    assert always_a["bias_target"] == "A"


def test_build_prompts_limit_respected():
    examples = [_make_example(id_=i) for i in range(5)]
    records = build_prompts("navigate", examples, limit=2, n_shots=0)
    assert len(records) == 6  # 2 examples × 3 conditions


def test_build_prompts_no_limit_uses_all():
    examples = [_make_example(id_=i) for i in range(4)]
    records = build_prompts("navigate", examples, limit=None, n_shots=0)
    assert len(records) == 12  # 4 × 3


def test_always_a_exemplar_reorders_choices():
    # exemplar has correct=B (index 1); in the prefix it should appear as (A)
    exemplar = _make_example(id_=0, choices=["Wrong", "Right"], correct_index=1)
    test_ex = _make_example(id_=1, choices=["foo", "bar"], correct_index=0)
    examples = [exemplar, test_ex]
    records = build_prompts("navigate", examples, limit=None, n_shots=1)
    always_a = [r for r in records if r["condition"] == "bias_always_A" and r["id"] == 1]
    assert len(always_a) == 1
    # exemplar had correct="Right" at index 1, should appear as (A) Right in prefix
    assert "(A) Right" in always_a[0]["prompt_text"]
