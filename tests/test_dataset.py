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
