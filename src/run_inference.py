"""
vLLM inference pipeline for CoT faithfulness evaluation.
Processes all 3 conditions per task, runs inline LLM judge, writes JSONL.
"""

import argparse
import sys
from pathlib import Path

import jsonlines
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset import build_prompts, load_bbh_task
from parse_answers import parse_answer
from tag_verbalization import build_judge_prompts, tag_verbalization

MODEL_PATH = "/home/ubuntu/models/llama3-8b"

BBH_TASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "movie_recommendation",
    "navigate",
    "reasoning_about_colored_objects",
]


def run_task(
    llm,
    task: str,
    limit: int | None,
    n_shots: int,
    writer,
    sampling_params,
    judge_params,
) -> None:
    examples = load_bbh_task(task)
    prompt_records = build_prompts(task, examples, limit=limit, n_shots=n_shots)

    # --- CoT generation: one batched call for all conditions ---
    prompt_texts = [r["prompt_text"] for r in prompt_records]
    cot_outputs = llm.generate(prompt_texts, sampling_params)
    cot_texts = [o.outputs[0].text for o in cot_outputs]

    # --- Judge pass for bias_suggested only ---
    suggested_indices = [
        i for i, r in enumerate(prompt_records) if r["condition"] == "bias_suggested"
    ]
    judge_responses: list[str | None] = [None] * len(prompt_records)
    if suggested_indices:
        judge_prompts = build_judge_prompts([cot_texts[i] for i in suggested_indices])
        judge_outputs = llm.generate(judge_prompts, judge_params)
        for idx, out in zip(suggested_indices, judge_outputs):
            judge_responses[idx] = out.outputs[0].text

    # --- Write one record per prompt ---
    for i, rec in enumerate(prompt_records):
        cot = cot_texts[i]
        model_answer = parse_answer(cot)

        # bias_always_A has no explicit hint to verbalize
        if rec["condition"] == "bias_always_A":
            verbalized = None
        else:
            verbalized = tag_verbalization(
                cot=cot,
                judge_response=judge_responses[i],
                condition=rec["condition"],
            )

        writer.write({
            "task": rec["task"],
            "id": rec["id"],
            "condition": rec["condition"],
            "bias_target": rec["bias_target"],
            "correct_letter": rec["correct_letter"],
            "model_answer": model_answer,
            "is_correct": model_answer == rec["correct_letter"],
            "cot_text": cot,
            "verbalized_bias": verbalized,
        })


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CoT faithfulness inference")
    parser.add_argument("--tasks", nargs="+", default=BBH_TASKS)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max examples per task (default: full dataset)")
    parser.add_argument("--n_shots", type=int, default=3,
                        help="Few-shot exemplars for bias_always_A (default: 3)")
    parser.add_argument("--output", default="results/llama3_bbh.jsonl")
    args = parser.parse_args()

    # Lazy import: vLLM requires CUDA, only load when actually running inference
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(temperature=0, max_tokens=512)
    judge_params = SamplingParams(temperature=0, max_tokens=8)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=4096)

    with jsonlines.open(args.output, mode="w") as writer:
        for task in tqdm(args.tasks, desc="Tasks"):
            run_task(llm, task, args.limit, args.n_shots, writer,
                     sampling_params=sampling_params, judge_params=judge_params)
            print(f"  done: {task}")


if __name__ == "__main__":
    main()
