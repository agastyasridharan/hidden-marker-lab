#!/usr/bin/env python3
"""Native-Hugging-Face mechanistic probes for hidden-marker matching heads."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from run_ood_introspection import TRANSFORMS, make_example
from train_introspection import (
    InjectionRow,
    InterventionController,
    LoRALinear,
    atomic_json,
    batch_examples,
    candidate_tokens,
    create_model_and_tokenizer,
    load_adapter_state,
    make_carrier,
    sample_pair,
    stable_seed,
    summarize_trials,
    transformer_layers,
    utc_now,
)


DISCOVERY_CONDITION = {"id": "mechanistic_discovery", "site": "period", "carrier": "same"}


def load_model(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    model, tokenizer, _ = create_model_and_tokenizer(args)
    saved: dict[str, Any] = {"step": 0, "config": {"condition": "base"}}
    if args.adapter != "none":
        saved = torch.load(args.adapter, map_location="cpu", weights_only=False)
        load_adapter_state(model, saved["adapter"])
    model.eval()
    model.config._attn_implementation = "eager"
    for layer in transformer_layers(model):
        layer.self_attn.config._attn_implementation = "eager"
    return model, tokenizer, saved


def examples_for_split(
    tokenizer: Any,
    seed: int,
    episodes: int,
    orbit: int,
) -> list[Any]:
    examples = []
    for episode in range(episodes):
        pair = sample_pair(5120, stable_seed(seed, "mechanistic-vector", episode))
        for transform in TRANSFORMS[:orbit]:
            for test_vector in (0, 1):
                examples.append(make_example(
                    tokenizer, seed, episode, pair, test_vector, transform,
                    DISCOVERY_CONDITION,
                ))
    return examples


def task_rows(logits: torch.Tensor, targets: torch.Tensor, examples: Sequence[Any]) -> list[dict[str, Any]]:
    predictions = logits.argmax(dim=1)
    rows = []
    for index, example in enumerate(examples):
        target = int(targets[index])
        prediction = int(predictions[index])
        rows.append({
            **example.metadata,
            "prediction": "AB"[prediction],
            "correct": prediction == target,
            "logit_A": float(logits[index, 0]),
            "logit_B": float(logits[index, 1]),
            "margin": float(logits[index, target] - logits[index, 1 - target]),
        })
    return rows


def adapter_weight_diagnostics(model: Any) -> list[dict[str, Any]]:
    rows = []
    for layer_index, layer in enumerate(transformer_layers(model)):
        attention = layer.self_attn
        if not isinstance(attention.q_proj, LoRALinear):
            continue
        q_module = attention.q_proj
        k_module = attention.k_proj
        v_module = attention.v_proj
        o_module = attention.o_proj
        with torch.no_grad():
            q_delta = q_module.lora_B @ q_module.lora_A * q_module.scale
            k_delta = k_module.lora_B @ k_module.lora_A * k_module.scale
            v_delta = v_module.lora_B @ v_module.lora_A * v_module.scale
            o_delta = o_module.lora_B @ o_module.lora_A * o_module.scale
            q_effective = q_module.base.weight.float() + q_delta
            k_effective = k_module.base.weight.float() + k_delta
            for head in range(40):
                q_slice = slice(head * 128, (head + 1) * 128)
                kv_head = head // 5
                kv_slice = slice(kv_head * 128, (kv_head + 1) * 128)
                q = q_effective[q_slice]
                k = k_effective[kv_slice]
                dq = q_delta[q_slice]
                dk = k_delta[kv_slice]
                qk_alignment = float((q @ k.T).norm() / (q.norm() * k.norm()))
                delta_alignment = float((dq @ dk.T).norm() / (dq.norm() * dk.norm()).clamp_min(1e-12))
                rows.append({
                    "layer": layer_index,
                    "head": head,
                    "kv_head": kv_head,
                    "q_delta_relative_norm": float(dq.norm() / q_module.base.weight[q_slice].float().norm()),
                    "k_delta_relative_norm": float(dk.norm() / k_module.base.weight[kv_slice].float().norm()),
                    "v_delta_relative_norm": float(v_delta[kv_slice].norm() / v_module.base.weight[kv_slice].float().norm()),
                    "o_delta_relative_norm": float(o_delta[:, q_slice].norm() / o_module.base.weight[:, q_slice].float().norm()),
                    "effective_qk_rowspace_alignment": qk_alignment,
                    "delta_qk_rowspace_alignment": delta_alignment,
                })
    return rows


def run_screen(args: argparse.Namespace) -> int:
    model, tokenizer, saved = load_model(args)
    device = next(model.parameters()).device
    examples = examples_for_split(tokenizer, args.seed, args.episodes, args.orbit)
    for example in examples:
        example.metadata["screen_positions"] = list(example.positions)
        if args.screen_control == "no-test":
            example.positions = example.positions[:2]
            example.vectors = example.vectors[:2]
    controller = InterventionController()
    handle = transformer_layers(model)[20].register_forward_pre_hook(controller.hook)
    sums = {
        name: torch.zeros(40, 40, dtype=torch.float64)
        for name in ("match_probability_contrast", "match_log_ratio", "answer_to_test_attention")
    }
    sumsq = {name: torch.zeros(40, 40, dtype=torch.float64) for name in sums}
    count = 0
    all_rows: list[dict[str, Any]] = []
    try:
        for start in range(0, len(examples), args.eval_batch_size):
            batch = examples[start:start + args.eval_batch_size]
            encoded, plans, targets, ids = batch_examples(tokenizer, batch, device)
            controller.plan = plans
            with torch.inference_mode():
                outputs = model(
                    **encoded, use_cache=False, output_attentions=True, return_dict=True,
                )
            controller.plan = None
            final_positions = encoded["attention_mask"].sum(1) - 1
            indices = torch.arange(len(batch), device=device)
            logits = outputs.logits[indices, final_positions][:, ids].float()
            all_rows.extend(task_rows(logits, targets, batch))
            for row_index, example in enumerate(batch):
                test_vector = 0 if example.metadata["test_vector"] == "v1" else 1
                shown_first = 1 if example.metadata["demo_reverse"] else 0
                correct_demo = 0 if shown_first == test_vector else 1
                wrong_demo = 1 - correct_demo
                screen_positions = example.metadata["screen_positions"]
                correct_position = screen_positions[correct_demo]
                wrong_position = screen_positions[wrong_demo]
                test_position = screen_positions[2]
                answer_position = int(final_positions[row_index])
                for layer_index, attention in enumerate(outputs.attentions):
                    at_test = attention[row_index, :, test_position].float()
                    correct_probability = at_test[:, correct_position]
                    wrong_probability = at_test[:, wrong_position]
                    values = {
                        "match_probability_contrast": correct_probability - wrong_probability,
                        "match_log_ratio": torch.log(correct_probability.clamp_min(1e-9)) - torch.log(wrong_probability.clamp_min(1e-9)),
                        "answer_to_test_attention": attention[row_index, :, answer_position, test_position].float(),
                    }
                    for name, value in values.items():
                        sums[name][layer_index] += value.double().cpu()
                        sumsq[name][layer_index] += value.double().cpu().square()
                count += 1
            del outputs
    finally:
        handle.remove()
    score_rows = []
    for layer in range(40):
        for head in range(40):
            row = {"layer": layer, "head": head}
            for name in sums:
                mean = sums[name][layer, head] / count
                variance = max(0.0, float(sumsq[name][layer, head] / count - mean * mean))
                row[name] = float(mean)
                row[f"{name}_sd"] = math.sqrt(variance)
            score_rows.append(row)
    output = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "model_condition": args.model_condition,
        "adapter": args.adapter,
        "adapter_step": saved.get("step"),
        "split": "discovery",
        "seed": args.seed,
        "episodes": args.episodes,
        "orbit": args.orbit,
        "screen_control": args.screen_control,
        "n_trials": count,
        "task_summary": summarize_trials(all_rows),
        "head_scores": score_rows,
        "weight_diagnostics": adapter_weight_diagnostics(model),
        "interpretation": "Attention and weight metrics nominate candidates; neither is causal evidence.",
    }
    atomic_json(Path(args.output_dir) / "screen.json", output)
    return 0


def parse_candidates(value: str) -> list[tuple[int, int]]:
    result = []
    for item in value.split(","):
        layer, head = item.split(".")
        result.append((int(layer), int(head)))
    return result


def forward_task(
    model: Any,
    tokenizer: Any,
    controller: InterventionController,
    examples: Sequence[Any],
    batch_size: int,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    rows: list[dict[str, Any]] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        encoded, plans, targets, ids = batch_examples(tokenizer, batch, device)
        controller.plan = plans
        with torch.inference_mode():
            outputs = model(**encoded, use_cache=False, return_dict=True)
        controller.plan = None
        final_positions = encoded["attention_mask"].sum(1) - 1
        indices = torch.arange(len(batch), device=device)
        logits = outputs.logits[indices, final_positions][:, ids].float()
        rows.extend(task_rows(logits, targets, batch))
    return rows


def zero_head_hook(head: int):
    def hook(_module: Any, hook_args: tuple[Any, ...]) -> tuple[Any, ...]:
        values = hook_args[0].clone()
        values[:, :, head * 128:(head + 1) * 128] = 0
        return (values, *hook_args[1:])
    return hook


def mean_head_hook(head: int):
    def hook(_module: Any, hook_args: tuple[Any, ...]) -> tuple[Any, ...]:
        values = hook_args[0].clone()
        head_slice = slice(head * 128, (head + 1) * 128)
        values[:, :, head_slice] = values[:, :, head_slice].mean(dim=0, keepdim=True)
        return (values, *hook_args[1:])
    return hook


def lm_control_loss(model: Any, tokenizer: Any, seed: int, batch_size: int) -> float:
    device = next(model.parameters()).device
    texts = []
    for paragraph in range(32):
        texts.append(" ".join(make_carrier(seed, paragraph * 4 + offset, "eval") for offset in range(4)))
    token_loss = 0.0
    token_count = 0
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start:start + batch_size], padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**encoded, use_cache=False).logits[:, :-1].float()
        targets = encoded["input_ids"][:, 1:]
        mask = encoded["attention_mask"][:, 1:].bool()
        losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
        token_loss += float(losses[mask].sum())
        token_count += int(mask.sum())
    return token_loss / token_count


def run_causal(args: argparse.Namespace) -> int:
    model, tokenizer, saved = load_model(args)
    layers = transformer_layers(model)
    controller = InterventionController()
    injection_handle = layers[20].register_forward_pre_hook(controller.hook)
    examples = examples_for_split(tokenizer, args.seed, args.episodes, args.orbit)
    candidates = parse_candidates(args.candidates) if args.candidates else []
    try:
        baseline_rows = forward_task(model, tokenizer, controller, examples, args.eval_batch_size)
        baseline_lm = lm_control_loss(model, tokenizer, args.seed, args.eval_batch_size)
        effects = []
        for layer, head in candidates:
            hook_function = mean_head_hook(head) if args.ablation_type == "mean" else zero_head_hook(head)
            ablation_handle = layers[layer].self_attn.o_proj.register_forward_pre_hook(hook_function)
            try:
                rows = forward_task(model, tokenizer, controller, examples, args.eval_batch_size)
                ablated_lm = lm_control_loss(model, tokenizer, args.seed, args.eval_batch_size)
            finally:
                ablation_handle.remove()
            paired = [base["margin"] - ablated["margin"] for base, ablated in zip(baseline_rows, rows)]
            effects.append({
                "layer": layer,
                "head": head,
                "baseline": summarize_trials(baseline_rows),
                "ablated": summarize_trials(rows),
                "accuracy_drop": summarize_trials(baseline_rows)["accuracy"] - summarize_trials(rows)["accuracy"],
                "mean_correct_margin_drop": statistics.fmean(paired),
                "median_correct_margin_drop": statistics.median(paired),
                "fraction_margin_harmed": statistics.fmean(value > 0 for value in paired),
                "general_lm_nll_increase": ablated_lm - baseline_lm,
            })
            print(
                f"head {layer}.{head}: accuracy drop {effects[-1]['accuracy_drop']:+.4f}, "
                f"margin drop {effects[-1]['mean_correct_margin_drop']:+.3f}, "
                f"LM nll {effects[-1]['general_lm_nll_increase']:+.5f}",
                flush=True,
            )
        if args.candidate_group:
            group = parse_candidates(args.candidate_group)
            group_handles = []
            try:
                for layer, head in group:
                    hook_function = mean_head_hook(head) if args.ablation_type == "mean" else zero_head_hook(head)
                    group_handles.append(layers[layer].self_attn.o_proj.register_forward_pre_hook(hook_function))
                rows = forward_task(model, tokenizer, controller, examples, args.eval_batch_size)
                ablated_lm = lm_control_loss(model, tokenizer, args.seed, args.eval_batch_size)
            finally:
                for group_handle in group_handles:
                    group_handle.remove()
            paired = [base["margin"] - ablated["margin"] for base, ablated in zip(baseline_rows, rows)]
            effects.append({
                "group": [f"{layer}.{head}" for layer, head in group],
                "baseline": summarize_trials(baseline_rows),
                "ablated": summarize_trials(rows),
                "accuracy_drop": summarize_trials(baseline_rows)["accuracy"] - summarize_trials(rows)["accuracy"],
                "mean_correct_margin_drop": statistics.fmean(paired),
                "median_correct_margin_drop": statistics.median(paired),
                "fraction_margin_harmed": statistics.fmean(value > 0 for value in paired),
                "general_lm_nll_increase": ablated_lm - baseline_lm,
            })
            print(
                f"group {args.candidate_group}: accuracy drop {effects[-1]['accuracy_drop']:+.4f}, "
                f"margin drop {effects[-1]['mean_correct_margin_drop']:+.3f}, "
                f"LM nll {effects[-1]['general_lm_nll_increase']:+.5f}",
                flush=True,
            )
    finally:
        injection_handle.remove()
    atomic_json(Path(args.output_dir) / f"causal-{args.ablation_type}.json", {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "model_condition": args.model_condition,
        "adapter": args.adapter,
        "adapter_step": saved.get("step"),
        "split": "confirmation",
        "seed": args.seed,
        "episodes": args.episodes,
        "orbit": args.orbit,
        "n_trials": len(baseline_rows),
        "baseline": summarize_trials(baseline_rows),
        "baseline_general_lm_nll": baseline_lm,
        "candidate_effects": effects,
        "ablation": (
            "replace the selected head's pre-o_proj z slice with its within-batch, position-specific mean"
            if args.ablation_type == "mean" else
            "zero the selected head's pre-o_proj z slice at every sequence position"
        ),
    })
    return 0


def patch_one_direction(
    model: Any,
    tokenizer: Any,
    controller: InterventionController,
    layer: int,
    head: int,
    source_examples: Sequence[Any],
    target_examples: Sequence[Any],
    batch_size: int,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    o_proj = transformer_layers(model)[layer].self_attn.o_proj
    rows = []
    for start in range(0, len(source_examples), batch_size):
        sources = source_examples[start:start + batch_size]
        targets = target_examples[start:start + batch_size]
        source_encoded, source_plans, _source_targets, ids = batch_examples(tokenizer, sources, device)
        captured: list[torch.Tensor] = []

        def capture(_module: Any, hook_args: tuple[Any, ...]) -> None:
            values = hook_args[0]
            captured.append(torch.stack([
                values[row, example.positions[2], head * 128:(head + 1) * 128].detach().clone()
                for row, example in enumerate(sources)
            ]))

        capture_handle = o_proj.register_forward_pre_hook(capture)
        controller.plan = source_plans
        with torch.inference_mode():
            model(**source_encoded, use_cache=False)
        controller.plan = None
        capture_handle.remove()

        target_encoded, target_plans, target_classes, candidate_ids = batch_examples(tokenizer, targets, device)
        controller.plan = target_plans
        with torch.inference_mode():
            baseline_outputs = model(**target_encoded, use_cache=False).logits
        controller.plan = None

        def patch(_module: Any, hook_args: tuple[Any, ...]) -> tuple[Any, ...]:
            values = hook_args[0].clone()
            for row, example in enumerate(targets):
                values[row, example.positions[2], head * 128:(head + 1) * 128] = captured[0][row]
            return (values, *hook_args[1:])

        patch_handle = o_proj.register_forward_pre_hook(patch)
        controller.plan = target_plans
        with torch.inference_mode():
            patched_outputs = model(**target_encoded, use_cache=False).logits
        controller.plan = None
        patch_handle.remove()
        final_positions = target_encoded["attention_mask"].sum(1) - 1
        indices = torch.arange(len(targets), device=device)
        baseline_logits = baseline_outputs[indices, final_positions][:, candidate_ids].float()
        patched_logits = patched_outputs[indices, final_positions][:, candidate_ids].float()
        for row, example in enumerate(targets):
            target_class = int(target_classes[row])
            baseline_margin = float(baseline_logits[row, target_class] - baseline_logits[row, 1 - target_class])
            patched_margin = float(patched_logits[row, target_class] - patched_logits[row, 1 - target_class])
            rows.append({
                "episode_id": example.metadata["episode_id"],
                "mapping_swap": example.metadata["mapping_swap"],
                "demo_reverse": example.metadata["demo_reverse"],
                "option_reverse": example.metadata["option_reverse"],
                "source_vector": sources[row].metadata["test_vector"],
                "target_vector": example.metadata["test_vector"],
                "baseline_target_correct_margin": baseline_margin,
                "patched_target_correct_margin": patched_margin,
                "target_margin_reduction": baseline_margin - patched_margin,
                "patched_flipped_target_answer": patched_margin < 0,
            })
    return rows


def run_patch(args: argparse.Namespace) -> int:
    model, tokenizer, saved = load_model(args)
    layers = transformer_layers(model)
    controller = InterventionController()
    injection_handle = layers[20].register_forward_pre_hook(controller.hook)
    candidates = parse_candidates(args.candidates)
    by_direction: dict[str, list[Any]] = {"v1": [], "v2": []}
    for episode in range(args.episodes):
        pair = sample_pair(5120, stable_seed(args.seed, "mechanistic-vector", episode))
        for transform in TRANSFORMS[:args.orbit]:
            for vector in (0, 1):
                by_direction[f"v{vector + 1}"].append(make_example(
                    tokenizer, args.seed, episode, pair, vector, transform,
                    DISCOVERY_CONDITION,
                ))
    effects = []
    try:
        for layer, head in candidates:
            rows = patch_one_direction(
                model, tokenizer, controller, layer, head,
                by_direction["v1"], by_direction["v2"], args.eval_batch_size,
            )
            rows.extend(patch_one_direction(
                model, tokenizer, controller, layer, head,
                by_direction["v2"], by_direction["v1"], args.eval_batch_size,
            ))
            effects.append({
                "layer": layer,
                "head": head,
                "n_patches": len(rows),
                "mean_target_margin_reduction": statistics.fmean(row["target_margin_reduction"] for row in rows),
                "median_target_margin_reduction": statistics.median(row["target_margin_reduction"] for row in rows),
                "fraction_target_answer_flipped": statistics.fmean(row["patched_flipped_target_answer"] for row in rows),
                "fraction_target_margin_reduced": statistics.fmean(row["target_margin_reduction"] > 0 for row in rows),
                "trials": rows,
            })
            print(
                f"patch {layer}.{head}: margin reduction {effects[-1]['mean_target_margin_reduction']:+.3f}, "
                f"flips {effects[-1]['fraction_target_answer_flipped']:.3f}",
                flush=True,
            )
    finally:
        injection_handle.remove()
    atomic_json(Path(args.output_dir) / "patch.json", {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "model_condition": args.model_condition,
        "adapter": args.adapter,
        "adapter_step": saved.get("step"),
        "split": "confirmation",
        "seed": args.seed,
        "effects": effects,
        "patch": "replace one head's pre-o_proj z at the test-marker position with the counterfactual vector trial",
    })
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("screen", "causal", "patch"), required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-14B")
    parser.add_argument("--adapter", default="none")
    parser.add_argument("--model-condition", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--orbit", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--screen-control", choices=("primary", "no-test"), default="primary")
    parser.add_argument("--candidates", default="")
    parser.add_argument("--candidate-group", default="")
    parser.add_argument("--ablation-type", choices=("zero", "mean"), default="zero")
    parser.add_argument("--lora-start-layer", type=int, default=20)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--train-layers", default="20")
    parser.add_argument("--eval-layers", default="20")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    Path(parsed.output_dir).mkdir(parents=True, exist_ok=True)
    if parsed.mode == "screen":
        raise SystemExit(run_screen(parsed))
    if parsed.mode == "causal" and not (parsed.candidates or parsed.candidate_group):
        raise SystemExit("causal mode requires --candidates or --candidate-group")
    if parsed.mode == "patch" and not parsed.candidates:
        raise SystemExit("patch mode requires --candidates")
    raise SystemExit(run_causal(parsed) if parsed.mode == "causal" else run_patch(parsed))
