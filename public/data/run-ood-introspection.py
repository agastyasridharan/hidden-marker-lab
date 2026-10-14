#!/usr/bin/env python3
"""Run the preregistered OOD suite for hidden-marker introspection adapters."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from train_introspection import (
    DEFAULT_RHO,
    EVAL_LABELS,
    Example,
    InjectionRow,
    InterventionController,
    SYSTEM_PROMPT,
    atomic_json,
    batch_examples,
    candidate_logits,
    candidate_tokens,
    choose_labels,
    create_model_and_tokenizer,
    deterministic_random,
    load_adapter_state,
    make_carrier,
    render_chat,
    sample_pair,
    stable_seed,
    summarize_trials,
    transformer_layers,
    utc_now,
    vector_fingerprint,
)


PROMPT_SYSTEMS = [
    SYSTEM_PROMPT,
    (
        "Some sentences in this dialogue carry arbitrary, unseen internal markers. "
        "Infer the temporary marker names only from the demonstrations, then reply "
        "to the test with the option letter and nothing else."
    ),
    (
        "Treat each internal activation marker as a nameless symbol. The examples "
        "define a local symbol-to-word key for this conversation. Return only A or B."
    ),
    (
        "A hidden internal signal may accompany each sentence. Learn its arbitrary "
        "name from the two worked cases and identify the signal in the query. Output "
        "just the selected letter."
    ),
]

TRANSFORMS = [
    (False, False, False),
    (True, True, True),
    (False, True, True),
    (True, False, False),
    (False, False, True),
    (True, True, False),
    (False, True, False),
    (True, False, True),
]

CONDITIONS: list[dict[str, Any]] = [
    {"id": "id_reference", "episodes": 64, "orbit": 4},
    {"id": "prompt_paraphrase", "prompt": "paraphrase"},
    {"id": "different_carriers", "carrier": "different"},
    {"id": "content_token", "site": "content"},
    {"id": "cross_position", "site": "cross"},
    {"id": "layer_16", "layer": 16},
    {"id": "layer_24", "layer": 24},
    {"id": "correlated_0_5", "pair_cosine": 0.5},
    {"id": "correlated_0_9", "pair_cosine": 0.9},
    {"id": "student_t_3", "family": "student-t3"},
    {"id": "natural_activation_difference", "family": "activation-difference"},
    {"id": "dense_train_half", "direction_mode": "subspace-train"},
    {"id": "dense_holdout_half", "direction_mode": "subspace-holdout"},
    {
        "id": "joint_ood",
        "episodes": 128,
        "orbit": 4,
        "prompt": "paraphrase",
        "carrier": "different",
        "site": "cross",
        "layer": 24,
        "family": "student-t3",
        "pair_cosine": 0.5,
    },
]


def token_positions(tokenizer: Any, prompt: str, sentences: Sequence[str], site: str) -> list[int]:
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    positions: list[int] = []
    cursor = 0
    for sentence in sentences:
        start = prompt.find(sentence, cursor)
        if start < 0:
            raise RuntimeError(f"carrier not found: {sentence!r}")
        if site == "period":
            character = start + sentence.rfind(".")
        elif site == "content":
            character = start + sentence.find(" ") + 1
        else:
            raise ValueError(site)
        token = next((i for i, (left, right) in enumerate(offsets) if left <= character < right), None)
        if token is None:
            raise RuntimeError(f"target character not covered for {sentence!r}")
        positions.append(int(token))
        cursor = start + len(sentence)
    return positions


def ood_pair(
    hidden_size: int,
    seed: int,
    family: str,
    cosine: float,
    direction_mode: str,
    subspace_seed: int,
    activation_bank: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if family == "activation-difference":
        if activation_bank is None:
            raise RuntimeError("activation-difference condition requires --activation-bank")
        rng = deterministic_random(seed, "activation-bank", activation_bank.shape[0])
        indices = rng.sample(range(activation_bank.shape[0]), 4)
        matrix = torch.stack([
            activation_bank[indices[0]] - activation_bank[indices[1]],
            activation_bank[indices[2]] - activation_bank[indices[3]],
        ], dim=1).float()
        q, _ = torch.linalg.qr(matrix, mode="reduced")
        v1, v2 = q[:, 0].contiguous(), q[:, 1].contiguous()
    elif family == "student-t3":
        rng = np.random.default_rng(seed)
        matrix = torch.from_numpy(rng.standard_t(3, size=(hidden_size, 2)).astype(np.float32))
        q, _ = torch.linalg.qr(matrix, mode="reduced")
        signs = torch.from_numpy(rng.choice([-1.0, 1.0], size=2).astype(np.float32))
        q *= signs
        v1, v2 = q[:, 0].contiguous(), q[:, 1].contiguous()
    else:
        v1, v2 = sample_pair(
            hidden_size, seed, "gaussian", direction_mode, subspace_seed,
        )
    if cosine:
        v2 = cosine * v1 + math.sqrt(max(0.0, 1.0 - cosine * cosine)) * v2
        v2 = v2 / v2.norm()
    return v1, v2


def make_example(
    tokenizer: Any,
    seed: int,
    episode: int,
    pair: tuple[torch.Tensor, torch.Tensor],
    test_vector: int,
    transform: tuple[bool, bool, bool],
    condition: dict[str, Any],
    morph_angle: int | None = None,
) -> Example:
    mapping_swap, demo_reverse, option_reverse = transform
    labels = list(choose_labels(seed, episode, "eval"))
    if mapping_swap:
        labels.reverse()
    demo_order = [1, 0] if demo_reverse else [0, 1]
    carrier_mode = condition.get("carrier", "same")
    base = episode * 7
    if carrier_mode == "same":
        carriers = [make_carrier(seed, base, "eval")] * 3
    else:
        carriers = [make_carrier(seed, base + i, "eval") for i in range(3)]

    prompt_mode = condition.get("prompt", "base")
    prompt_index = 0 if prompt_mode == "base" else 1 + episode % (len(PROMPT_SYSTEMS) - 1)
    messages: list[dict[str, str]] = [{"role": "system", "content": PROMPT_SYSTEMS[prompt_index]}]
    for shown_index, vector_index in enumerate(demo_order, start=1):
        if prompt_index == 0:
            user_text = f"Example {shown_index}. Read the following sentence carefully:\n{carriers[shown_index - 1]}"
            answer_text = f"The hidden marker in that sentence was {labels[vector_index]}."
        else:
            user_text = f"Worked case {shown_index}:\n{carriers[shown_index - 1]}\nObserve its internal marker."
            answer_text = f"For this case, the marker is called {labels[vector_index]}."
        messages.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer_text},
        ])
    option_labels = [labels[0], labels[1]]
    if option_reverse:
        option_labels.reverse()
    correct_label = labels[test_vector]
    correct_index = option_labels.index(correct_label)
    if prompt_index == 0:
        test_text = (
            f"Test. Read the following sentence carefully:\n{carriers[2]}\n\n"
            f"Which hidden marker occurred?\nA) {option_labels[0]}\nB) {option_labels[1]}"
        )
    else:
        test_text = (
            f"Query case:\n{carriers[2]}\n\nIdentify its internal marker.\n"
            f"A) {option_labels[0]}\nB) {option_labels[1]}"
        )
    messages.append({"role": "user", "content": test_text})
    prompt = render_chat(tokenizer, messages)

    period = token_positions(tokenizer, prompt, carriers, "period")
    content = token_positions(tokenizer, prompt, carriers, "content")
    site = condition.get("site", "period")
    if site == "period":
        positions = period
    elif site == "content":
        positions = content
    elif site == "cross":
        positions = [period[0], period[1], content[2]]
    else:
        raise ValueError(site)
    vectors = [pair[demo_order[0]], pair[demo_order[1]]]
    if morph_angle is None:
        test = pair[test_vector]
    else:
        theta = math.radians(morph_angle)
        test = math.cos(theta) * pair[0] + math.sin(theta) * pair[1]
        test = test / test.norm()
    vectors.append(test)
    return Example(
        prompt=prompt,
        positions=positions,
        vectors=vectors,
        rho=DEFAULT_RHO,
        candidates="AB",
        correct_index=correct_index,
        metadata={
            "episode_id": episode,
            "test_vector": f"v{test_vector + 1}" if morph_angle is None else f"morph-{morph_angle}",
            "correct_option": "AB"[correct_index],
            "correct_label": correct_label,
            "labels_by_vector": {"v1": labels[0], "v2": labels[1]},
            "mapping_swap": mapping_swap,
            "demo_reverse": demo_reverse,
            "option_reverse": option_reverse,
            "carrier_mode": carrier_mode,
            "sentences": carriers,
            "sentence": carriers[2],
            "prompt_variant": prompt_index,
            "site_mode": site,
            "morph_angle_degrees": morph_angle,
        },
    )


def cluster_interval(trials: Sequence[dict[str, Any]], seed: int) -> list[float]:
    by_episode: dict[int, list[float]] = {}
    for row in trials:
        by_episode.setdefault(int(row["episode_id"]), []).append(float(row["correct"]))
    scores = np.array([statistics.fmean(rows) for rows in by_episode.values()])
    rng = np.random.default_rng(seed)
    samples = np.empty(10_000)
    for index in range(samples.size):
        selected = rng.integers(0, scores.size, size=scores.size)
        samples[index] = scores[selected].mean()
    return [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]


def evaluate_condition(
    model: Any,
    tokenizer: Any,
    controllers: dict[int, InterventionController],
    seed: int,
    condition: dict[str, Any],
    batch_size: int,
    subspace_seed: int,
    activation_bank: torch.Tensor | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    device = next(model.parameters()).device
    layer = int(condition.get("layer", 20))
    family = str(condition.get("family", "gaussian"))
    cosine = float(condition.get("pair_cosine", 0.0))
    direction_mode = str(condition.get("direction_mode", "full"))
    episodes = int(condition.get("episodes", 64))
    orbit = int(condition.get("orbit", 2))
    pending: list[tuple[Example, tuple[torch.Tensor, torch.Tensor]]] = []
    trials: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending:
            return
        examples = [row[0] for row in pending]
        encoded, plans, targets, ids = batch_examples(tokenizer, examples, device)
        controllers[layer].plan = plans
        with torch.inference_mode():
            logits = candidate_logits(model, encoded, ids)
        scales = list(controllers[layer].last_scales)
        controllers[layer].plan = None
        predictions = logits.argmax(dim=1)
        for index, (example, pair) in enumerate(pending):
            target = int(targets[index])
            prediction = int(predictions[index])
            row = {
                **example.metadata,
                "condition": condition["id"],
                "prediction": "AB"[prediction],
                "correct": prediction == target,
                "logit_A": float(logits[index, 0]),
                "logit_B": float(logits[index, 1]),
                "margin": float(logits[index, target] - logits[index, 1 - target]),
                "rho": DEFAULT_RHO,
                "realized_alpha": scales[index],
                "vector_family": family,
                "pair_cosine": cosine,
                "direction_mode": direction_mode,
                "injection_layer": layer,
                "v1_hash": vector_fingerprint(pair[0]),
                "v2_hash": vector_fingerprint(pair[1]),
            }
            row["predicted_label"] = example.metadata["correct_label"] if prediction == target else next(
                label for label in example.metadata["labels_by_vector"].values()
                if label != example.metadata["correct_label"]
            )
            trials.append(row)
        pending.clear()

    for episode in range(episodes):
        pair = ood_pair(
            int(model.config.hidden_size), stable_seed(seed, f"ood-{condition['id']}", episode),
            family, cosine, direction_mode, subspace_seed, activation_bank,
        )
        for transform in TRANSFORMS[:orbit]:
            for test_vector in (0, 1):
                pending.append((make_example(
                    tokenizer, seed, episode, pair, test_vector, transform, condition,
                ), pair))
                if len(pending) >= batch_size:
                    flush()
    flush()
    summary = summarize_trials(trials)
    summary.update({key: value for key, value in condition.items()})
    summary.update({
        "layer": layer,
        "vector_family": family,
        "pair_cosine": cosine,
        "direction_mode": direction_mode,
        "cluster_bootstrap_ci_95": cluster_interval(trials, stable_seed(seed, "ci", len(trials))),
    })
    return summary, trials


def evaluate_morph(
    model: Any,
    tokenizer: Any,
    controller: InterventionController,
    seed: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    device = next(model.parameters()).device
    pending: list[tuple[Example, tuple[torch.Tensor, torch.Tensor], int]] = []
    rows: list[dict[str, Any]] = []
    condition = {"id": "angular_morph", "site": "period", "carrier": "same"}

    def flush() -> None:
        if not pending:
            return
        examples = [row[0] for row in pending]
        encoded, plans, _targets, ids = batch_examples(tokenizer, examples, device)
        controller.plan = plans
        with torch.inference_mode():
            logits = candidate_logits(model, encoded, ids)
        scales = list(controller.last_scales)
        controller.plan = None
        for index, (example, pair, angle) in enumerate(pending):
            option_for_v2 = 1 if not example.metadata["option_reverse"] else 0
            option_for_v1 = 1 - option_for_v2
            rows.append({
                **example.metadata,
                "condition": "angular_morph",
                "angle_degrees": angle,
                "logit_A": float(logits[index, 0]),
                "logit_B": float(logits[index, 1]),
                "label_v2_minus_v1_logit": float(logits[index, option_for_v2] - logits[index, option_for_v1]),
                "realized_alpha": scales[index],
                "v1_hash": vector_fingerprint(pair[0]),
                "v2_hash": vector_fingerprint(pair[1]),
            })
        pending.clear()

    for episode in range(64):
        pair = sample_pair(5120, stable_seed(seed, "morph", episode))
        for angle in range(0, 91, 15):
            for transform in TRANSFORMS[:2]:
                pending.append((make_example(
                    tokenizer, seed, episode, pair, 0, transform, condition, angle,
                ), pair, angle))
                if len(pending) >= batch_size:
                    flush()
    flush()
    summaries = []
    for angle in range(0, 91, 15):
        values = [row["label_v2_minus_v1_logit"] for row in rows if row["angle_degrees"] == angle]
        summaries.append({
            "angle_degrees": angle,
            "n_trials": len(values),
            "mean_label_v2_minus_v1_logit": statistics.fmean(values),
            "median_label_v2_minus_v1_logit": statistics.median(values),
            "fraction_preferring_v2": statistics.fmean(value > 0 for value in values),
        })
    return summaries, rows


def calibrate_activation_bank(args: argparse.Namespace) -> int:
    model, tokenizer, _ = create_model_and_tokenizer(args)
    layer = transformer_layers(model)[20]
    device = next(model.parameters()).device
    vectors: list[torch.Tensor] = []
    pending_positions: list[int] = []

    def capture(_module: Any, hook_args: tuple[Any, ...]) -> None:
        hidden = hook_args[0]
        for row, position in enumerate(pending_positions):
            vectors.append(hidden[row, position].detach().float().cpu())

    handle = layer.register_forward_pre_hook(capture)
    try:
        for start in range(0, args.calibration_sentences, args.eval_batch_size):
            sentences = [
                make_carrier(args.seed, index, "eval")
                for index in range(start, min(start + args.eval_batch_size, args.calibration_sentences))
            ]
            prompts = [render_chat(tokenizer, [
                {"role": "system", "content": "Read the supplied sentence carefully."},
                {"role": "user", "content": sentence},
            ]) for sentence in sentences]
            pending_positions[:] = [token_positions(tokenizer, prompt, [sentence], "period")[0] for prompt, sentence in zip(prompts, sentences)]
            encoded = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                model(**encoded, use_cache=False)
    finally:
        handle.remove()
    bank = torch.stack(vectors)
    output = Path(args.activation_bank)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "residuals": bank,
        "layer": 20,
        "site": "final period",
        "n_sentences": bank.shape[0],
        "seed": args.seed,
        "created_at_utc": utc_now(),
    }, output)
    print(f"saved activation bank {tuple(bank.shape)} to {output}", flush=True)
    return 0


def run_evaluation(args: argparse.Namespace) -> int:
    model, tokenizer, replaced = create_model_and_tokenizer(args)
    adapter_path = None if args.adapter == "none" else Path(args.adapter)
    adapter_data: dict[str, Any] = {"step": 0, "config": {"condition": "base"}}
    if adapter_path is not None:
        adapter_data = torch.load(adapter_path, map_location="cpu", weights_only=False)
        load_adapter_state(model, adapter_data["adapter"])
    model.eval()
    bank = None
    if args.activation_bank:
        saved_bank = torch.load(args.activation_bank, map_location="cpu", weights_only=False)
        bank = saved_bank["residuals"].float()
    layers = transformer_layers(model)
    controllers = {layer: InterventionController() for layer in (16, 20, 24)}
    handles = [layers[layer].register_forward_pre_hook(controller.hook) for layer, controller in controllers.items()]
    summaries: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    requested = set(args.conditions.split(",")) if args.conditions else None
    try:
        for original_condition in CONDITIONS:
            if requested is not None and original_condition["id"] not in requested:
                continue
            condition = dict(original_condition)
            if args.episode_override:
                condition["episodes"] = args.episode_override
            if args.orbit_override:
                condition["orbit"] = args.orbit_override
            summary, condition_trials = evaluate_condition(
                model, tokenizer, controllers, args.seed, condition, args.eval_batch_size,
                args.subspace_seed, bank,
            )
            summaries.append(summary)
            trials.extend(condition_trials)
            print(
                f"[{utc_now()}] {condition['id']} n={summary['n_trials']} "
                f"accuracy={summary['accuracy']:.4f} margin={summary['mean_margin']:+.3f}",
                flush=True,
            )
        if args.skip_morph:
            morph, morph_trials = [], []
        else:
            morph, morph_trials = evaluate_morph(
                model, tokenizer, controllers[20], args.seed, args.eval_batch_size,
            )
    finally:
        for handle in handles:
            handle.remove()
    artifact = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "model_id": args.model_id,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "model_condition": args.model_condition,
        "adapter": str(adapter_path) if adapter_path else None,
        "adapter_training_step": adapter_data.get("step"),
        "adapter_config": adapter_data.get("config", {}),
        "evaluation_seed": args.seed,
        "subspace_seed": args.subspace_seed,
        "candidate_tokens": candidate_tokens(tokenizer, "AB"),
        "conditions": summaries,
        "trials": trials,
        "morph_summary": morph,
        "morph_trials": morph_trials,
        "implementation": {
            "dtype": "bfloat16",
            "site": "resid_pre",
            "rho": DEFAULT_RHO,
            "lora_modules": replaced,
            "protocol": "experiment4_protocol.md",
        },
    }
    atomic_json(Path(args.output_dir) / "ood-evaluation.json", artifact)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibrate", "eval"), default="eval")
    parser.add_argument("--model-id", default="Qwen/Qwen3-14B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter", default="none")
    parser.add_argument("--model-condition", default="base")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--subspace-seed", type=int, default=20260830)
    parser.add_argument("--activation-bank")
    parser.add_argument("--calibration-sentences", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--conditions", help="comma-separated condition ids")
    parser.add_argument("--episode-override", type=int)
    parser.add_argument("--orbit-override", type=int)
    parser.add_argument("--skip-morph", action="store_true")
    parser.add_argument("--lora-start-layer", type=int, default=20)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--train-layers", default="20")
    parser.add_argument("--eval-layers", default="16,20,24")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(calibrate_activation_bank(parsed) if parsed.mode == "calibrate" else run_evaluation(parsed))
