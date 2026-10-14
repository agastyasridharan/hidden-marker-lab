# Hidden Marker Lab

Experiment 1 tests whether Qwen3-14B can learn an arbitrary in-context mapping
between two injected residual-stream directions and the nonce labels `dax` and
`wug`, then identify the direction repeated at test time.

The completed minimal pilot contains 64 fresh orthonormal vector pairs and two
paired trials per sentence. It obtained 64/128 correct (50.0%; 95% Wilson CI
41.5%–58.5%) with a mean signed correct-choice margin of 0.000. Every trial
predicted `A · dax`, so the aggregate is fully explained by a fixed-choice bias.

Experiment 2 converts the task to a binary `YES`/`NO` question: did the `dax`
marker occur? It also records a matched pre-exposure response prior and the
post-exposure logits after the test carrier. It obtained 64/128 correct with all
128 posterior predictions equal to `NO`. The mean `YES − NO` score shifted by
`+0.575`, but nearly identically under `v1` (`+0.572`) and `v2` (`+0.578`).

Experiment 3 fine-tunes rank-16 LoRA adapters with fresh online random vectors,
carriers, nonce labels, mappings, and presentation orders. Across three
independent fixed-layer seeds, held-out primary accuracy was 92.6%–94.0% (mean
93.25%) on 4,096 counterfactually balanced trials per seed, versus 50.0% for
the matched base model. The paired improvement was +43.25 percentage points
(95% episode-cluster bootstrap CI +42.15 to +44.31). A separately trained
cross-layer adapter scored 100% at block 20 and 77.3%/100% at blocks 16/24.

The controls localize the effect: omitting the test intervention or replacing
it with a third orthogonal vector returned accuracy to 50.0%; sign reversal
reduced mean accuracy to 12.8%; Rademacher and sparse vectors retained 94.5%
and 93.4%. Changing all three carriers within an episode reduced accuracy to
72.3%, so carrier invariance remains incomplete. These are functional-access
results and do not establish subjective introspection.

## Project structure

- `experiment/run_experiment.py` is the self-contained BF16 GPU runner.
- `experiment/train_introspection.py` trains and evaluates the online-random
fine-tuning variant with counterfactual and out-of-distribution controls.
- `experiment/run_ood_introspection.py` evaluates frozen adapters across vector,
  prompt, carrier, token-site, layer, and joint distribution shifts.
- `experiment/mechanistic_introspection.py` screens attention heads and runs
  held-out mean ablation, counterfactual patching, and matched-group controls.
- `experiment/notebooks/introspection_head_analysis.ipynb` is the executed
  Experiment 4 analysis notebook, with rationales, figures, and exercises.
- `experiment/notebooks/hidden_marker_lab_replication.ipynb` is the complete
  executable walkthrough for reproducing Experiments 1–4 from scratch. The
  downloadable notebook embeds its support scripts and recorded summaries, so
  its default CPU analysis runs without a separate repository checkout.
- `experiment/local_gui/` contains Residual Marker Studio, a separate local
  token-level steering interface that connects privately to one Athena GPU and
  can switch between the base model and all four trained LoRA variants.
- `experiment/aggregate_introspection.py` combines matched base and multi-seed
  evaluation artifacts for the dashboard.
- `experiment/results/full/` preserves the complete audit JSON, checkpoint, and
  all full FP32 vectors.
- `experiment/results/binary-full/` preserves the binary experiment artifacts.
- `experiment/results/introspection-training/` preserves all five matched
  evaluation ledgers, training traces and metadata, plus a checksummed manifest
  for the four adapters retained on Athena.
- `experiment/results/experiment4/` preserves the OOD, exactly orthogonal-span,
  and mechanistic analyses; the dashboard exposes their complete trial ledgers.
- `app/` contains the interactive results dashboard.
- `public/data/` exposes the audit JSON and vector archive for download.
- `tests/rendered-html.test.mjs` checks the rendered dashboard and artifact
  invariants.

## Run the experiment

The script is designed for the Athena `gpurun` wrapper, which uploads one Python
file, selects an unused GPU, and uses the shared Hugging Face cache.

```bash
GPU=auto gpurun experiment/run_experiment.py \
  --output-dir /data/agastyas/hidden-marker-lab/full
```

For the binary variant:

```bash
GPU=auto gpurun experiment/run_experiment.py \
  --experiment binary-dax \
  --output-dir /data/agastyas/hidden-marker-lab/binary-full
```

Use `--resume` with a matching checkpoint or `--overwrite` to intentionally
replace an existing completed run. No API key is needed for the public
`Qwen/Qwen3-14B` checkpoint.

## Run the dashboard

Requires Node.js 22.13 or newer.

```bash
npm install
npm run dev
```

Use `npm test` for the deployment build and artifact checks.

The dashboard now begins with a plain-language study overview at `/`. The four
experiment pages follow a shared question, design, measurement, result, and
interpretation structure; Experiment 1 is at `/label-choice`.

## Experiment 4: OOD generalization and circuit analysis

Experiment 4 freezes the three Experiment 3 adapters and evaluates distribution
shifts separately before combining them. It also trains three new adapters whose
directions are restricted to one 2,560-dimensional half of a fixed dense
orthonormal basis, reserving the exactly orthogonal half for final evaluation.
The frozen protocol is `experiment/experiment4_protocol.md`.

Across the new adapters, accuracy was 95.8% on the training half and 94.7% on
the reserved half (1.1 percentage-point mean gap). The frozen original adapters
transferred to Student-t(3) directions (94.0%) and natural activation-difference
directions (94.9%), but not to a content-token injection site (49.9%), a crossed
test position (48.8%), or block 24 (49.9%). Prompt paraphrases reached 60.2% and
different within-episode carriers reached 72.4%.

The mechanistic analysis replicates block 22 head 17 as a causal marker-matching
component across three independently trained adapters. Its mean ablation reduced
accuracy by 7.2 percentage points, counterfactual activation patching reduced the
correct margin by 3.32 logits, and a six-head matcher group reduced accuracy by
25.8 percentage points versus 0.0 for a matched control group. These results do
not establish a general-purpose or phenomenological “introspection head.”
