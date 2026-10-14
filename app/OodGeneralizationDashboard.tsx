"use client";

import { useEffect, useMemo, useState } from "react";
import { ExperimentPrimer, ReadingKey } from "./ExperimentPrimer";
import { ResearchHeader } from "./ResearchHeader";

type Condition = {
  id: string;
  n_trials_per_seed: number;
  base_accuracy: number;
  mean_accuracy: number;
  min_accuracy: number;
  max_accuracy: number;
  mean_margin: number;
  episode_cluster_ci_95: [number, number];
  per_seed: Array<{ seed: number; accuracy: number; mean_margin: number; cluster_ci_95: [number, number] }>;
};

type SubspaceRun = {
  seed: number;
  train_span: { n_trials: number; accuracy: number; mean_margin: number; cluster_bootstrap_ci_95: [number, number] };
  holdout_span: { n_trials: number; accuracy: number; mean_margin: number; cluster_bootstrap_ci_95: [number, number] };
  holdout_gap: number;
  training: { steps: number; trainable_parameters: number; best_validation_accuracy: number; duration_seconds: number; gpu: string };
};

type HeadSummary = {
  layer: number;
  head: number;
  role_hypothesis: string;
  attention: { mean_match_probability_contrast: number; per_seed: number[] };
  mean_ablation: { mean_accuracy_drop: number; mean_margin_drop: number; per_seed_margin_drop: number[]; per_seed_general_lm_nll_change: number[] };
  counterfactual_patch: null | { mean_margin_reduction: number; mean_answer_flip_fraction: number; per_seed_margin_reduction: number[] };
};

type Results = {
  status: string;
  model_id: string;
  model_revision: string;
  fixed_adapter_conditions: Condition[];
  morph: Array<{ angle_degrees: number; n_trials_per_seed: number; mean_label_v2_minus_v1_logit: number; mean_fraction_preferring_v2: number }>;
  subspace: {
    basis: string;
    train_dimension: number;
    holdout_dimension: number;
    dot_product_numerical_check_max: number;
    success_gate: string;
    passed: boolean;
    mean_train_accuracy: number;
    mean_holdout_accuracy: number;
    mean_holdout_gap: number;
    runs: SubspaceRun[];
  };
  mechanistic: {
    verdict: string;
    discovery_trials_per_model: number;
    confirmation_trials_per_model: number;
    head_summaries: HeadSummary[];
    head_22_17_specificity: {
      trained_primary_match_contrast: number;
      trained_no_test_match_contrast: number;
      base_primary_match_contrast: number;
      base_counterfactual_patch_margin_reduction: number;
      trained_control_head_22_8_patch_margin_reduction: number;
    };
    matcher_group: { members: string[]; accuracy_drop: number; margin_drop: number; general_lm_nll_change: number };
    matched_control_group: { members: string[]; accuracy_drop: number; margin_drop: number; general_lm_nll_change: number };
    interpretation_limit: string;
  };
  artifacts: { ood_audit: string; subspace_audit: string; morph_audit: string; notebook: string; protocol: string };
};

type AuditTrial = {
  model_condition: string;
  condition: string;
  episode_id: number;
  test_vector: string;
  correct_option: "A" | "B";
  correct_label: string;
  labels_by_vector: { v1: string; v2: string };
  mapping_swap: boolean;
  demo_reverse: boolean;
  option_reverse: boolean;
  sentences?: string[];
  sentence: string;
  prompt_variant?: number;
  site_mode: string;
  prediction: "A" | "B";
  predicted_label: string;
  correct: boolean;
  logit_A: number;
  logit_B: number;
  margin: number;
  rho: number;
  realized_alpha: number;
  vector_family: string;
  pair_cosine: number;
  direction_mode: string;
  injection_layer: number;
  v1_hash: string;
  v2_hash: string;
};

const pct = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const signed = (value: number, digits = 3) => `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
const conditionNames: Record<string, string> = {
  id_reference: "ID reference",
  prompt_paraphrase: "Prompt paraphrase",
  different_carriers: "Different carriers",
  content_token: "All markers at content token",
  cross_position: "Test marker changes position",
  layer_16: "Injection at block 16",
  layer_24: "Injection at block 24",
  correlated_0_5: "Pair cosine 0.5",
  correlated_0_9: "Pair cosine 0.9",
  student_t_3: "Student-t(3) directions",
  natural_activation_difference: "Activation-difference directions",
  dense_train_half: "Dense basis · train half",
  dense_holdout_half: "Dense basis · opposite half",
  joint_ood: "Joint OOD shift",
};

function Metric({ label, value, detail, tone = "neutral" }: { label: string; value: string; detail: string; tone?: string }) {
  return <div className={`metric-card metric-${tone}`}><span className="eyebrow">{label}</span><strong>{value}</strong><span className="metric-detail">{detail}</span></div>;
}

function AccuracyBars({ rows }: { rows: Array<{ label: string; value: number; low?: number; high?: number; note?: string }> }) {
  return <div className="accuracy-chart"><div className="accuracy-chart-axis"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div><div className="ft-bars exp4-bars">{rows.map((row) => <div className="ft-bar" key={row.label}><div><span>{row.label}</span><small>{row.note}</small></div><div className="ft-track"><i className="ft-chance" /><b className={row.value < .55 ? "bar-fail" : row.value < .75 ? "bar-warn" : ""} style={{ width: `${row.value * 100}%` }} />{row.low !== undefined && row.high !== undefined && <em style={{ left: `${row.low * 100}%`, width: `${(row.high - row.low) * 100}%` }} />}</div><strong>{pct(row.value)}</strong></div>)}</div></div>;
}

function MorphPlot({ rows }: { rows: Results["morph"] }) {
  const width = 760, height = 260, left = 50, right = 20, top = 24, bottom = 42;
  const values = rows.map((row) => row.mean_label_v2_minus_v1_logit);
  const bound = Math.max(1, ...values.map(Math.abs));
  const x = (angle: number) => left + angle / 90 * (width - left - right);
  const y = (value: number) => top + (bound - value) / (2 * bound) * (height - top - bottom);
  return <div className="exp4-morph"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Mean v2 minus v1 logit contrast across angular morphs"><line x1={left} x2={width - right} y1={y(0)} y2={y(0)} className="morph-zero" />{[-bound, 0, bound].map((tick) => <g key={tick}><text x={left - 8} y={y(tick) + 3} textAnchor="end">{signed(tick, 1)}</text></g>)}<polyline points={rows.map((row) => `${x(row.angle_degrees)},${y(row.mean_label_v2_minus_v1_logit)}`).join(" ")} fill="none" className="morph-line" />{rows.map((row) => <g key={row.angle_degrees}><circle cx={x(row.angle_degrees)} cy={y(row.mean_label_v2_minus_v1_logit)} r="5" /><text x={x(row.angle_degrees)} y={height - 16} textAnchor="middle">{row.angle_degrees}°</text></g>)}</svg><div className="morph-caption"><span>v1 endpoint</span><span>intermediate angles have no categorical ground truth</span><span>v2 endpoint</span></div></div>;
}

function AuditLedgerContents({ results, ledger }: { results: Results; ledger: "ood" | "subspace" }) {
  const [trials, setTrials] = useState<AuditTrial[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [model, setModel] = useState("all");
  const [condition, setCondition] = useState("all");
  const [outcome, setOutcome] = useState("all");
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<AuditTrial | null>(null);
  const pageSize = 100;

  useEffect(() => {
    let active = true;
    const url = ledger === "ood" ? results.artifacts.ood_audit : results.artifacts.subspace_audit;
    fetch(url).then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    }).then(async (artifact: { trials?: AuditTrial[]; chunks?: string[] }) => {
      if (artifact.trials) return artifact.trials;
      if (!artifact.chunks) throw new Error("Ledger has neither trials nor chunks");
      const chunks = await Promise.all(artifact.chunks.map(async (chunkUrl) => {
        const response = await fetch(chunkUrl);
        if (!response.ok) throw new Error(`HTTP ${response.status} for ${chunkUrl}`);
        return response.json() as Promise<{ trials: AuditTrial[] }>;
      }));
      return chunks.flatMap((chunk) => chunk.trials);
    }).then((loadedTrials) => {
      if (!active) return;
      setTrials(loadedTrials); setSelected(loadedTrials[0] ?? null); setLoading(false);
    }).catch((reason: Error) => { if (active) { setError(reason.message); setLoading(false); } });
    return () => { active = false; };
  }, [ledger, results.artifacts.ood_audit, results.artifacts.subspace_audit]);

  const models = useMemo(() => [...new Set(trials.map((trial) => trial.model_condition))], [trials]);
  const conditions = useMemo(() => [...new Set(trials.map((trial) => trial.condition))], [trials]);
  const filtered = useMemo(() => trials.filter((trial) => {
    const haystack = `${trial.sentence} ${trial.correct_label} ${trial.predicted_label} ${trial.v1_hash} ${trial.v2_hash}`.toLowerCase();
    return (model === "all" || trial.model_condition === model)
      && (condition === "all" || trial.condition === condition)
      && (outcome === "all" || (outcome === "correct") === trial.correct)
      && (!query || haystack.includes(query.toLowerCase()));
  }), [trials, model, condition, outcome, query]);
  const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const visible = filtered.slice(page * pageSize, (page + 1) * pageSize);
  const update = (setter: (value: string) => void) => (value: string) => { setter(value); setPage(0); };

  return <div className="audit-workspace exp4-audit"><div className="trial-browser"><div className="filter-bar exp4-filter"><label className="search-field">Search sentence, label, or hash<input value={query} onChange={(event) => update(setQuery)(event.target.value)} placeholder="Filter every trial…" /></label><label>Model<select value={model} onChange={(event) => update(setModel)(event.target.value)}><option value="all">All models</option>{models.map((item) => <option key={item}>{item}</option>)}</select></label><label>Condition<select value={condition} onChange={(event) => update(setCondition)(event.target.value)}><option value="all">All conditions</option>{conditions.map((item) => <option key={item} value={item}>{conditionNames[item] ?? item}</option>)}</select></label><label>Outcome<select value={outcome} onChange={(event) => update(setOutcome)(event.target.value)}><option value="all">All</option><option value="correct">Correct</option><option value="incorrect">Incorrect</option></select></label><span className="result-count">{loading ? "Loading complete ledger…" : error ? `Load error · ${error}` : `${filtered.length.toLocaleString()} / ${trials.length.toLocaleString()}`}</span></div><div className="table-shell exp4-table-shell"><table><thead><tr><th>Model</th><th>Condition</th><th>Ep.</th><th>Sentence</th><th>Vector</th><th>Family</th><th>Layer / site</th><th>Correct</th><th>Pred.</th><th>Margin</th><th>Outcome</th></tr></thead><tbody>{visible.map((trial, index) => <tr key={`${trial.model_condition}-${trial.condition}-${trial.episode_id}-${trial.test_vector}-${Number(trial.mapping_swap)}-${Number(trial.demo_reverse)}-${Number(trial.option_reverse)}-${index}`} className={selected === trial ? "selected-row" : ""} onClick={() => setSelected(trial)}><td className="mono">{trial.model_condition}</td><td>{conditionNames[trial.condition] ?? trial.condition}</td><td className="mono">{trial.episode_id}</td><td className="sentence-cell">{trial.sentence}</td><td><i className={`vector-badge ${trial.test_vector}`}>{trial.test_vector}</i></td><td className="mono">{trial.vector_family}</td><td className="mono">{trial.injection_layer} · {trial.site_mode}</td><td className="mono">{trial.correct_option} · {trial.correct_label}</td><td className="mono">{trial.prediction} · {trial.predicted_label}</td><td className={`mono ${trial.margin >= 0 ? "positive-margin" : "negative-margin"}`}>{signed(trial.margin)}</td><td><span className={`outcome-dot ${trial.correct ? "is-correct" : "is-error"}`}><i />{trial.correct ? "correct" : "error"}</span></td></tr>)}</tbody></table></div><div className="ledger-pagination"><span>Page {page + 1} of {pages}</span><div><button disabled={page === 0} onClick={() => setPage((value) => value - 1)}>Previous 100</button><button disabled={page + 1 >= pages} onClick={() => setPage((value) => value + 1)}>Next 100</button></div></div></div>{selected && <aside className="trial-detail exp4-detail"><div className="detail-heading"><div><span className="eyebrow">{conditionNames[selected.condition] ?? selected.condition}</span><h3>{selected.test_vector} → {selected.correct_label}</h3></div><span className={`outcome-pill ${selected.correct ? "is-correct" : "is-error"}`}>{selected.correct ? "correct" : "error"}</span></div><div className="sentence-card"><span className="detail-label">Test carrier</span><p>{selected.sentence}</p><span className="sentence-note">block {selected.injection_layer} resid_pre · {selected.site_mode} · ρ {selected.rho}</span></div><div className="detail-block"><div className="block-title"><span>Intervention geometry</span><span>{selected.direction_mode}</span></div><dl className="detail-grid"><div><dt>family</dt><dd>{selected.vector_family}</dd></div><div><dt>pair cosine</dt><dd>{selected.pair_cosine.toFixed(2)}</dd></div><div><dt>realized α</dt><dd>{selected.realized_alpha.toFixed(3)}</dd></div><div><dt>prompt</dt><dd>variant {selected.prompt_variant ?? 0}</dd></div><div><dt>v1 hash</dt><dd>{selected.v1_hash}</dd></div><div><dt>v2 hash</dt><dd>{selected.v2_hash}</dd></div></dl></div><div className="detail-block"><div className="block-title"><span>In-context binding</span><span>{selected.option_reverse ? "options B/A" : "options A/B"}</span></div><dl className="detail-grid"><div><dt>v1 label</dt><dd>{selected.labels_by_vector.v1}</dd></div><div><dt>v2 label</dt><dd>{selected.labels_by_vector.v2}</dd></div><div><dt>mapping</dt><dd>{selected.mapping_swap ? "swapped" : "canonical"}</dd></div><div><dt>demo order</dt><dd>{selected.demo_reverse ? "reversed" : "canonical"}</dd></div></dl></div><div className="detail-block"><div className="block-title"><span>Next-token logits</span><span>{selected.prediction} selected</span></div><div className="exp4-logits"><div><span>A</span><b>{selected.logit_A.toFixed(3)}</b></div><div><span>B</span><b>{selected.logit_B.toFixed(3)}</b></div></div><div className="choice-line"><span>Correct-choice margin</span><b className={selected.margin >= 0 ? "positive-margin" : "negative-margin"}>{signed(selected.margin)}</b></div></div>{selected.sentences && <div className="detail-block"><span className="detail-label">Three visible carriers</span>{selected.sentences.map((sentence, index) => <p className="audit-carrier" key={`${sentence}-${index}`}>{index < 2 ? `Demo ${index + 1}` : "Test"} · {sentence}</p>)}</div>}</aside>}</div>;
}

function AuditLedger({ results }: { results: Results }) {
  const [ledger, setLedger] = useState<"ood" | "subspace">("ood");
  return <div className="exp4-ledger-wrap"><div className="ledger-tabs" role="tablist" aria-label="Trial ledgers"><button className={ledger === "ood" ? "active" : ""} onClick={() => setLedger("ood")}>Frozen-adapter OOD · 13,824</button><button className={ledger === "subspace" ? "active" : ""} onClick={() => setLedger("subspace")}>Disjoint-span · 24,576</button></div><AuditLedgerContents key={ledger} results={results} ledger={ledger} /></div>;
}

export function OodGeneralizationDashboard({ results: rawResults }: { results: unknown }) {
  const results = rawResults as Results;
  const condition = (id: string) => results.fixed_adapter_conditions.find((row) => row.id === id)!;
  const matcher = results.mechanistic.head_summaries.find((head) => head.layer === 22 && head.head === 17)!;
  const geometry = ["id_reference", "student_t_3", "natural_activation_difference", "correlated_0_5", "correlated_0_9"].map(condition);
  const language = ["prompt_paraphrase", "different_carriers"].map(condition);
  const site = ["content_token", "cross_position", "layer_16", "layer_24", "joint_ood"].map(condition);
  return <main className="exp4-page"><ResearchHeader active="ood" status="Complete · 7 training/evaluation runs" />
    <section className="exp4-hero"><div><span className="kicker">Experiment 4 · generalization and causal analysis</span><h1>What did the trained model <em>actually learn?</em></h1><p className="lede">We keep the trained adapters fixed, then change one part of the task at a time: the vector distribution, sentence wording, prompt, token position, or transformer layer. Separately, we intervene on attention heads to locate parts of the learned computation.</p><div className="hero-facts"><span><b>Generalization trials</b> 13,824</span><span><b>Orthogonal-span trials</b> 24,576</span><span><b>Independent adapters</b> 3 fixed + 3 new</span></div></div><div className="exp4-hero-verdict"><span className="eyebrow">Strongest positive result</span><strong>{pct(results.subspace.mean_holdout_accuracy)}</strong><p>Accuracy on a 2,560-dimensional vector subspace excluded from every training step.</p><div><span>trained subspace <b>{pct(results.subspace.mean_train_accuracy)}</b></span><i>→</i><span>accuracy gap <b>{(results.subspace.mean_holdout_gap * 100).toFixed(1)} points</b></span></div><small>Every seed passed the preregistered threshold</small></div></section>
    <ExperimentPrimer
      question="Does the learned rule extend beyond the exact training distribution?"
      idea="A model can score well while relying on a narrow cue. We therefore change vector geometry, language, token position, and layer separately. We also train new adapters on one half of an orthonormal basis and reserve the exactly perpendicular half for evaluation."
      fixed="The trained adapter weights during the stress tests, the label-binding task, deterministic next-token scoring, and counterbalancing."
      changed="One factor at a time: vector family or correlation, prompt wording, carrier sentences, injection token, or injection layer. A final condition changes several factors together."
      measured="Held-out accuracy and correct-answer logit margin. For candidate attention heads, we additionally measure attention patterns, ablation effects, counterfactual patching, and general-language loss."
      answer="The comparison rule generalizes strongly to new vector directions, but it is tied to where the intervention enters the computation. The mechanism is distributed across several heads rather than contained in one general-purpose introspection head."
    />

    <section className="result-section"><div className="section-heading"><div><span className="section-index">01</span><span className="eyebrow">The strongest generalization test</span><h2>Train on one vector subspace; test on a perpendicular one</h2></div><p>We split a fixed 5,120-dimensional orthonormal basis into two equal halves. Each new adapter sees directions only from the first half. The second half is exactly orthogonal and is excluded from training, validation, and checkpoint selection.</p></div><div className="metric-grid"><Metric label="Unseen half" value={pct(results.subspace.mean_holdout_accuracy)} detail="3 seeds · 4,096 trials each" tone="green" /><Metric label="Seen half" value={pct(results.subspace.mean_train_accuracy)} detail="same evaluation protocol" tone="green" /><Metric label="Accuracy gap" value={`${(results.subspace.mean_holdout_gap * 100).toFixed(1)} points`} detail="preregistered maximum · 10 points" tone="green" /><Metric label="Largest cross-half dot product" value={results.subspace.dot_product_numerical_check_max.toExponential(1)} detail="numerical zero in FP32" tone="neutral" /></div><article className="paper-panel exp4-span-panel"><div className="panel-heading"><div><span className="eyebrow">Three independent training runs</span><h3>Accuracy on the seen and unseen halves</h3></div><span className="panel-note">dashed rule · 50% chance</span></div><AccuracyBars rows={results.subspace.runs.flatMap((run) => [{ label: `seed ${run.seed} · seen half`, value: run.train_span.accuracy, low: run.train_span.cluster_bootstrap_ci_95[0], high: run.train_span.cluster_bootstrap_ci_95[1], note: "directions available during training" }, { label: `seed ${run.seed} · unseen half`, value: run.holdout_span.accuracy, low: run.holdout_span.cluster_bootstrap_ci_95[0], high: run.holdout_span.cluster_bootstrap_ci_95[1], note: `accuracy gap ${(run.holdout_gap * 100).toFixed(1)} points` }])} /><ReadingKey>Each pair of rows is one independently trained adapter. If the model memorized only the training subspace, the unseen-half rows should fall toward 50%.</ReadingKey><p className="figure-caption">The unseen half performs almost as well as the seen half. This supports a rule that extends beyond the directions used during training, although it remains one structured basis construction.</p></article></section>

    <section className="method-section exp4-ood-section"><div className="section-heading"><div><span className="section-index">02</span><span className="eyebrow">Frozen-adapter stress test</span><h2>Direction geometry transfers; site and prompt invariance do not</h2></div><p>The original three adapters were frozen before this suite. Each bar averages independent seeds and uses fresh episodes; confidence intervals cluster by episode.</p></div><div className="exp4-ood-grid"><article className="paper-panel"><div className="panel-heading"><div><span className="eyebrow">Vector geometry</span><h3>Unseen distributions and correlations</h3></div></div><AccuracyBars rows={geometry.map((row) => ({ label: conditionNames[row.id], value: row.mean_accuracy, low: row.episode_cluster_ci_95[0], high: row.episode_cluster_ci_95[1], note: `margin ${signed(row.mean_margin)}` }))} /></article><article className="paper-panel"><div className="panel-heading"><div><span className="eyebrow">Language</span><h3>Prompt and carrier changes</h3></div></div><AccuracyBars rows={language.map((row) => ({ label: conditionNames[row.id], value: row.mean_accuracy, low: row.episode_cluster_ci_95[0], high: row.episode_cluster_ci_95[1], note: `${row.n_trials_per_seed} trials / seed` }))} /></article><article className="paper-panel exp4-wide"><div className="panel-heading"><div><span className="eyebrow">Computation site</span><h3>Token position, injection depth, and joint shift</h3></div><span className="panel-note">trained site · block 20 final period</span></div><AccuracyBars rows={site.map((row) => ({ label: conditionNames[row.id], value: row.mean_accuracy, low: row.episode_cluster_ci_95[0], high: row.episode_cluster_ci_95[1], note: `${row.n_trials_per_seed} trials / seed` }))} /></article></div><div className="interpretation-note exp4-boundary"><span>Boundary of the result</span><p>Student-t(3) directions reached {pct(condition("student_t_3").mean_accuracy)} and natural activation-difference directions reached {pct(condition("natural_activation_difference").mean_accuracy)}. Moving all injections to a content token produced {pct(condition("content_token").mean_accuracy)}, and block 24 produced {pct(condition("layer_24").mean_accuracy)}. The learned access is direction-general but strongly site-anchored.</p></div></section>

    <section className="result-section"><div className="section-heading"><div><span className="section-index">03</span><span className="eyebrow">A continuous probe</span><h2>Rotate smoothly from v₁ toward v₂</h2></div><p>Instead of repeating exactly v₁ or v₂, we inject a weighted mixture. The endpoints have known labels; intermediate angles do not. We therefore plot the raw label preference rather than calling intermediate predictions correct or incorrect.</p></div><article className="paper-panel exp4-morph-panel"><div className="panel-heading"><div><span className="eyebrow">Angular morph</span><h3>Mean v₂-label score minus v₁-label score</h3></div><span className="panel-note">128 trials per angle per seed</span></div><MorphPlot rows={results.morph} /><ReadingKey>Negative values favor the label attached to v₁; positive values favor the label attached to v₂. A smooth crossing suggests graded comparison rather than two unrelated endpoint shortcuts.</ReadingKey></article></section>

    <section className="method-section exp4-mech"><div className="section-heading"><div><span className="section-index">04</span><span className="eyebrow">Mechanistic localization</span><h2>A reliable marker-matching head inside a redundant circuit</h2></div><p>Attention patterns nominated candidates on one adapter. Position-specific mean ablation and counterfactual test-token patching were then evaluated on held-out episodes and independent adapters.</p></div><div className="circuit-flow"><article><span>Block 20 · head 29</span><b>Marker processor</b><p>Builds an early low-rank representation of the injected residual delta.</p></article><i>→</i><article className="circuit-primary"><span>Block 22 · head 17</span><b>Marker matcher</b><p>Test-period attention selects the demonstration with the same injected direction.</p></article><i>→</i><article><span>Block 23 · head 36</span><b>Answer router</b><p>Routes test-position information into later answer computation.</p></article></div><div className="metric-grid exp4-mech-metrics"><Metric label="22.17 match contrast" value={matcher.attention.mean_match_probability_contrast.toFixed(3)} detail={`base ${results.mechanistic.head_22_17_specificity.base_primary_match_contrast.toFixed(3)} · no-test ${results.mechanistic.head_22_17_specificity.trained_no_test_match_contrast.toFixed(3)}`} tone="green" /><Metric label="Mean-ablation loss" value={`${(matcher.mean_ablation.mean_accuracy_drop * 100).toFixed(1)} pp`} detail={`${matcher.mean_ablation.mean_margin_drop.toFixed(2)} logits · 3/3 seeds`} tone="green" /><Metric label="Patch margin shift" value={matcher.counterfactual_patch ? matcher.counterfactual_patch.mean_margin_reduction.toFixed(2) : "—"} detail={`${matcher.counterfactual_patch ? pct(matcher.counterfactual_patch.mean_answer_flip_fraction) : "—"} answers flipped`} tone="green" /><Metric label="Six-head cluster" value={`${(results.mechanistic.matcher_group.accuracy_drop * 100).toFixed(1)} pp`} detail={`matched control ${(results.mechanistic.matched_control_group.accuracy_drop * 100).toFixed(1)} pp`} tone="ochre" /></div><div className="exp4-head-grid">{results.mechanistic.head_summaries.map((head) => <article className="paper-panel" key={`${head.layer}.${head.head}`}><div className="head-index">{head.layer}<span>.</span>{head.head}</div><div><span className="eyebrow">{head.role_hypothesis}</span><h3>{head.layer === 22 ? "Same-vector attention and causal identity" : head.layer === 20 ? "Causal feature construction" : "Downstream answer routing"}</h3><dl><div><dt>match contrast</dt><dd>{head.attention.mean_match_probability_contrast.toFixed(3)}</dd></div><div><dt>margin lost on ablation</dt><dd>{head.mean_ablation.mean_margin_drop.toFixed(2)}</dd></div><div><dt>counterfactual patch</dt><dd>{head.counterfactual_patch ? head.counterfactual_patch.mean_margin_reduction.toFixed(2) : "not run"}</dd></div></dl></div></article>)}</div><div className="interpretation-note"><span>Terminology</span><p>{results.mechanistic.verdict} {results.mechanistic.interpretation_limit}</p></div></section>

    <section className="audit-section"><div className="section-heading"><div><span className="section-index">05</span><span className="eyebrow">Complete trial ledgers</span><h2>Inspect every condition, vector, transform, and logit</h2></div><p>Switch between the 13,824 frozen-adapter OOD trials and 24,576 disjoint-span trials. Every record exposes the carrier, intervention family, basis half, layer/site, nonce binding, counterbalancing state, logits, realized scale, and FP32 vector fingerprints.</p></div><AuditLedger results={results} /></section>

    <section className="method-section"><div className="section-heading compact-heading"><div><span className="section-index">06</span><span className="eyebrow">Notebook and evidence</span><h2>Rationales, exercises, source, and complete artifacts</h2></div><p>The executed notebook follows the ARENA evidential ladder: capability, activation pattern, held-out causal ablation, counterfactual patching, controls, and weight-level diagnostics. It also states what would count as convincing positive or negative evidence.</p></div><div className="artifact-links exp4-artifacts"><a href="/data/introspection-head-analysis.html" target="_blank" rel="noreferrer">Open rendered analysis notebook <span>HTML · executed outputs and figures</span></a><a href="/data/introspection-head-analysis.ipynb" download>Download notebook <span>IPYNB · 27 cells</span></a><a href="/data/experiment4-protocol.md" download>Download frozen protocol <span>Markdown · claims and gates</span></a><a href="/data/experiment4-results.json" download>Download aggregate results <span>JSON · OOD, span, and circuit summaries</span></a><a href="/data/experiment4-ood-audit.json" download>Download OOD ledger <span>JSON · 13,824 trials</span></a><a href="/data/experiment4-subspace-audit.json" download>Download disjoint-span manifest <span>JSON · 24,576 trials in 3 seed chunks</span></a><a href="/data/experiment4-morph-audit.json" download>Download morph ledger <span>JSON · 2,688 diagnostics</span></a><a href="/data/mechanistic-introspection.py" download>Download mechanistic runner <span>Python · native Qwen3 hooks</span></a></div></section>
    <footer><div><span className="wordmark-mark">H</span><p><b>Hidden Marker Lab</b><br />Experiment 04 · OOD generalization and circuit analysis</p></div><p>{results.model_id} · revision {results.model_revision.slice(0, 10)}</p></footer>
  </main>;
}
