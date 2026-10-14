"use client";

import { useEffect, useMemo, useState } from "react";
import { ExperimentPrimer, ReadingKey } from "./ExperimentPrimer";
import { ResearchHeader } from "./ResearchHeader";

type Summary = {
  n_trials: number;
  n_correct: number;
  accuracy: number;
  ci_95_wilson_descriptive: [number, number];
  cluster_bootstrap_ci_95?: [number, number];
  mean_margin: number;
  complete_counterfactual_orbit_success_rate?: number;
  paired_transform_success_rate?: number;
};

type CurvePoint = {
  step: number;
  objective: string;
  mean_loss: number;
  mean_batch_accuracy: number;
  mean_rho: number;
  learning_rate: number;
};

type TrainingRun = {
  training_seed: number;
  condition: string;
  training: {
    gpu: string;
    steps: number;
    train_layers: number[];
    trainable_parameters: number;
    best_validation_accuracy: number;
    duration_seconds: number;
  };
  primary: Summary;
  curve: CurvePoint[];
};

type Condition = Summary & {
  primary: boolean;
  layer: number;
  rho: number;
  vector_family: string;
  carrier_mode: string;
  control: string;
};

type AggregateCondition = {
  layer: number;
  rho: number;
  vector_family: string;
  carrier_mode: string;
  control: string;
  mean_accuracy: number;
  min_accuracy: number;
  max_accuracy: number;
  accuracy_std_across_seeds: number;
  mean_margin: number;
  per_seed: Array<{ training_seed: number; accuracy: number; mean_margin: number; n_trials: number }>;
};

type Trial = {
  model_condition: string;
  episode_id: number;
  test_vector: "v1" | "v2";
  correct_option: "A" | "B";
  correct_label: string;
  labels_by_vector: { v1: string; v2: string };
  mapping_swap: boolean;
  demo_reverse: boolean;
  option_reverse: boolean;
  carrier_mode: string;
  sentence: string;
  control: string;
  prediction: "A" | "B";
  correct: boolean;
  logit_A: number;
  logit_B: number;
  margin: number;
  rho: number;
  realized_alpha: number;
  vector_family: string;
  injection_layer: number;
  v1_hash: string;
  v2_hash: string;
  predicted_label: string;
};

type Results = {
  status: string;
  model_id: string;
  evaluation_seed: number;
  protocol: { primary: string; training: string; interpretation_limit: string };
  aggregate: {
    base_accuracy: number;
    fixed_seed_mean_accuracy: number;
    fixed_seed_min_accuracy: number;
    fixed_seed_max_accuracy: number;
    fixed_seed_accuracy_std: number;
    random_layer_accuracy: number;
    paired_improvement: { mean_accuracy_difference: number; cluster_bootstrap_ci_95: [number, number]; cluster_unit: string };
    mean_counterfactual_orbit_success: number;
    all_fixed_seeds_above_75_percent: boolean;
  };
  base_primary: Summary;
  runs: TrainingRun[];
  fixed_condition_matrix: AggregateCondition[];
  random_layer_conditions: Condition[];
  candidate_tokens: Record<string, { id: number; decoded: string }>;
  audit: { trial_count: number; url: string; unit: string };
};

const pct = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const signed = (value: number, digits = 3) => `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
const compact = (value: number) => new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
const tones = ["#4a7c59", "#2a9d8f", "#b77525", "#a44e3a"];

function Metric({ label, value, detail, tone = "neutral" }: { label: string; value: string; detail: string; tone?: string }) {
  return <div className={`metric-card metric-${tone}`}><span className="eyebrow">{label}</span><strong>{value}</strong><span className="metric-detail">{detail}</span></div>;
}

function TrainingCurves({ runs }: { runs: TrainingRun[] }) {
  const width = 900;
  const height = 260;
  const x = (step: number) => 54 + step / Math.max(...runs.map((run) => run.training.steps)) * 816;
  const y = (accuracy: number) => 220 - accuracy * 190;
  return (
    <div className="training-chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Online training batch accuracy by optimization step">
        {[0, .25, .5, .75, 1].map((tick) => <g key={tick}><line x1="54" x2="870" y1={y(tick)} y2={y(tick)} className="chart-grid" /><text x="44" y={y(tick) + 4} textAnchor="end">{pct(tick, 0)}</text></g>)}
        <line x1={x(200)} x2={x(200)} y1="25" y2="220" className="phase-line" />
        <line x1={x(700)} x2={x(700)} y1="25" y2="220" className="phase-line" />
        <text x={x(100)} y="17" textAnchor="middle">localize</text><text x={x(450)} y="17" textAnchor="middle">same / different</text><text x={x(1600)} y="17" textAnchor="middle">bind arbitrary labels + anneal ρ</text>
        {runs.map((run, index) => <polyline key={run.training_seed} points={run.curve.map((point) => `${x(point.step)},${y(point.mean_batch_accuracy)}`).join(" ")} fill="none" stroke={tones[index]} strokeWidth="2" />)}
        {[0, 500, 1000, 1500, 2000, 2500].map((tick) => <text key={tick} x={x(tick)} y="244" textAnchor="middle">{tick}</text>)}
      </svg>
      <div className="chart-legend">{runs.map((run, index) => <span key={run.training_seed}><i style={{ background: tones[index] }} />seed {run.training_seed}{index === 3 ? " · layers 16/20/24" : ""}</span>)}</div>
    </div>
  );
}

function AccuracyBars({ rows, chance = true }: { rows: Array<{ label: string; value: number; low?: number; high?: number; note?: string }>; chance?: boolean }) {
  return <div className="accuracy-chart"><div className="accuracy-chart-axis"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div><div className="ft-bars">{rows.map((row) => <div className="ft-bar" key={row.label}><div><span>{row.label}</span><small>{row.note}</small></div><div className="ft-track">{chance && <i className="ft-chance" />}<b style={{ width: `${Math.max(0, Math.min(1, row.value)) * 100}%` }} />{row.low !== undefined && row.high !== undefined && <em style={{ left: `${row.low * 100}%`, width: `${(row.high - row.low) * 100}%` }} />}</div><strong>{pct(row.value)}</strong></div>)}</div></div>;
}

function AuditLedger({ config }: { config: Results["audit"] }) {
  const [trials, setTrials] = useState<Trial[]>([]);
  const [error, setError] = useState("");
  const [model, setModel] = useState("all");
  const [vector, setVector] = useState("all");
  const [outcome, setOutcome] = useState("all");
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<Trial | null>(null);
  const pageSize = 100;

  useEffect(() => {
    fetch(config.url).then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    }).then((artifact: { trials: Trial[] }) => {
      setTrials(artifact.trials);
      setSelected(artifact.trials[0] ?? null);
    }).catch((reason: Error) => setError(reason.message));
  }, [config.url]);

  const models = useMemo(() => [...new Set(trials.map((trial) => trial.model_condition))], [trials]);
  const filtered = useMemo(() => trials.filter((trial) => {
    const text = `${trial.sentence} ${trial.correct_label} ${trial.predicted_label} ${trial.v1_hash} ${trial.v2_hash}`.toLowerCase();
    return (model === "all" || trial.model_condition === model)
      && (vector === "all" || trial.test_vector === vector)
      && (outcome === "all" || (outcome === "correct") === trial.correct)
      && (!query || text.includes(query.toLowerCase()));
  }), [trials, model, vector, outcome, query]);
  const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const visible = filtered.slice(page * pageSize, (page + 1) * pageSize);
  const update = (setter: (value: string) => void) => (value: string) => { setter(value); setPage(0); };
  const selectedLogitMax = selected ? Math.max(Math.abs(selected.logit_A), Math.abs(selected.logit_B), 1) : 1;

  return <div className="audit-workspace ft-audit">
    <div className="trial-browser">
      <div className="filter-bar ft-filter">
        <label className="search-field">Search sentence, label, or hash<input value={query} onChange={(event) => update(setQuery)(event.target.value)} placeholder="Filter the ledger…" /></label>
        <label>Model<select value={model} onChange={(event) => update(setModel)(event.target.value)}><option value="all">All models</option>{models.map((item) => <option key={item}>{item}</option>)}</select></label>
        <label>Vector<select value={vector} onChange={(event) => update(setVector)(event.target.value)}><option value="all">Both</option><option value="v1">v1</option><option value="v2">v2</option></select></label>
        <label>Outcome<select value={outcome} onChange={(event) => update(setOutcome)(event.target.value)}><option value="all">All</option><option value="correct">Correct</option><option value="incorrect">Incorrect</option></select></label>
        <span className="result-count">{trials.length ? `${filtered.length.toLocaleString()} / ${trials.length.toLocaleString()}` : error ? `Load error · ${error}` : `Loading ${config.trial_count.toLocaleString()} trials…`}</span>
      </div>
      <div className="table-shell ft-table-shell"><table><thead><tr><th>Model</th><th>Ep.</th><th>Sentence</th><th>Vector</th><th>Map</th><th>Demo</th><th>Options</th><th>Correct</th><th>Pred.</th><th>Margin</th><th>Outcome</th></tr></thead><tbody>{visible.map((trial) => <tr key={`${trial.model_condition}-${trial.episode_id}-${trial.test_vector}-${Number(trial.mapping_swap)}-${Number(trial.demo_reverse)}-${Number(trial.option_reverse)}`} onClick={() => setSelected(trial)} className={selected === trial ? "selected-row" : ""}><td className="mono">{trial.model_condition}</td><td className="mono">{trial.episode_id}</td><td className="sentence-cell">{trial.sentence}</td><td><i className={`vector-badge ${trial.test_vector}`}>{trial.test_vector}</i></td><td className="mono">{trial.mapping_swap ? "swap" : "base"}</td><td className="mono">{trial.demo_reverse ? "rev" : "base"}</td><td className="mono">{trial.option_reverse ? "B/A" : "A/B"}</td><td className="mono">{trial.correct_option} · {trial.correct_label}</td><td className="mono">{trial.prediction} · {trial.predicted_label}</td><td className={`mono ${trial.margin >= 0 ? "positive-margin" : "negative-margin"}`}>{signed(trial.margin)}</td><td><span className={`outcome-dot ${trial.correct ? "is-correct" : "is-error"}`}><i />{trial.correct ? "correct" : "error"}</span></td></tr>)}</tbody></table></div>
      <div className="ledger-pagination"><span>Page {page + 1} of {pages}</span><div><button disabled={page === 0} onClick={() => setPage((value) => value - 1)}>Previous 100</button><button disabled={page + 1 >= pages} onClick={() => setPage((value) => value + 1)}>Next 100</button></div></div>
    </div>
    {selected && <aside className="trial-detail ft-detail">
      <div className="detail-heading"><div><span className="eyebrow">{selected.model_condition} · episode {selected.episode_id}</span><h3>{selected.test_vector} → {selected.correct_label}</h3></div><span className={`outcome-pill ${selected.correct ? "is-correct" : "is-error"}`}>{selected.correct ? "correct" : "error"}</span></div>
      <div className="sentence-card"><span className="detail-label">Held-out carrier</span><p>{selected.sentence.slice(0, -1)}<mark>.</mark></p><span className="sentence-note">block {selected.injection_layer} resid_pre · ρ {selected.rho} · α {selected.realized_alpha.toFixed(3)}</span></div>
      <div className="detail-block"><div className="block-title"><span>In-context binding</span><span className="mono">test {selected.test_vector}</span></div><dl className="detail-grid"><div><dt>v1 label</dt><dd>{selected.labels_by_vector.v1}</dd></div><div><dt>v2 label</dt><dd>{selected.labels_by_vector.v2}</dd></div><div><dt>mapping</dt><dd>{selected.mapping_swap ? "swapped" : "canonical"}</dd></div><div><dt>demo order</dt><dd>{selected.demo_reverse ? "v2 → v1" : "v1 → v2"}</dd></div><div><dt>option order</dt><dd>{selected.option_reverse ? "reversed" : "canonical"}</dd></div><div><dt>family</dt><dd>{selected.vector_family}</dd></div></dl></div>
      <div className="detail-block"><div className="block-title"><span>Next-token logits</span><span>{selected.prediction} selected</span></div>{(["A", "B"] as const).map((option) => { const value = option === "A" ? selected.logit_A : selected.logit_B; return <div className="logit-row" key={option}><span>{option} · {option === "A" ? (selected.option_reverse ? selected.labels_by_vector.v2 : selected.labels_by_vector.v1) : (selected.option_reverse ? selected.labels_by_vector.v1 : selected.labels_by_vector.v2)}</span><div><i className={option === "A" ? "bar-a" : "bar-b"} style={{ width: `${Math.abs(value) / selectedLogitMax * 100}%` }} /></div><b>{value.toFixed(3)}</b></div>; })}<div className="choice-line"><span>Correct-choice margin</span><b className={selected.margin >= 0 ? "positive-margin" : "negative-margin"}>{signed(selected.margin)}</b></div></div>
      <div className="detail-block"><span className="detail-label">FP32 vector fingerprints</span><code className="ft-hash">v1 {selected.v1_hash}</code><code className="ft-hash">v2 {selected.v2_hash}</code></div>
    </aside>}
  </div>;
}

export function IntrospectionTrainingDashboard({ results: rawResults }: { results: unknown }) {
  const results = rawResults as Results;
  const fixedRuns = results.runs.filter((run) => run.condition === "fixed layer 20");
  const randomRun = results.runs.find((run) => run.condition !== "fixed layer 20")!;
  const primaryRows = [
    { label: "Base Qwen3-14B", value: results.aggregate.base_accuracy, low: results.base_primary.cluster_bootstrap_ci_95?.[0], high: results.base_primary.cluster_bootstrap_ci_95?.[1], note: "no adapter" },
    ...fixedRuns.map((run) => ({ label: `Fine-tuned · seed ${run.training_seed}`, value: run.primary.accuracy, low: run.primary.cluster_bootstrap_ci_95?.[0], high: run.primary.cluster_bootstrap_ci_95?.[1], note: "trained + tested at block 20" })),
    { label: `Cross-layer · seed ${randomRun.training_seed}`, value: randomRun.primary.accuracy, low: randomRun.primary.cluster_bootstrap_ci_95?.[0], high: randomRun.primary.cluster_bootstrap_ci_95?.[1], note: "trained across blocks 16 / 20 / 24" },
  ];
  const strength = results.fixed_condition_matrix.filter((row) => row.layer === 20 && row.vector_family === "gaussian" && row.carrier_mode === "same" && row.control === "primary").sort((a, b) => a.rho - b.rho);
  const controls = results.fixed_condition_matrix.filter((row) => row.layer === 20 && row.rho === .1 && row.vector_family === "gaussian" && row.carrier_mode === "same" && row.control !== "primary");
  const ood = results.fixed_condition_matrix.filter((row) => row.layer === 20 && row.rho === .1 && row.control === "primary" && (row.vector_family !== "gaussian" || row.carrier_mode !== "same"));
  const layers = results.fixed_condition_matrix.filter((row) => row.rho === .1 && row.vector_family === "gaussian" && row.carrier_mode === "same" && row.control === "primary").sort((a, b) => a.layer - b.layer);
  const randomLayerSites = results.random_layer_conditions.filter((row) => row.rho === .1 && row.vector_family === "gaussian" && row.carrier_mode === "same" && row.control === "primary").sort((a, b) => a.layer - b.layer);
  const improvement = results.aggregate.paired_improvement;
  const successful = results.aggregate.all_fixed_seeds_above_75_percent;

  return <main>
    <ResearchHeader active="training" status="Complete · 4 H100 runs" />
    <section className="ft-hero"><div><span className="kicker">Experiment 3 · online training with fresh interventions</span><h1>Can fine-tuning teach the model to <em>compare hidden markers?</em></h1><p className="lede">Every training episode uses new vectors and arbitrary new labels. The model must infer the label mapping from the examples, compare the hidden test intervention with them, and answer under counterbalanced presentation orders.</p><div className="hero-facts"><span><b>Replication</b> {fixedRuns.length} independent seeds</span><span><b>Evaluation</b> {results.base_primary.n_trials.toLocaleString()} trials per model</span><span><b>Presentation checks</b> 8 versions of every episode</span></div></div><div className="ft-verdict"><span className="eyebrow">Preregistered success criterion</span><strong>{successful ? "Passed" : "Not passed"}</strong><p>{successful ? "Every fixed-layer seed exceeded 75% on examples excluded from training." : "At least one fixed-layer seed did not exceed 75% on the held-out evaluation."}</p><div><span>untrained <b>{pct(results.aggregate.base_accuracy)}</b></span><i>→</i><span>fine-tuned mean <b>{pct(results.aggregate.fixed_seed_mean_accuracy)}</b></span></div></div></section>
    <ExperimentPrimer
      question="Can fine-tuning teach the model to compare arbitrary hidden interventions?"
      idea="Training examples are generated online. The model cannot memorize a finite list of vectors or labels because each episode uses fresh directions, sentences, nonce labels, mappings, and presentation orders."
      fixed="Qwen3-14B’s original weights, the intervention operation, and the answer-scoring rule. Only rank-16 LoRA updates downstream of the injection are trainable."
      changed="Fresh intervention pairs and arbitrary label bindings are sampled throughout training. The curriculum gradually reduces intervention strength."
      measured="Accuracy on held-out sentences, labels, vector pairs, mappings, demonstration orders, and option orders. Every episode is evaluated under eight counterfactual presentations."
      answer={`Yes at the trained location. Three independent adapters averaged ${pct(results.aggregate.fixed_seed_mean_accuracy)}, versus ${pct(results.aggregate.base_accuracy)} for the untrained model.`}
    />
    <section className="result-section"><div className="section-heading"><div><span className="section-index">01</span><span className="eyebrow">Main result</span><h2>Three independently trained adapters learned the task</h2></div><p>For each underlying episode, we swap the label mapping, reverse the demonstrations, reverse the answer options, and test both vectors. This prevents a fixed label, position, or option preference from looking like success.</p></div>
      <div className="metric-grid"><Metric label="Fixed-seed mean" value={pct(results.aggregate.fixed_seed_mean_accuracy)} detail={`range ${pct(results.aggregate.fixed_seed_min_accuracy)}–${pct(results.aggregate.fixed_seed_max_accuracy)}`} tone={successful ? "green" : "red"} /><Metric label="Base model" value={pct(results.aggregate.base_accuracy)} detail={`${results.base_primary.n_trials.toLocaleString()} matched trials`} tone="ochre" /><Metric label="Paired gain" value={signed(improvement.mean_accuracy_difference * 100, 1) + " pp"} detail={`95% CI ${signed(improvement.cluster_bootstrap_ci_95[0] * 100, 1)} to ${signed(improvement.cluster_bootstrap_ci_95[1] * 100, 1)} pp`} tone="green" /><Metric label="Complete orbits" value={pct(results.aggregate.mean_counterfactual_orbit_success)} detail="all 8 transforms correct" tone="green" /></div>
      <div className="paper-panel ft-primary-panel"><div className="panel-heading"><div><span className="eyebrow">Held-out accuracy</span><h3>Base model, three replications, and cross-layer training</h3></div><span className="panel-note">dashed rule · 50% chance</span></div><AccuracyBars rows={primaryRows} /><ReadingKey>Bar length is accuracy. Thin black-ended intervals are 95% episode-cluster bootstrap intervals. The dashed vertical line is chance.</ReadingKey><p className="figure-caption">The cross-layer run is separate because it was trained at blocks 16, 20, and 24 instead of only block 20.</p></div>
    </section>
    <section className="method-section"><div className="section-heading"><div><span className="section-index">02</span><span className="eyebrow">How training works</span><h2>Teach simpler operations before the full task</h2></div><p>Phase 1 asks where an intervention occurred. Phase 2 asks whether two interventions match. Phase 3 combines comparison with arbitrary in-context labels while gradually shrinking the intervention. New data are generated at every step.</p></div><article className="paper-panel ft-chart-panel"><div className="panel-heading"><div><span className="eyebrow">Training curves</span><h3>Batch accuracy over 2,500 optimization steps</h3></div><span className="panel-note">one line per training seed</span></div><TrainingCurves runs={results.runs} /><ReadingKey>The x-axis is optimization step and the y-axis is batch accuracy. Vertical dashed lines mark curriculum transitions; sharp changes at those boundaries reflect a harder objective.</ReadingKey></article><div className="ft-run-cards">{results.runs.map((run, index) => <article key={run.training_seed}><i style={{ background: tones[index] }} /><span className="eyebrow">seed {run.training_seed}</span><b>{run.condition}</b><dl><div><dt>best validation</dt><dd>{pct(run.training.best_validation_accuracy)}</dd></div><div><dt>parameters</dt><dd>{compact(run.training.trainable_parameters)}</dd></div><div><dt>steps</dt><dd>{run.training.steps.toLocaleString()}</dd></div><div><dt>GPU</dt><dd>{run.training.gpu}</dd></div></dl></article>)}</div></section>
    <section className="result-section ft-controls"><div className="section-heading"><div><span className="section-index">03</span><span className="eyebrow">Confound suite</span><h2>What the learned behavior depends on</h2></div><p>Null interventions, sign reversal, an orthogonal test vector, non-Gaussian vector families, changed carriers, and off-site layers test whether accuracy can be explained by label or option shortcuts.</p></div><div className="ft-control-grid"><article className="paper-panel"><div className="panel-heading"><div><span className="eyebrow">Dose response</span><h3>Intervention strength ρ</h3></div><span className="panel-note">fixed-seed mean</span></div><AccuracyBars rows={strength.map((row) => ({ label: `ρ = ${row.rho.toFixed(2)}`, value: row.mean_accuracy, note: `range ${pct(row.min_accuracy)}–${pct(row.max_accuracy)}` }))} /></article><article className="paper-panel"><div className="panel-heading"><div><span className="eyebrow">Negative controls</span><h3>Remove or alter the test marker</h3></div><span className="panel-note">target ρ = 0.10</span></div><AccuracyBars rows={controls.map((row) => ({ label: row.control, value: row.mean_accuracy, note: `margin ${signed(row.mean_margin)}` }))} /></article><article className="paper-panel"><div className="panel-heading"><div><span className="eyebrow">Distribution shift</span><h3>Unseen vectors and carriers</h3></div><span className="panel-note">target ρ = 0.10</span></div><AccuracyBars rows={ood.map((row) => ({ label: row.vector_family === "gaussian" ? "different carriers" : row.vector_family, value: row.mean_accuracy, note: `${row.carrier_mode} carrier mode` }))} /></article><article className="paper-panel"><div className="panel-heading"><div><span className="eyebrow">Site specificity</span><h3>Injection block at evaluation</h3></div><span className="panel-note">fixed vs cross-layer training</span></div><AccuracyBars rows={[...layers.map((row) => ({ label: `fixed · block ${row.layer}`, value: row.mean_accuracy, note: row.layer === 20 ? "trained site" : "unseen site" })), ...randomLayerSites.map((row) => ({ label: `cross · block ${row.layer}`, value: row.accuracy, note: "trained site" }))]} /></article></div><div className="interpretation-note ft-boundary"><span>Interpretation boundary</span><p>{results.protocol.interpretation_limit} Strong primary accuracy is evidence for intervention-dependent computation only if the null and counterfactual controls behave as predicted; those results are reported above rather than treated as ancillary.</p></div></section>
    <section className="audit-section"><div className="section-heading"><div><span className="section-index">04</span><span className="eyebrow">Complete primary ledger</span><h2>Audit every model, vector, transform, and logit</h2></div><p>{results.audit.trial_count.toLocaleString()} records. Each row exposes the carrier, vector identity, nonce binding, counterbalancing state, realized injection scale, next-token logits, and FP32 vector fingerprints.</p></div><AuditLedger config={results.audit} /></section>
    <section className="method-section"><div className="section-heading compact-heading"><div><span className="section-index">05</span><span className="eyebrow">Reproducibility</span><h2>Protocol and downloadable evidence</h2></div><p>{results.protocol.primary}</p></div><div className="method-flow"><div><span>1</span><b>Sample online</b><p>Fresh FP32 QR pairs</p></div><i>→</i><div><span>2</span><b>Inject</b><p>resid_pre final period</p></div><i>→</i><div><span>3</span><b>Bind in context</b><p>held-out nonce labels</p></div><i>→</i><div><span>4</span><b>Score A / B</b><p>one causal forward pass</p></div></div><div className="artifact-links ft-artifacts"><a href="/data/introspection-training-results.json" download>Download aggregate results <span>JSON · summaries and training curves</span></a><a href="/data/introspection-training-audit.json" download>Download complete trial ledger <span>JSON · {results.audit.trial_count.toLocaleString()} primary records</span></a></div></section>
    <footer><div><span className="wordmark-mark">H</span><p><b>Hidden Marker Lab</b><br />Experiment 03 · functional access after fine-tuning</p></div><p>Evaluation seed {results.evaluation_seed} · {results.model_id}</p></footer>
  </main>;
}
