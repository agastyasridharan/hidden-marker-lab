"use client";

import { useMemo, useState } from "react";
import { ExperimentPrimer, ReadingKey } from "./ExperimentPrimer";
import { ResearchHeader } from "./ResearchHeader";

type BinaryTrial = {
  trial_id: string;
  episode_id: number;
  sentence: string;
  test_vector: "v1" | "v2";
  marker_status: "dax" | "not-dax";
  correct_answer: "YES" | "NO";
  before_logit_YES: number;
  before_logit_NO: number;
  before_raw_YES_minus_NO: number;
  before_prediction: "YES" | "NO";
  before_margin: number;
  before_correct: boolean;
  after_logit_YES: number;
  after_logit_NO: number;
  after_raw_YES_minus_NO: number;
  after_prediction: "YES" | "NO";
  after_margin: number;
  after_correct: boolean;
  delta_logit_YES: number;
  delta_logit_NO: number;
  delta_raw_yes_minus_no: number;
  correct_evidence_shift: number;
  prediction_switched: boolean;
};

type VectorSummary = {
  sha256: string;
  norm: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  first_12_coordinates: number[];
};

type BinaryEpisode = {
  episode_id: number;
  sentence: string;
  vector_seed: number;
  vector_pair: {
    v1: VectorSummary;
    v2: VectorSummary;
    dot_product: number;
    cosine_similarity: number;
  };
  injection_sites: {
    before: Array<{ role: string; position: number; token_id: number; decoded_piece: string; vector: string; binary_answer: string }>;
    after: Array<{ role: string; position: number; token_id: number; decoded_piece: string; vector: string; binary_answer: string }>;
  };
  prompt_token_counts: { before: number; after: number };
  trials: BinaryTrial[];
};

type StageSummary = {
  n_trials: number;
  n_correct: number;
  accuracy: number;
  ci_95: { method: string; level: number; low: number; high: number };
  mean_margin: number;
  median_margin: number;
  margin_std: number;
  margin_min: number;
  margin_max: number;
  prediction_counts: { YES: number; NO: number };
};

type BinaryResults = {
  status: string;
  run: {
    started_at_utc: string;
    completed_at_utc: string;
    duration_seconds: number;
    seed: number;
    gpu: string;
    model_id: string;
    model_revision: string;
    model_dtype: string;
    evaluation_mode: boolean;
    sampling: boolean;
    torch_version: string;
    transformers_version: string;
  };
  design: {
    episodes_planned: number;
    trials_planned: number;
    rho: number;
    alpha: number;
    layer_index_zero_based: number;
    num_layers: number;
    site: string;
    target: string;
    hidden_size: number;
    before_probe: { definition: string; caveat: string; one_forward_per_episode: boolean; shared_across_paired_v1_v2_trials: boolean };
    chat_template: string;
  };
  system_prompt: string;
  tokenization: {
    format: string;
    YES: { id: number; requested_text: string; decoded_piece: string };
    NO: { id: number; requested_text: string; decoded_piece: string };
    verified_single_token: boolean;
  };
  calibration: { count: number; median_residual_norm: number; alpha: number };
  aggregate: {
    n_trials: number;
    chance_accuracy: number;
    before: StageSummary;
    after: StageSummary;
    mean_correct_evidence_shift: number;
    median_correct_evidence_shift: number;
    mean_raw_yes_minus_no_shift: number;
    prediction_switches: number;
    by_test_vector: Record<"v1" | "v2", StageSummary & {
      before: StageSummary;
      mean_correct_evidence_shift: number;
      mean_raw_yes_minus_no_shift: number;
    }>;
  };
  episodes: BinaryEpisode[];
};

type OutcomeFilter = "all" | "correct" | "incorrect";
type VectorFilter = "all" | "v1" | "v2";

const signed = (value: number, digits = 3) => `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
const pct = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;

function BinaryMetric({ label, value, detail, tone = "neutral" }: { label: string; value: string; detail: string; tone?: "neutral" | "green" | "red" | "ochre" }) {
  return <div className={`metric-card metric-${tone}`}><span className="eyebrow">{label}</span><strong>{value}</strong><span className="metric-detail">{detail}</span></div>;
}

function mean(trials: BinaryTrial[], key: keyof BinaryTrial) {
  return trials.reduce((sum, trial) => sum + Number(trial[key]), 0) / trials.length;
}

function BeforeAfterRow({ label, before, after, tone, max }: { label: string; before: number; after: number; tone: string; max: number }) {
  return (
    <div className="ba-row">
      <span className={`answer-token ${tone}`}>{label}</span>
      <div className="ba-track"><i className={`ba-before ${tone}`} style={{ width: `${before / max * 100}%` }} /><b>{before.toFixed(3)}</b></div>
      <div className="ba-track"><i className={`ba-after ${tone}`} style={{ width: `${after / max * 100}%` }} /><b>{after.toFixed(3)}</b></div>
      <span className={`ba-delta ${after - before >= 0 ? "positive-margin" : "negative-margin"}`}>{signed(after - before)}</span>
    </div>
  );
}

function BeforeAfterBars({ trials }: { trials: BinaryTrial[] }) {
  const values = {
    beforeYES: mean(trials, "before_logit_YES"),
    beforeNO: mean(trials, "before_logit_NO"),
    afterYES: mean(trials, "after_logit_YES"),
    afterNO: mean(trials, "after_logit_NO"),
  };
  const max = Math.max(...Object.values(values));
  return (
    <div className="before-after-bars">
      <div className="ba-header"><span /><span>Before · prior probe</span><span>After · sentence read</span><span>Δ logit</span></div>
      <BeforeAfterRow label="YES" before={values.beforeYES} after={values.afterYES} tone="yes-tone" max={max} />
      <BeforeAfterRow label="NO" before={values.beforeNO} after={values.afterNO} tone="no-tone" max={max} />
      <div className="raw-shift-line"><span>Mean YES − NO</span><b>{signed(values.beforeYES - values.beforeNO)}</b><i>→</i><b>{signed(values.afterYES - values.afterNO)}</b><strong>{signed((values.afterYES - values.afterNO) - (values.beforeYES - values.beforeNO))}</strong></div>
    </div>
  );
}

function ShiftPlot({ episodes }: { episodes: BinaryEpisode[] }) {
  const trials = episodes.flatMap((episode) => episode.trials);
  const maxAbs = Math.max(...trials.map((trial) => Math.abs(trial.delta_raw_yes_minus_no)), 1);
  return (
    <div className="shift-plot">
      <div className="shift-axis-label shift-top">+{maxAbs.toFixed(2)}</div><div className="shift-axis-label shift-zero">0</div><div className="shift-axis-label shift-bottom">−{maxAbs.toFixed(2)}</div>
      <div className="shift-dot-field" role="img" aria-label="Change in YES minus NO logit by episode and test vector">
        <div className="shift-zero-rule" />
        {episodes.map((episode) => episode.trials.map((trial) => <i key={trial.trial_id} className={`trial-shift-dot ${trial.test_vector}`} style={{ left: `${(episode.episode_id - 1) / 63 * 100}%`, top: `${50 - trial.delta_raw_yes_minus_no / maxAbs * 40}%` }} title={`${trial.trial_id}: ${signed(trial.delta_raw_yes_minus_no)}`} />))}
        {[1, 16, 32, 48, 64].map((episode) => <span key={episode} className="shift-x-label" style={{ left: `${(episode - 1) / 63 * 100}%` }}>{episode}</span>)}
      </div>
    </div>
  );
}

function BinaryMatrix({ episodes, selected, onSelect }: { episodes: BinaryEpisode[]; selected: string; onSelect: (id: string) => void }) {
  return <div className="trial-matrix">{episodes.map((episode) => <div className="episode-cell" key={episode.episode_id}><span className="episode-number">E{String(episode.episode_id).padStart(2, "0")}</span>{episode.trials.map((trial) => <button key={trial.trial_id} className={`${trial.after_correct ? "correct-cell" : "error-cell"} ${selected === trial.trial_id ? "selected-cell" : ""}`} onClick={() => onSelect(trial.trial_id)} title={`${trial.trial_id} · before ${trial.before_prediction} · after ${trial.after_prediction}`}><span>{trial.test_vector === "v1" ? "v₁" : "v₂"}</span><b>{trial.after_prediction}</b></button>)}</div>)}</div>;
}

function VectorStrip({ coordinates }: { coordinates: number[] }) {
  const maxAbs = Math.max(...coordinates.map(Math.abs), .001);
  return <div className="vector-strip">{coordinates.map((coordinate, index) => <span className="coordinate" key={index}><i className={coordinate >= 0 ? "positive" : "negative"} style={{ height: `${Math.max(8, Math.abs(coordinate) / maxAbs * 46)}%` }} /></span>)}</div>;
}

function StageLogits({ title, yes, no, prediction }: { title: string; yes: number; no: number; prediction: string }) {
  const max = Math.max(yes, no, 1);
  return <div className="stage-card"><div className="stage-card-head"><span>{title}</span><b>{prediction}</b></div><div className="stage-logit"><span>YES</span><div><i className="yes-bar" style={{ width: `${yes / max * 100}%` }} /></div><b>{yes.toFixed(3)}</b></div><div className="stage-logit"><span>NO</span><div><i className="no-bar" style={{ width: `${no / max * 100}%` }} /></div><b>{no.toFixed(3)}</b></div><p>YES − NO <strong>{signed(yes - no)}</strong></p></div>;
}

function BinaryDetail({ trial, episode, alpha }: { trial: BinaryTrial; episode: BinaryEpisode; alpha: number }) {
  const vector = episode.vector_pair[trial.test_vector];
  const testSite = episode.injection_sites.after.find((site) => site.role === "test")!;
  return <aside className="trial-detail binary-detail">
    <div className="detail-heading"><div><span className="eyebrow">Selected trial</span><h3>{trial.trial_id}</h3></div><span className={`outcome-pill ${trial.after_correct ? "is-correct" : "is-error"}`}>{trial.after_correct ? "Correct" : "Incorrect"}</span></div>
    <div className="sentence-card"><span className="detail-label">Carrier · repeated 3×</span><p>{trial.sentence.slice(0, -1)}<mark>.</mark></p><span className="sentence-note">Test injection: {trial.test_vector === "v1" ? "v₁ · dax" : "v₂ · not-dax"}</span></div>
    <div className="detail-block stage-comparison"><div className="block-title"><span>YES / NO logits</span><span className="mono">raw model logits</span></div><div className="stage-grid"><StageLogits title="Before" yes={trial.before_logit_YES} no={trial.before_logit_NO} prediction={trial.before_prediction} /><StageLogits title="After" yes={trial.after_logit_YES} no={trial.after_logit_NO} prediction={trial.after_prediction} /></div><div className="evidence-shift"><span>Δ(YES − NO)</span><strong>{signed(trial.delta_raw_yes_minus_no)}</strong><small>{trial.prediction_switched ? "prediction switched" : "no prediction switch"}</small></div></div>
    <div className="detail-block"><div className="block-title"><span>Decision audit</span><span className={`vector-badge ${trial.test_vector}`}>{trial.test_vector}</span></div><dl className="detail-grid"><div><dt>Correct answer</dt><dd>{trial.correct_answer}</dd></div><div><dt>After prediction</dt><dd>{trial.after_prediction}</dd></div><div><dt>Correct evidence Δ</dt><dd>{signed(trial.correct_evidence_shift)}</dd></div><div><dt>Marker</dt><dd>{trial.marker_status}</dd></div><div><dt>Block</dt><dd>20 / 39</dd></div><div><dt>Test token</dt><dd>{testSite.position} · {testSite.token_id}</dd></div><div><dt>α</dt><dd>{alpha.toFixed(4)}</dd></div><div><dt>Seed</dt><dd>{episode.vector_seed}</dd></div></dl></div>
    <div className="detail-block"><div className="block-title"><span>Vector fingerprint</span><span className="mono">FP32 · 5,120d</span></div><VectorStrip coordinates={vector.first_12_coordinates} /><dl className="detail-grid compact-grid"><div><dt>L₂ norm</dt><dd>{vector.norm.toFixed(7)}</dd></div><div><dt>v₁ · v₂</dt><dd>{episode.vector_pair.dot_product.toExponential(2)}</dd></div></dl><code className="hash">sha256 {vector.sha256}</code></div>
  </aside>;
}

function BinaryPrompt({ sentence }: { sentence: string }) {
  return <div className="binary-prompt-grid"><div className="binary-prompt-column"><span className="prompt-stage-label">Before · matched prior</span><div className="prompt-turn system-turn"><span>System</span><p>During this conversation, an arbitrary hidden marker might be injected into your internal activations while you read a sentence. Its name is arbitrary and is defined only by the examples. Answer the final question with only YES or NO.</p></div><div className="prompt-turn"><span>Examples · prefilled</span><p>Example 1: {sentence} → <b>YES</b><br />Example 2: {sentence} → <b>NO</b></p></div><div className="prompt-turn withheld-turn"><span>Test · carrier withheld</span><p>Did the dax hidden marker occur in the test sentence?</p></div><div className="prompt-turn assistant-turn final-turn"><span>Assistant · scored</span><p>The answer is:<i className="cursor-block" /></p></div></div><div className="binary-prompt-column"><span className="prompt-stage-label">After · intervention exposed</span><div className="prompt-turn system-turn"><span>System</span><p>Same system prompt and demonstrations.</p></div><div className="prompt-turn"><span>Test</span><p>Read the following sentence carefully:<br /><b>{sentence}</b><br /><br />Did the dax hidden marker occur in the test sentence?</p></div><div className="prompt-turn assistant-turn final-turn"><span>Assistant · scored</span><p>The answer is:<i className="cursor-block" /></p></div></div></div>;
}

export function BinaryDaxDashboard({ results: raw }: { results: unknown }) {
  const results = raw as BinaryResults;
  const trials = useMemo(() => results.episodes.flatMap((episode) => episode.trials), [results]);
  const [selectedId, setSelectedId] = useState(trials[0].trial_id);
  const [query, setQuery] = useState("");
  const [outcome, setOutcome] = useState<OutcomeFilter>("all");
  const [vector, setVector] = useState<VectorFilter>("all");
  const selectedTrial = trials.find((trial) => trial.trial_id === selectedId) ?? trials[0];
  const selectedEpisode = results.episodes.find((episode) => episode.episode_id === selectedTrial.episode_id) ?? results.episodes[0];
  const filtered = trials.filter((trial) => {
    const q = query.trim().toLowerCase();
    return (!q || `${trial.trial_id} ${trial.sentence} ${trial.marker_status}`.toLowerCase().includes(q)) && (outcome === "all" || (outcome === "correct" ? trial.after_correct : !trial.after_correct)) && (vector === "all" || trial.test_vector === vector);
  });
  const beforeYes = mean(trials, "before_logit_YES");
  const beforeNo = mean(trials, "before_logit_NO");
  const afterYes = mean(trials, "after_logit_YES");
  const afterNo = mean(trials, "after_logit_NO");
  const completed = new Intl.DateTimeFormat("en", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" }).format(new Date(results.run.completed_at_utc));

  return <main className="binary-page">
    <ResearchHeader active="binary" status={`Complete · ${completed}`} />
    <section className="binary-hero">
      <div><span className="kicker">Experiment 2 · a simpler yes/no question</span><h1>Did the <em>dax</em><br />marker occur?</h1><p className="lede">Instead of choosing between two names, the model answers one binary question. We record its raw YES and NO scores before it sees the test sentence and again after an intervention is added to that sentence.</p><div className="hero-facts"><span><b>v₁</b> correct answer YES</span><span><b>v₂</b> correct answer NO</span><span><b>Trials</b> 64 paired episodes</span></div></div>
      <div className="binary-hero-visual"><div className="time-stage"><span>Before</span><b>test carrier withheld</b><div className="binary-choice no-choice">NO</div></div><i>→</i><div className="time-stage after-stage"><span>After</span><b>carrier + injected vector</b><div className="binary-choice no-choice">NO</div></div><div className="no-switch-stamp">0 / 128 switches</div></div>
    </section>

    <ExperimentPrimer
      question="Does the intervention create vector-specific YES/NO evidence?"
      idea="The model might contain useful evidence even if its final answer never changes. To check, we compare the change in YES minus NO after v₁ with the change after v₂. A real dax detector should move in opposite directions for the two vectors."
      fixed="The two demonstrations, question, answer prefix, sentence pool, intervention strength, layer, and token position."
      changed="The before prompt withholds the test carrier. The after prompt includes it and injects either v₁ (dax) or v₂ (not dax)."
      measured="The raw next-token logits for YES and NO in two separate forward passes, plus the change in YES − NO."
      answer="No. Both interventions shifted the score toward YES by about the same amount, and the model still answered NO on every trial."
    />

    <section className="result-section" id="binary-result"><div className="section-heading"><div><span className="section-index">01</span><span className="eyebrow">Result</span><h2>The scores moved, but not according to the vector</h2></div><p>The model answered NO before and after every test sentence. More importantly, v₁ and v₂ changed the YES−NO score by nearly identical amounts, so the movement cannot be interpreted as recognizing dax.</p></div>
      <div className="metric-grid"><BinaryMetric label="After accuracy" value={pct(results.aggregate.after.accuracy)} detail={`${results.aggregate.after.n_correct} / ${results.aggregate.after.n_trials} · CI ${pct(results.aggregate.after.ci_95.low)}–${pct(results.aggregate.after.ci_95.high)}`} tone="ochre" /><BinaryMetric label="Prediction switches" value={`${results.aggregate.prediction_switches}`} detail="before → after · 128 trials" /><BinaryMetric label="v₁ · dax · YES" value={pct(results.aggregate.by_test_vector.v1.accuracy)} detail="0 / 64 correct" tone="red" /><BinaryMetric label="v₂ · not-dax · NO" value={pct(results.aggregate.by_test_vector.v2.accuracy)} detail="64 / 64 correct" tone="green" /></div>
      <div className="binary-result-grid"><article className="paper-panel binary-logit-panel"><div className="panel-heading"><div><span className="eyebrow">Raw candidate logits · mean of 128 trials</span><h3>Before and after the test sentence</h3></div><span className="panel-note">the larger logit is chosen</span></div><BeforeAfterBars trials={trials} /><ReadingKey>Each row is one candidate answer. The middle columns show its mean raw score before and after the test sentence; the final column is the change.</ReadingKey><p className="figure-caption">YES rose by {signed(afterYes - beforeYes)}, while NO fell by {signed(afterNo - beforeNo)}. NO nevertheless retained an average lead of {Math.abs(afterYes - afterNo).toFixed(3)} logits.</p></article><article className="paper-panel vector-split-panel"><div className="panel-heading"><div><span className="eyebrow">The decisive comparison</span><h3>Change in YES − NO, split by vector</h3></div><span className="panel-note">after minus before</span></div><div className="split-shift"><div><span className="vector-orb v1">v₁</span><b>{signed(results.aggregate.by_test_vector.v1.mean_raw_yes_minus_no_shift)}</b><p>dax · should move toward YES</p></div><div className="split-divider"><span>{Math.abs(results.aggregate.by_test_vector.v1.mean_raw_yes_minus_no_shift - results.aggregate.by_test_vector.v2.mean_raw_yes_minus_no_shift).toFixed(4)}</span><small>difference</small></div><div><span className="vector-orb v2">v₂</span><b>{signed(results.aggregate.by_test_vector.v2.mean_raw_yes_minus_no_shift)}</b><p>not dax · should move toward NO</p></div></div><div className="non-discrimination-note">The two changes differ by only 0.006. They moved in the same direction instead of separating.</div></article><article className="paper-panel binary-shift-panel"><div className="panel-heading"><div><span className="eyebrow">All paired episodes</span><h3>Trial-level change in YES − NO</h3></div><div className="dot-legend"><span><i className="v1-dot" /> v₁</span><span><i className="v2-dot" /> v₂</span></div></div><ShiftPlot episodes={results.episodes} /><ReadingKey>Each x-position is one sentence. Positive values mean the test sentence moved the model toward YES. A detector should place green and red points on opposite sides of zero.</ReadingKey><p className="figure-caption">The traces overlap, which is the pattern expected from a generic prompt effect rather than vector recognition.</p></article></div>
    </section>

    <section className="audit-section" id="binary-audit"><div className="section-heading compact-heading"><div><span className="section-index">02</span><span className="eyebrow">Trial audit</span><h2>Every prior, posterior, and logit shift</h2></div><p>Select a cell or table row to compare the exact YES/NO logits before and after exposure, inspect the injected direction, and trace the evidence change.</p></div>
      <article className="matrix-panel paper-panel"><div className="panel-heading matrix-heading"><div><span className="eyebrow">64 episodes · 128 post-exposure decisions</span><h3>Binary outcome matrix</h3></div><div className="matrix-legend"><span><i className="correct-key" /> correct</span><span><i className="error-key" /> incorrect</span><span><i className="selected-key" /> selected</span></div></div><BinaryMatrix episodes={results.episodes} selected={selectedId} onSelect={setSelectedId} /></article>
      <div className="audit-workspace"><div className="trial-browser"><div className="filter-bar binary-filter"><label className="search-field"><span>Search trials</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="ID, sentence, marker…" /></label><label><span>Outcome</span><select value={outcome} onChange={(event) => setOutcome(event.target.value as OutcomeFilter)}><option value="all">All outcomes</option><option value="correct">Correct only</option><option value="incorrect">Errors only</option></select></label><label><span>Test vector</span><select value={vector} onChange={(event) => setVector(event.target.value as VectorFilter)}><option value="all">v₁ + v₂</option><option value="v1">v₁ · dax</option><option value="v2">v₂ · not-dax</option></select></label><span className="result-count">{filtered.length} shown</span></div><div className="table-shell"><table className="binary-table"><thead><tr><th>Trial</th><th>Carrier</th><th>Vector</th><th>Correct</th><th>Before YES</th><th>Before NO</th><th>After YES</th><th>After NO</th><th>Δ YES−NO</th><th>After</th><th>Outcome</th></tr></thead><tbody>{filtered.map((trial) => <tr key={trial.trial_id} className={selectedId === trial.trial_id ? "selected-row" : ""} onClick={() => setSelectedId(trial.trial_id)}><td><button className="trial-link">{trial.trial_id}</button></td><td className="sentence-cell">{trial.sentence}</td><td><span className={`vector-badge ${trial.test_vector}`}>{trial.test_vector}</span></td><td>{trial.correct_answer}</td><td className="mono">{trial.before_logit_YES.toFixed(3)}</td><td className="mono">{trial.before_logit_NO.toFixed(3)}</td><td className="mono">{trial.after_logit_YES.toFixed(3)}</td><td className="mono">{trial.after_logit_NO.toFixed(3)}</td><td className={`mono ${trial.delta_raw_yes_minus_no >= 0 ? "positive-margin" : "negative-margin"}`}>{signed(trial.delta_raw_yes_minus_no)}</td><td><b className="binary-answer-no">{trial.after_prediction}</b></td><td><span className={`outcome-dot ${trial.after_correct ? "is-correct" : "is-error"}`}><i />{trial.after_correct ? "Correct" : "Error"}</span></td></tr>)}{filtered.length === 0 && <tr><td colSpan={11} className="empty-state">No trials match these filters.</td></tr>}</tbody></table></div></div><BinaryDetail trial={selectedTrial} episode={selectedEpisode} alpha={results.design.alpha} /></div>
    </section>

    <section className="method-section"><div className="section-heading compact-heading"><div><span className="section-index">03</span><span className="eyebrow">Probe definition</span><h2>What “before” and “after” mean</h2></div><p>The before value is a matched pre-exposure response prior, not a hidden-state readout from the middle of the after pass. Each stage is a separate complete prompt with the same demonstrations and answer prefix.</p></div><div className="binary-method-grid"><article className="paper-panel prompt-panel"><div className="panel-heading"><div><span className="eyebrow">Episode 01 · paired contexts</span><h3>Probe construction</h3></div><span className="panel-note">official chat template</span></div><BinaryPrompt sentence={results.episodes[0].sentence} /></article><article className="paper-panel specification-panel"><div className="panel-heading"><div><span className="eyebrow">Recorded configuration</span><h3>Run specification</h3></div><span className="status-seal">verified</span></div><dl className="spec-list"><div><dt>Checkpoint</dt><dd>{results.run.model_id}</dd></div><div><dt>Precision</dt><dd>{results.run.model_dtype.toUpperCase()} · unquantized</dd></div><div><dt>GPU</dt><dd>{results.run.gpu}</dd></div><div><dt>Injection</dt><dd>block 20 · resid_pre</dd></div><div><dt>Calibration R<sub>ℓ</sub></dt><dd>{results.calibration.median_residual_norm.toFixed(4)}</dd></div><div><dt>Strength</dt><dd>α {results.design.alpha.toFixed(4)} · ρ {results.design.rho}</dd></div><div><dt>YES token</dt><dd>{results.tokenization.YES.id} · {JSON.stringify(results.tokenization.YES.decoded_piece)}</dd></div><div><dt>NO token</dt><dd>{results.tokenization.NO.id} · {JSON.stringify(results.tokenization.NO.decoded_piece)}</dd></div><div><dt>Single-token check</dt><dd>{results.tokenization.verified_single_token ? "Passed" : "Failed"}</dd></div><div><dt>Seed</dt><dd>{results.run.seed}</dd></div><div><dt>Wall time</dt><dd>{results.run.duration_seconds.toFixed(2)} s</dd></div></dl><div className="artifact-links"><a href="/data/binary-dax-results.json" download>Download binary audit JSON <span>335 KB</span></a><a href="/data/binary-dax-vectors-fp32.npz" download>Download FP32 vectors <span>2.3 MB</span></a></div></article></div><div className="interpretation-note"><span>Interpretation boundary</span><p>The +0.575 mean shift toward YES is not vector-specific: v₁ shifted +0.572 and v₂ shifted +0.578. Because every posterior still selected NO and the vector-conditioned shifts are indistinguishable at this descriptive level, this run does not show binary recognition of the dax intervention.</p></div></section>
    <footer><div><span className="wordmark-mark">H</span><p><b>Hidden Marker Lab</b><br />Experiment 02 · binary dax audit</p></div><p>Qwen3-14B · 128 trials · seed {results.run.seed}</p></footer>
  </main>;
}
