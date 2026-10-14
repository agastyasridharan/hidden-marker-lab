"use client";

import { useEffect, useMemo, useState } from "react";
import { ResearchHeader } from "./ResearchHeader";

type TaskName = "kth_word" | "cot_prediction" | "paraphrase" | "headsup";
type Variant = "base" | "trained_20260820";

type TaskSummary = {
  primary_score: number;
  primary_score_pct: number;
  primary_metric: string;
  primary_ci95: [number, number];
  [key: string]: unknown;
};

type Comparison = {
  pair: string;
  target: Variant;
  observer: Variant;
  is_self: boolean;
  average_primary_score: number;
  average_primary_score_pct: number;
  average_primary_ci95: [number, number];
  tasks: Record<TaskName, TaskSummary>;
};

type Results = {
  metadata: {
    completed_at_utc: string;
    model_id: string;
    model_revision: string;
    model_dtype: string;
    device: string;
    gpu_count_used: number;
    chat_template: string;
    sampling: string;
    adapter: { sha256: string; config: { trainable_parameters: number; lora_rank: number; lora_alpha: number; lora_start_layer: number; seed: number } };
    official_repository: { url: string; commit: string };
    paper: { url: string; version: string };
    known_release_discrepancies: string[];
  };
  coverage: Record<TaskName, { released_n: number; evaluated_items: number }>;
  comparisons: Comparison[];
  contrasts: Record<"trained_self_minus_base_self" | "base_self_minus_trained_observer" | "trained_self_minus_base_observer", { mean_difference: number; ci95: [number, number] }>;
  downloads: { full_audit: string; summary: string; task_audits: Record<TaskName, string> };
};

type RawTaskAudit = {
  comparisons: Record<string, { summary: TaskSummary; trials: Array<Record<string, unknown>> }>;
};

type DisplayTrial = {
  key: string;
  id: string;
  source: string;
  actual: string;
  predicted: string;
  measure: string;
  correct: boolean | null;
  parseValid: boolean;
  details: Array<{ label: string; value: string }>;
};

const taskInfo: Record<TaskName, { short: string; title: string; paperN: number; question: string; metric: string }> = {
  kth_word: {
    short: "K-th word",
    title: "Predict the beginning of a future answer",
    paperN: 200,
    question: "Before answering a question normally, can the observer predict the first word the target model will produce?",
    metric: "Exact match for the first word. Predictions for words 1 through 3 are retained in the audit.",
  },
  cot_prediction: {
    short: "CoT prediction",
    title: "Forecast a decision after extended reasoning",
    paperN: 196,
    question: "Can the observer predict the target model’s final A/B probabilities before the target writes its detailed reasoning?",
    metric: "One minus the mean absolute error in P(A). Higher is better. Parse failures receive the repository scorer’s default of 0.5 and are also counted explicitly.",
  },
  paraphrase: {
    short: "Paraphrase",
    title: "Infer which paraphrase produced an answer",
    paperN: 200,
    question: "After seeing a target response and three closely related prompts, can the observer identify the prompt that produced it?",
    metric: "Three-way prompt-choice accuracy over unique target outputs.",
  },
  headsup: {
    short: "Heads-Up",
    title: "Write clues that the target can decode",
    paperN: 200,
    question: "Can the observer describe a secret word in ten clues that the target model can use to recover the word?",
    metric: "The repository counts success when the target word occurs anywhere in the guess. Strict exact-match accuracy is reported separately.",
  },
};

const pct = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const pp = (value: number) => `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)} pp`;
const number = (value: unknown, digits = 3) => typeof value === "number" ? value.toFixed(digits) : "not available";
const text = (value: unknown) => value === null || value === undefined ? "not available" : String(value);

function scoreNote(task: TaskName, summary: TaskSummary) {
  if (task === "kth_word") return `${text(summary.k1_correct)} of ${text(summary.k1_n)} first-word predictions; all-k accuracy ${pct(Number(summary.all_k_accuracy))}`;
  if (task === "cot_prediction") return `MAE ${number(summary.official_mae)}; direct baseline ${pct(Number(summary.official_direct_baseline_score))}; forecast minus baseline ${pp(Number(summary.official_improvement_over_direct))}; ${text(summary.actual_parse_valid_n)} of ${text(summary.n)} target probabilities parsed`;
  if (task === "paraphrase") return `${text(summary.correct)} of ${text(summary.n)} output-to-prompt judgments; ${text(summary.parse_valid_n)} parsed`;
  return `${text(summary.official_successes)} of ${text(summary.n)} substring matches; strict exact match ${pct(Number(summary.strict_exact_score))}`;
}

function ComparisonPlot({ base, trained }: { base: Comparison; trained: Comparison }) {
  return <figure className="ib-score-figure">
    <div className="ib-score-axis"><span>0</span><span>25</span><span>50</span><span>75</span><span>100%</span></div>
    {(Object.keys(taskInfo) as TaskName[]).map((task) => {
      const baseValue = base.tasks[task].primary_score;
      const trainedValue = trained.tasks[task].primary_score;
      return <div className="ib-score-row" key={task}>
        <div><strong>{taskInfo[task].short}</strong><small>{pp(trainedValue - baseValue)} trained minus base</small></div>
        <div className="ib-score-track">
          <i className="ib-grid-25" /><i className="ib-grid-50" /><i className="ib-grid-75" />
          <span className="ib-score-link" style={{ left: `${Math.min(baseValue, trainedValue) * 100}%`, width: `${Math.abs(trainedValue - baseValue) * 100}%` }} />
          <b className="ib-dot ib-dot-base" style={{ left: `${baseValue * 100}%` }} aria-label={`Base ${pct(baseValue)}`} />
          <b className="ib-dot ib-dot-trained" style={{ left: `${trainedValue * 100}%` }} aria-label={`Trained ${pct(trainedValue)}`} />
        </div>
        <output><span>{pct(baseValue)}</span><b>{pct(trainedValue)}</b></output>
      </div>;
    })}
    <figcaption><span><i className="ib-legend-base" />Base model used as both target and observer</span><span><i className="ib-legend-trained" />Hidden-marker LoRA used as both target and observer</span></figcaption>
  </figure>;
}

function normaliseTrials(task: TaskName, rows: Array<Record<string, unknown>>): DisplayTrial[] {
  if (task === "kth_word") return rows.flatMap((row) => {
    const metrics = row.metrics as Array<Record<string, unknown>>;
    return metrics.map((metric) => ({
      key: `${row.id}-k${metric.k}`,
      id: `${row.id} · k=${metric.k}`,
      source: text(row.question),
      actual: text(metric.actual_word),
      predicted: text(metric.predicted),
      measure: metric.correct ? "exact match" : "different word",
      correct: Boolean(metric.correct),
      parseValid: Boolean(metric.prediction_parse_valid),
      details: [
        { label: "Question", value: text(row.question) },
        { label: "Target model’s complete answer", value: text(row.target_response) },
        { label: "Observer’s raw prediction", value: text(metric.prediction_raw) },
        { label: "Compared words", value: `predicted “${text(metric.predicted)}”; actual “${text(metric.actual_word)}”` },
      ],
    }));
  });
  if (task === "cot_prediction") return rows.map((row) => {
    const direct = row.direct as Record<string, unknown>;
    const predicted = row.predicted_cot as Record<string, unknown>;
    const actual = row.actual_cot as Record<string, unknown>;
    return {
      key: text(row.id), id: text(row.id), source: text(row.dilemma),
      actual: actual.valid ? `P(A) ${number(actual.p_a)}` : "parse failed",
      predicted: predicted.valid ? `P(A) ${number(predicted.p_a)}` : "parse failed",
      measure: `absolute error ${number(row.official_abs_error)}`,
      correct: null,
      parseValid: Boolean(predicted.valid) && Boolean(actual.valid),
      details: [
        { label: "Ethical dilemma", value: text(row.dilemma) },
        { label: "Direct baseline, before reasoning", value: text(direct.raw) },
        { label: "Observer’s forecast", value: text(predicted.raw) },
        { label: "Target’s reasoning and final probabilities", value: text(actual.raw) },
        { label: "Scoring", value: `forecast absolute error ${number(row.official_abs_error)}; direct-baseline absolute error ${number(row.official_direct_abs_error)}` },
      ],
    };
  });
  if (task === "paraphrase") return rows.map((row) => ({
    key: `${row.id}-${row.correct_prompt_index}`,
    id: `${row.id} · source ${Number(row.correct_prompt_index) + 1}`,
    source: text(row.generated_text),
    actual: `prompt ${Number(row.correct_prompt_index) + 1}`,
    predicted: row.predicted_prompt_index === null ? "parse failed" : `prompt ${Number(row.predicted_prompt_index) + 1}`,
    measure: row.correct ? "correct source" : "wrong source",
    correct: Boolean(row.correct), parseValid: Boolean(row.parse_valid),
    details: [
      { label: "Original dilemma", value: text(row.original_prompt) },
      ...((row.paraphrases as string[]).map((prompt, index) => ({ label: `Candidate prompt ${index + 1}`, value: prompt }))),
      { label: "Target model’s generated text", value: text(row.generated_text) },
      { label: "Observer’s raw judgment", value: text(row.judge_raw) },
    ],
  }));
  return rows.map((row) => ({
    key: text(row.id), id: text(row.id), source: text(row.clues),
    actual: text(row.target_word), predicted: text(row.guess),
    measure: row.official_substring_success ? "target found" : "target absent",
    correct: Boolean(row.official_substring_success),
    parseValid: Boolean(row.guess_json_valid),
    details: [
      { label: "Secret word", value: text(row.target_word) },
      { label: "Observer’s raw clue output", value: text(row.clues_raw) },
      { label: "Clues passed to the target", value: text(row.clues) },
      { label: "Target’s raw guess", value: text(row.guess_raw) },
      { label: "Scoring", value: `repository substring match: ${row.official_substring_success ? "yes" : "no"}; strict exact match: ${row.strict_exact_success ? "yes" : "no"}` },
    ],
  }));
}

function TrialLedger({ results }: { results: Results }) {
  const [task, setTask] = useState<TaskName>("kth_word");
  const [target, setTarget] = useState<Variant>("base");
  const [observer, setObserver] = useState<Variant>("base");
  const [artifact, setArtifact] = useState<RawTaskAudit | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("all");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<DisplayTrial | null>(null);
  const pageSize = 75;

  useEffect(() => {
    let current = true;
    fetch(results.downloads.task_audits[task]).then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    }).then((data: RawTaskAudit) => { if (current) setArtifact(data); })
      .catch((reason: Error) => { if (current) setError(reason.message); })
      .finally(() => { if (current) setLoading(false); });
    return () => { current = false; };
  }, [results.downloads.task_audits, task]);

  const pair = `target=${target}|observer=${observer}`;
  const trials = useMemo(() => artifact?.comparisons[pair] ? normaliseTrials(task, artifact.comparisons[pair].trials) : [], [artifact, pair, task]);
  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return trials.filter((trial) => {
      if (status === "correct" && trial.correct !== true) return false;
      if (status === "incorrect" && trial.correct !== false) return false;
      if (status === "parsed" && !trial.parseValid) return false;
      if (status === "parse-failure" && trial.parseValid) return false;
      return !needle || `${trial.id} ${trial.source} ${trial.actual} ${trial.predicted}`.toLowerCase().includes(needle);
    });
  }, [query, status, trials]);
  const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const visible = filtered.slice(page * pageSize, (page + 1) * pageSize);

  return <div className="ib-audit-workspace">
    <div className="ib-ledger">
      <div className="ib-filter-bar">
        <label>Task<select value={task} onChange={(event) => { setTask(event.target.value as TaskName); setLoading(true); setError(""); setSelected(null); setPage(0); }}>{(Object.keys(taskInfo) as TaskName[]).map((name) => <option value={name} key={name}>{taskInfo[name].short}</option>)}</select></label>
        <label>Target model<select value={target} onChange={(event) => { setTarget(event.target.value as Variant); setSelected(null); setPage(0); }}>{VARIANT_OPTIONS}</select></label>
        <label>Observer model<select value={observer} onChange={(event) => { setObserver(event.target.value as Variant); setSelected(null); setPage(0); }}>{VARIANT_OPTIONS}</select></label>
        <label>Status<select value={status} onChange={(event) => { setStatus(event.target.value); setPage(0); }}><option value="all">All trials</option><option value="correct">Correct</option><option value="incorrect">Incorrect</option><option value="parsed">Parsed</option><option value="parse-failure">Parse failure</option></select></label>
        <label>Search<input value={query} onChange={(event) => { setQuery(event.target.value); setPage(0); }} placeholder="ID, prompt, or output" /></label>
      </div>
      {loading ? <p className="empty-state">Loading the selected audit file…</p> : error ? <p className="empty-state">The audit file could not be loaded: {error}</p> : <>
        <div className="ib-ledger-count"><span>{filtered.length.toLocaleString()} visible trials</span><span>Page {page + 1} of {pages}</span></div>
        <div className="ib-table-shell"><table className="ib-trial-table"><thead><tr><th>Trial</th><th>Prompt, response, or clues</th><th>Target result</th><th>Observer result</th><th>Measure</th></tr></thead><tbody>{visible.map((trial) => <tr key={trial.key} className={selected?.key === trial.key ? "selected-row" : ""} onClick={() => setSelected(trial)}><td><button className="trial-link">{trial.id}</button></td><td className="sentence-cell">{trial.source}</td><td>{trial.actual}</td><td>{trial.predicted}</td><td>{trial.correct === null ? <span className={trial.parseValid ? "ib-parsed" : "ib-failed"}>{trial.measure}</span> : <span className={`outcome-dot ${trial.correct ? "is-correct" : "is-error"}`}><i />{trial.measure}</span>}</td></tr>)}</tbody></table></div>
        <div className="ib-pagination"><button disabled={page === 0} onClick={() => setPage((value) => Math.max(0, value - 1))}>Previous</button><button disabled={page + 1 >= pages} onClick={() => setPage((value) => Math.min(pages - 1, value + 1))}>Next</button></div>
      </>}
    </div>
    <aside className="ib-trial-detail">
      {selected ? <><div className="detail-heading"><div><span className="eyebrow">Selected trial</span><h3>{selected.id}</h3></div>{selected.correct !== null && <span className={`outcome-pill ${selected.correct ? "is-correct" : "is-error"}`}>{selected.correct ? "correct" : "incorrect"}</span>}</div>{selected.details.map((detail) => <div className="ib-detail-block" key={detail.label}><span className="detail-label">{detail.label}</span><p>{detail.value}</p></div>)}</> : <div className="ib-detail-empty"><span className="eyebrow">Trial detail</span><p>Select a row to read the complete prompt and both model outputs. The table is abbreviated only to keep the ledger scannable.</p></div>}
    </aside>
  </div>;
}

const VARIANT_OPTIONS = <><option value="base">Base Qwen3-14B</option><option value="trained_20260820">Hidden-marker LoRA</option></>;

export function IntrospectBenchDashboard({ results: rawResults }: { results: unknown }) {
  const results = rawResults as Results;
  const base = results.comparisons.find((row) => row.target === "base" && row.observer === "base")!;
  const trained = results.comparisons.find((row) => row.target === "trained_20260820" && row.observer === "trained_20260820")!;
  const crossBaseTarget = results.comparisons.find((row) => row.target === "base" && row.observer === "trained_20260820")!;
  const crossTrainedTarget = results.comparisons.find((row) => row.target === "trained_20260820" && row.observer === "base")!;
  const delta = trained.average_primary_score - base.average_primary_score;
  const trainingContrast = results.contrasts.trained_self_minus_base_self;
  const baseSelfContrast = results.contrasts.base_self_minus_trained_observer;
  const trainedSelfContrast = results.contrasts.trained_self_minus_base_observer;
  const trainedSelfAdvantage = trained.average_primary_score - crossTrainedTarget.average_primary_score;
  const baseSelfAdvantage = base.average_primary_score - crossBaseTarget.average_primary_score;
  const changePhrase = delta > 0.01 ? "increased" : delta < -0.01 ? "decreased" : "left nearly unchanged";

  return <main className="ib-page">
    <ResearchHeader active="benchmark" status="IntrospectBench run · complete" />

    <section className="ib-hero">
      <div>
        <span className="kicker">Experiment 5 · broader policy introspection</span>
        <h1>Does hidden-marker training transfer to <em>IntrospectBench?</em></h1>
        <p className="lede">IntrospectBench asks a model to predict or reconstruct its own behavior in four ordinary language tasks. We ran the unmodified Qwen3-14B and the primary hidden-marker LoRA adapter as both the model being studied and the model making the introspective judgment.</p>
        <p className="ib-answer"><b>Result:</b> the adapter {changePhrase} the four-task compatibility average, changing it from {pct(base.average_primary_score)} to {pct(trained.average_primary_score)} ({pp(delta)}). The paired bootstrap interval for this difference is {pp(trainingContrast.ci95[0])} to {pp(trainingContrast.ci95[1])}. This comparison does not show that hidden-marker training produced a general introspective faculty, because the task-level changes and cross-model comparisons must be examined separately.</p>
      </div>
      <aside className="ib-hero-result">
        <span className="eyebrow">Four-task self score</span>
        <div><span>Base</span><strong>{pct(base.average_primary_score)}</strong><small>95% bootstrap interval {pct(base.average_primary_ci95[0])} to {pct(base.average_primary_ci95[1])}</small></div>
        <div><span>LoRA</span><strong>{pct(trained.average_primary_score)}</strong><small>95% bootstrap interval {pct(trained.average_primary_ci95[0])} to {pct(trained.average_primary_ci95[1])}</small></div>
        <p>These are unweighted means of four differently defined benchmark scores. They are summaries, not interchangeable accuracy rates.</p>
      </aside>
    </section>

    <section className="overview-section pale-section">
      <div className="prose-heading"><span className="section-index">A</span><div><span className="eyebrow">What was evaluated</span><h2>One model produces behavior; another model tries to predict it</h2></div><p>“Target” means the model whose answer, decision, or guess is being studied. “Observer” means the model asked to predict or reconstruct that target behavior. A self comparison uses the same variant in both roles. The two cross comparisons test whether apparent self-prediction is actually a general ability to guess another model.</p></div>
      <div className="ib-role-diagram">
        <article><span>1</span><div><b>Target behavior</b><p>Qwen answers a question, reasons through a dilemma, responds to a paraphrase, or guesses a secret word.</p></div></article><i>→</i>
        <article><span>2</span><div><b>Observer judgment</b><p>Base Qwen or the LoRA predicts a word or probability, identifies a source prompt, or constructs clues.</p></div></article><i>→</i>
        <article><span>3</span><div><b>Comparison</b><p>The benchmark checks the forecast against the target behavior using the released task-specific scoring rule.</p></div></article>
      </div>
      <div className="ib-adapter-note"><b>The LoRA was not trained on IntrospectBench.</b><p>It was trained only to match fresh residual-stream interventions to arbitrary in-context labels. The original Qwen weights remain frozen; rank-16 updates with alpha 32 affect attention and MLP projections in blocks 20 through 39. Testing IntrospectBench therefore measures out-of-distribution behavioral transfer, not benchmark fine-tuning.</p></div>
    </section>

    <section className="overview-section">
      <div className="prose-heading"><span className="section-index">B</span><div><span className="eyebrow">Primary result</span><h2>Compare the models task by task before interpreting the average</h2></div><p>Each row shows the repository-compatible score with the model serving as both target and observer. The horizontal position is the score from 0% to 100%; the printed values are authoritative.</p></div>
      <ComparisonPlot base={base} trained={trained} />
      <div className="ib-cot-baseline"><b>Read the CoT score against its direct baseline.</b><p>For the base self comparison, the introspective forecast score is {pct(base.tasks.cot_prediction.primary_score)} while direct, non-introspective probabilities score {pct(Number(base.tasks.cot_prediction.official_direct_baseline_score))}, a change of {pp(Number(base.tasks.cot_prediction.official_improvement_over_direct))}. For the LoRA self comparison, the corresponding change is {pp(Number(trained.tasks.cot_prediction.official_improvement_over_direct))}. A high raw score is not evidence of useful introspection when the direct baseline is higher.</p></div>
      <div className="ib-task-cards">{(Object.keys(taskInfo) as TaskName[]).map((task) => <article key={task}><span className="eyebrow">{taskInfo[task].short}</span><h3>{taskInfo[task].title}</h3><p>{taskInfo[task].question}</p><dl><div><dt>Base self score</dt><dd>{pct(base.tasks[task].primary_score)}</dd></div><div><dt>LoRA self score</dt><dd>{pct(trained.tasks[task].primary_score)}</dd></div><div><dt>Observed counts</dt><dd>{scoreNote(task, trained.tasks[task])}</dd></div></dl><small>{taskInfo[task].metric}</small></article>)}</div>
    </section>

    <section className="overview-section pale-section">
      <div className="prose-heading"><span className="section-index">C</span><div><span className="eyebrow">Self versus cross-model prediction</span><h2>A self score is not sufficient evidence of privileged self-access</h2></div><p>If a model is especially good at predicting itself, its diagonal cell should exceed the other observer’s score on the same target. Here the base self advantage is {pp(baseSelfAdvantage)} (95% paired interval {pp(baseSelfContrast.ci95[0])} to {pp(baseSelfContrast.ci95[1])}), and the LoRA self advantage is {pp(trainedSelfAdvantage)} ({pp(trainedSelfContrast.ci95[0])} to {pp(trainedSelfContrast.ci95[1])}).</p></div>
      <div className="ib-cross-matrix">
        <div className="ib-corner">Target ↓<br />Observer →</div><div className="ib-col-head">Base observer</div><div className="ib-col-head">LoRA observer</div>
        <div className="ib-row-head">Base target</div><div className="ib-matrix-cell ib-self-cell"><span>self</span><strong>{pct(base.average_primary_score)}</strong><small>{base.tasks.kth_word.primary_score_pct.toFixed(1)} · {base.tasks.cot_prediction.primary_score_pct.toFixed(1)} · {base.tasks.paraphrase.primary_score_pct.toFixed(1)} · {base.tasks.headsup.primary_score_pct.toFixed(1)}</small></div><div className="ib-matrix-cell"><span>cross</span><strong>{pct(crossBaseTarget.average_primary_score)}</strong><small>{crossBaseTarget.tasks.kth_word.primary_score_pct.toFixed(1)} · {crossBaseTarget.tasks.cot_prediction.primary_score_pct.toFixed(1)} · {crossBaseTarget.tasks.paraphrase.primary_score_pct.toFixed(1)} · {crossBaseTarget.tasks.headsup.primary_score_pct.toFixed(1)}</small></div>
        <div className="ib-row-head">LoRA target</div><div className="ib-matrix-cell"><span>cross</span><strong>{pct(crossTrainedTarget.average_primary_score)}</strong><small>{crossTrainedTarget.tasks.kth_word.primary_score_pct.toFixed(1)} · {crossTrainedTarget.tasks.cot_prediction.primary_score_pct.toFixed(1)} · {crossTrainedTarget.tasks.paraphrase.primary_score_pct.toFixed(1)} · {crossTrainedTarget.tasks.headsup.primary_score_pct.toFixed(1)}</small></div><div className="ib-matrix-cell ib-self-cell"><span>self</span><strong>{pct(trained.average_primary_score)}</strong><small>{trained.tasks.kth_word.primary_score_pct.toFixed(1)} · {trained.tasks.cot_prediction.primary_score_pct.toFixed(1)} · {trained.tasks.paraphrase.primary_score_pct.toFixed(1)} · {trained.tasks.headsup.primary_score_pct.toFixed(1)}</small></div>
      </div>
      <p className="ib-matrix-caption">Each cell is the unweighted mean of K-th word, CoT prediction, paraphrase recognition, and Heads-Up. The four small numbers list those task scores in that order. Positive diagonal advantages would be consistent with model-specific access, but they would not by themselves establish introspection.</p>
    </section>

    <section className="overview-section">
      <div className="prose-heading"><span className="section-index">D</span><div><span className="eyebrow">Released data coverage</span><h2>The public repository does not contain the complete paper-sized benchmark</h2></div><p>We did not synthesize replacements for missing cases. The run uses the first 200 released K-th-word questions, the first 196 dilemmas, and every released paraphrase and Heads-Up record. This preserves provenance but leaves the latter two tasks below the paper’s stated size.</p></div>
      <div className="ib-coverage-table"><div className="ib-coverage-head"><span>Task</span><span>Paper states</span><span>Repository releases</span><span>Evaluated here</span></div>{(Object.keys(taskInfo) as TaskName[]).map((task) => <div key={task}><strong>{taskInfo[task].short}</strong><span>{taskInfo[task].paperN}</span><span className={results.coverage[task].released_n < taskInfo[task].paperN ? "ib-shortfall" : ""}>{results.coverage[task].released_n}</span><span>{results.coverage[task].evaluated_items}</span></div>)}</div>
      <div className="ib-release-note"><span className="eyebrow">Why this is a qualified replication</span><ol>{results.metadata.known_release_discrepancies.map((note) => <li key={note}>{note}</li>)}</ol></div>
    </section>

    <section className="overview-section pale-section ib-audit-section">
      <div className="prose-heading"><span className="section-index">E</span><div><span className="eyebrow">Complete trial audit</span><h2>Read every prompt, target output, observer output, and score</h2></div><p>Select a task and target–observer pair, then filter or search the complete ledger. K-th-word predictions are shown separately for k=1, 2, and 3 even though the compatibility score uses k=1.</p></div>
      <TrialLedger results={results} />
    </section>

    <section className="overview-section">
      <div className="prose-heading"><span className="section-index">F</span><div><span className="eyebrow">Protocol and interpretation</span><h2>What the run supports, and what it does not</h2></div><p>The benchmark was run locally on three H100s with unquantized BF16 weights, the official Qwen chat template with thinking disabled, and greedy decoding. Base and LoRA generations came from one loaded checkpoint with the adapter switched on or off.</p></div>
      <div className="ib-method-grid">
        <article><h3>Reproducibility</h3><dl><div><dt>Model</dt><dd>{results.metadata.model_id}</dd></div><div><dt>Revision</dt><dd>{results.metadata.model_revision.slice(0, 12)}…</dd></div><div><dt>Adapter SHA-256</dt><dd>{results.metadata.adapter.sha256.slice(0, 12)}…</dd></div><div><dt>Decoding</dt><dd>{results.metadata.sampling}</dd></div><div><dt>Precision</dt><dd>{results.metadata.model_dtype}, no quantization</dd></div><div><dt>Completed</dt><dd>{results.metadata.completed_at_utc.replace("T", " ").slice(0, 19)} UTC</dd></div></dl></article>
        <article><h3>Interpretive limit</h3><p>These tasks concern prediction of model behavior, not direct reporting of a residual-stream perturbation. A difference between base and LoRA could arise from changed answer policies, formatting reliability, or broad language behavior. The cross-model cells help diagnose model-specific prediction, but this two-variant study cannot rule out all shared heuristics.</p><p>The ethical score is especially sensitive to output parsing and to the repository’s 0.5 default. The dashboard therefore reports parse counts, parsed-only errors, and KL diagnostics in the audit artifact.</p></article>
      </div>
      <div className="artifact-links ib-downloads"><a href={results.downloads.summary} download>Download result summary<span>JSON · methods and aggregate scores</span></a><a href={results.downloads.full_audit} download>Download full audit<span>JSON · every model output</span></a><a href={results.metadata.paper.url} target="_blank" rel="noreferrer">Read the paper<span>IntrospectBench · arXiv v1</span></a><a href={`${results.metadata.official_repository.url}/tree/${results.metadata.official_repository.commit}`} target="_blank" rel="noreferrer">Inspect pinned benchmark code<span>Official repository · exact commit</span></a></div>
    </section>

    <footer><div><span className="wordmark-mark">H</span><p><b>Hidden Marker Lab</b><br />IntrospectBench replication and complete audit</p></div><p>Qwen3-14B · base and hidden-marker LoRA</p></footer>
  </main>;
}
