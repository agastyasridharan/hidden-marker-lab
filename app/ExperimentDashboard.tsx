"use client";

import { useMemo, useState } from "react";
import { ExperimentPrimer, ReadingKey } from "./ExperimentPrimer";
import { ResearchHeader } from "./ResearchHeader";

type Trial = {
  trial_id: string;
  episode_id: number;
  sentence: string;
  test_vector: "v1" | "v2";
  correct_option: "A" | "B";
  correct_label: "dax" | "wug";
  predicted_option: "A" | "B";
  predicted_label: "dax" | "wug";
  correct: boolean;
  logit_A: number;
  logit_B: number;
  raw_A_minus_B: number;
  margin: number;
  answer_prefix_last_token_id: number;
  answer_prefix_last_token_piece: string;
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

type InjectionSite = {
  role: string;
  position: number;
  token_id: number;
  decoded_piece: string;
  vector: string;
  label: string;
};

type Episode = {
  episode_id: number;
  sentence: string;
  vector_seed: number;
  vector_pair: {
    v1: VectorSummary;
    v2: VectorSummary;
    dot_product: number;
    cosine_similarity: number;
  };
  injection_sites: InjectionSite[];
  prompt_token_count: number;
  trials: Trial[];
};

type ExperimentResults = {
  status: string;
  run: {
    id: string;
    started_at_utc: string;
    completed_at_utc: string;
    duration_seconds: number;
    seed: number;
    host: string;
    python_version: string;
    torch_version: string;
    transformers_version: string;
    cuda_version: string;
    gpu: string;
    model_id: string;
    model_revision: string;
    model_dtype: string;
    evaluation_mode: boolean;
    sampling: boolean;
  };
  design: {
    episodes_planned: number;
    trials_planned: number;
    rho: number;
    alpha: number;
    layer_index_zero_based: number;
    layer_ordinal_one_based: number;
    num_layers: number;
    site: string;
    target: string;
    hidden_size: number;
    vector_distribution: string;
    vector_storage_dtype: string;
    injection_dtype: string;
    complete_prompt_single_causal_forward: boolean;
    paired_trials_batched_together: boolean;
    chat_template: string;
  };
  tokenization: {
    format: string;
    A: { id: number; requested_text: string; decoded_piece: string };
    B: { id: number; requested_text: string; decoded_piece: string };
    verified_single_token: boolean;
    example_carrier_sites: {
      positions: number[];
      token_ids: number[];
      decoded_pieces: string[];
    };
  };
  calibration: {
    count: number;
    format: string;
    median_residual_norm: number;
    rho: number;
    alpha: number;
    min: number;
    q1: number;
    q3: number;
    max: number;
  };
  aggregate: {
    n_trials: number;
    n_correct: number;
    accuracy: number;
    chance_accuracy: number;
    ci_95: { method: string; level: number; low: number; high: number };
    mean_margin: number;
    median_margin: number;
    margin_std: number;
    margin_min: number;
    margin_max: number;
    margin_quartiles: number[];
    by_test_vector: Record<"v1" | "v2", {
      n_trials: number;
      n_correct: number;
      accuracy: number;
      ci_95_wilson: number[];
      mean_margin: number;
    }>;
  };
  episodes: Episode[];
};

type OutcomeFilter = "all" | "correct" | "incorrect";
type VectorFilter = "all" | "v1" | "v2";
type SortKey = "episode" | "margin-low" | "margin-high";

const pct = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const signed = (value: number, digits = 3) => `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
const compact = (value: number) => value.toExponential(2);

function MetricCard({
  eyebrow,
  value,
  detail,
  tone = "neutral",
}: {
  eyebrow: string;
  value: string;
  detail: string;
  tone?: "neutral" | "green" | "red" | "ochre";
}) {
  return (
    <div className={`metric-card metric-${tone}`}>
      <span className="eyebrow">{eyebrow}</span>
      <strong>{value}</strong>
      <span className="metric-detail">{detail}</span>
    </div>
  );
}

function AccuracyPlot({ results }: { results: ExperimentResults }) {
  const { accuracy, chance_accuracy: chance, ci_95: ci } = results.aggregate;
  return (
    <div className="accuracy-plot" aria-label={`Accuracy ${pct(accuracy)} with 95% confidence interval ${pct(ci.low)} to ${pct(ci.high)}`}>
      <div className="accuracy-axis">
        <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
      </div>
      <div className="accuracy-track">
        <div className="chance-line" style={{ left: `${chance * 100}%` }} />
        <div
          className="confidence-line"
          style={{ left: `${ci.low * 100}%`, width: `${(ci.high - ci.low) * 100}%` }}
        />
        <div className="accuracy-point" style={{ left: `${accuracy * 100}%` }}>
          <span>{pct(accuracy)}</span>
        </div>
      </div>
      <div className="accuracy-legend">
        <span><i className="legend-point" /> observed accuracy</span>
        <span><i className="legend-whisker" /> 95% Wilson CI</span>
        <span><i className="legend-chance" /> chance</span>
      </div>
    </div>
  );
}

function PairedMarginPlot({ episodes }: { episodes: Episode[] }) {
  const maxAbs = Math.max(...episodes.flatMap((episode) => episode.trials.map((trial) => Math.abs(trial.margin))), 1);
  const topFor = (margin: number) => `${((maxAbs - margin) / (2 * maxAbs)) * 100}%`;
  return (
    <div className="paired-plot-shell">
      <div className="paired-axis-label paired-axis-positive">+{maxAbs.toFixed(1)}</div>
      <div className="paired-axis-label paired-axis-zero">0</div>
      <div className="paired-axis-label paired-axis-negative">−{maxAbs.toFixed(1)}</div>
      <div className="paired-plot" aria-label="Signed correct-choice margins for all 64 paired episodes">
        <div className="paired-zero-line" />
        {episodes.map((episode) => (
          <div className="paired-episode" key={episode.episode_id} title={`Episode ${episode.episode_id}: ${episode.sentence}`}>
            <div
              className="pair-connector"
              style={{
                top: topFor(Math.max(...episode.trials.map((trial) => trial.margin))),
                height: `${(Math.abs(episode.trials[0].margin - episode.trials[1].margin) / (2 * maxAbs)) * 100}%`,
              }}
            />
            {episode.trials.map((trial) => (
              <span
                key={trial.trial_id}
                className={`margin-dot ${trial.test_vector}`}
                style={{ top: topFor(trial.margin) }}
              />
            ))}
            <span className="episode-tick">{episode.episode_id % 8 === 0 ? episode.episode_id : ""}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function VectorStrip({ coordinates }: { coordinates: number[] }) {
  const maxAbs = Math.max(...coordinates.map(Math.abs), 0.001);
  return (
    <div className="vector-strip" aria-label="First 12 vector coordinates">
      {coordinates.map((coordinate, index) => (
        <span className="coordinate" key={index}>
          <i
            className={coordinate >= 0 ? "positive" : "negative"}
            style={{ height: `${Math.max(8, (Math.abs(coordinate) / maxAbs) * 46)}%` }}
          />
        </span>
      ))}
    </div>
  );
}

function TrialDetail({ trial, episode }: { trial: Trial; episode: Episode }) {
  const vector = episode.vector_pair[trial.test_vector];
  const testSite = episode.injection_sites.find((site) => site.role === "test")!;
  const largestLogit = Math.max(trial.logit_A, trial.logit_B, 1);
  return (
    <aside className="trial-detail" aria-label={`Details for ${trial.trial_id}`}>
      <div className="detail-heading">
        <div>
          <span className="eyebrow">Selected trial</span>
          <h3>{trial.trial_id}</h3>
        </div>
        <span className={`outcome-pill ${trial.correct ? "is-correct" : "is-error"}`}>
          {trial.correct ? "Correct" : "Incorrect"}
        </span>
      </div>

      <div className="sentence-card">
        <span className="detail-label">Carrier · repeated 3×</span>
        <p>{trial.sentence.slice(0, -1)}<mark title="Injection token">.</mark></p>
        <span className="sentence-note">Highlighted token receives the residual intervention.</span>
      </div>

      <div className="sequence-diagram" aria-label="Injection sequence">
        <div className="sequence-step">
          <span>Example 1</span><b className="vector-badge v1">v₁</b><em>dax</em>
        </div>
        <div className="sequence-arrow">→</div>
        <div className="sequence-step">
          <span>Example 2</span><b className="vector-badge v2">v₂</b><em>wug</em>
        </div>
        <div className="sequence-arrow">→</div>
        <div className="sequence-step test-step">
          <span>Test</span><b className={`vector-badge ${trial.test_vector}`}>{trial.test_vector === "v1" ? "v₁" : "v₂"}</b><em>?</em>
        </div>
      </div>

      <div className="detail-block">
        <div className="block-title"><span>Next-token logits</span><span className="mono">margin {signed(trial.margin)}</span></div>
        <div className="logit-row">
          <span>A · dax</span>
          <div><i className="bar-a" style={{ width: `${(trial.logit_A / largestLogit) * 100}%` }} /></div>
          <b>{trial.logit_A.toFixed(2)}</b>
        </div>
        <div className="logit-row">
          <span>B · wug</span>
          <div><i className="bar-b" style={{ width: `${(trial.logit_B / largestLogit) * 100}%` }} /></div>
          <b>{trial.logit_B.toFixed(2)}</b>
        </div>
        <div className="choice-line">
          expected <b>{trial.correct_option} · {trial.correct_label}</b>
          <span>predicted <b>{trial.predicted_option} · {trial.predicted_label}</b></span>
        </div>
      </div>

      <div className="detail-block">
        <div className="block-title"><span>Injection site</span><span className="mono">resid_pre</span></div>
        <dl className="detail-grid">
          <div><dt>Block</dt><dd>20 / 39</dd></div>
          <div><dt>Token position</dt><dd>{testSite.position}</dd></div>
          <div><dt>Token ID</dt><dd>{testSite.token_id}</dd></div>
          <div><dt>Decoded piece</dt><dd>{JSON.stringify(testSite.decoded_piece)}</dd></div>
          <div><dt>α</dt><dd>12.3593</dd></div>
          <div><dt>ρ</dt><dd>0.1</dd></div>
        </dl>
      </div>

      <div className="detail-block vector-block">
        <div className="block-title"><span>Vector fingerprint</span><span className={`vector-badge ${trial.test_vector}`}>{trial.test_vector}</span></div>
        <VectorStrip coordinates={vector.first_12_coordinates} />
        <dl className="detail-grid compact-grid">
          <div><dt>Seed</dt><dd>{episode.vector_seed}</dd></div>
          <div><dt>L₂ norm</dt><dd>{vector.norm.toFixed(7)}</dd></div>
          <div><dt>v₁ · v₂</dt><dd>{compact(episode.vector_pair.dot_product)}</dd></div>
          <div><dt>Std.</dt><dd>{vector.std.toFixed(6)}</dd></div>
        </dl>
        <code className="hash">sha256 {vector.sha256}</code>
      </div>
    </aside>
  );
}

function TrialMatrix({
  episodes,
  selectedId,
  onSelect,
}: {
  episodes: Episode[];
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="trial-matrix" aria-label="All 128 trials grouped into 64 paired episodes">
      {episodes.map((episode) => (
        <div className="episode-cell" key={episode.episode_id}>
          <span className="episode-number">E{String(episode.episode_id).padStart(2, "0")}</span>
          {episode.trials.map((trial) => (
            <button
              className={`${trial.correct ? "correct-cell" : "error-cell"} ${selectedId === trial.trial_id ? "selected-cell" : ""}`}
              key={trial.trial_id}
              onClick={() => onSelect(trial.trial_id)}
              title={`${trial.trial_id} · ${trial.correct ? "correct" : "incorrect"} · margin ${signed(trial.margin)}`}
              aria-label={`Select ${trial.trial_id}, ${trial.test_vector}, ${trial.correct ? "correct" : "incorrect"}`}
            >
              <span>{trial.test_vector === "v1" ? "v₁" : "v₂"}</span>
              <b>{trial.predicted_option}</b>
            </button>
          ))}
        </div>
      ))}
    </div>
  );
}

function PromptPreview({ sentence }: { sentence: string }) {
  return (
    <div className="prompt-preview">
      <div className="prompt-turn system-turn">
        <span>System</span>
        <p>During this conversation, two arbitrary hidden markers may occur while you read a sentence. Their names are arbitrary and are defined only by the examples. Answer the final question with only A or B.</p>
      </div>
      <div className="prompt-turn">
        <span>User · Example 1</span>
        <p>Read the following sentence carefully:<br />{sentence}</p>
      </div>
      <div className="prompt-turn assistant-turn">
        <span>Assistant · prefilled</span>
        <p>The hidden marker in that sentence was <b>dax</b>.</p>
      </div>
      <div className="prompt-turn">
        <span>User · Example 2</span>
        <p>Read the following sentence carefully:<br />{sentence}</p>
      </div>
      <div className="prompt-turn assistant-turn">
        <span>Assistant · prefilled</span>
        <p>The hidden marker in that sentence was <b>wug</b>.</p>
      </div>
      <div className="prompt-turn">
        <span>User · Test</span>
        <p>Read the following sentence carefully:<br />{sentence}<br /><br />Which hidden marker occurred?<br />A) dax<br />B) wug</p>
      </div>
      <div className="prompt-turn assistant-turn final-turn">
        <span>Assistant · scored next token</span>
        <p>The answer is:<i className="cursor-block" /></p>
      </div>
    </div>
  );
}

export function ExperimentDashboard({ results: rawResults }: { results: unknown }) {
  const results = rawResults as ExperimentResults;
  const allTrials = useMemo(() => results.episodes.flatMap((episode) => episode.trials), [results]);
  const [selectedId, setSelectedId] = useState(allTrials[0].trial_id);
  const [query, setQuery] = useState("");
  const [outcome, setOutcome] = useState<OutcomeFilter>("all");
  const [vector, setVector] = useState<VectorFilter>("all");
  const [sort, setSort] = useState<SortKey>("episode");

  const selectedTrial = allTrials.find((trial) => trial.trial_id === selectedId) ?? allTrials[0];
  const selectedEpisode = results.episodes.find((episode) => episode.episode_id === selectedTrial.episode_id) ?? results.episodes[0];
  const filteredTrials = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    const filtered = allTrials.filter((trial) => {
      const matchesQuery = !normalizedQuery || `${trial.trial_id} ${trial.sentence} ${trial.predicted_label} ${trial.correct_label}`.toLowerCase().includes(normalizedQuery);
      const matchesOutcome = outcome === "all" || (outcome === "correct" ? trial.correct : !trial.correct);
      const matchesVector = vector === "all" || trial.test_vector === vector;
      return matchesQuery && matchesOutcome && matchesVector;
    });
    return [...filtered].sort((a, b) => {
      if (sort === "margin-low") return a.margin - b.margin;
      if (sort === "margin-high") return b.margin - a.margin;
      return a.episode_id - b.episode_id || a.test_vector.localeCompare(b.test_vector);
    });
  }, [allTrials, outcome, query, sort, vector]);

  const predictedA = allTrials.filter((trial) => trial.predicted_option === "A").length;
  const predictedB = allTrials.length - predictedA;
  const completedDate = new Intl.DateTimeFormat("en", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" }).format(new Date(results.run.completed_at_utc));

  return (
    <main>
      <ResearchHeader active="label" status={`Complete · ${completedDate}`} />

      <section className="hero" id="top">
        <div className="hero-copy">
          <span className="kicker">Experiment 1 · untrained Qwen3-14B</span>
          <h1>Can the base model distinguish two <em>hidden interventions?</em></h1>
          <p className="lede">We give the model two examples. In one, an internal vector called “dax” is added; in the other, a different vector called “wug” is added. We then repeat one vector and ask which name belongs to it.</p>
          <div className="hero-facts">
            <span><b>Model</b> Qwen3-14B</span>
            <span><b>Site</b> block 20 · resid_pre</span>
            <span><b>Strength</b> ρ = 0.1</span>
          </div>
        </div>
        <div className="hero-diagram" aria-label="Experiment mapping">
          <div className="mapping-card"><span>Demonstration 1</span><b className="vector-orb v1">v₁</b><i>→</i><strong>dax</strong></div>
          <div className="mapping-card"><span>Demonstration 2</span><b className="vector-orb v2">v₂</b><i>→</i><strong>wug</strong></div>
          <div className="mapping-card test-mapping"><span>Test</span><b className="vector-orb split-orb">?</b><i>→</i><strong>A or B</strong></div>
          <p>Same sentence · same token · different direction</p>
        </div>
      </section>

      <ExperimentPrimer
        question="Does the untrained model use the identity of the injected vector?"
        idea="If it does, its answer should switch with the test vector: v₁ should produce dax, while v₂ should produce wug. Overall accuracy alone is not enough, because always choosing one option would also score 50% in this balanced design."
        fixed="The visible sentence, prompt wording, demonstration order, option order, injection layer, token position, and intervention strength."
        changed="Only whether the test sentence receives v₁ or v₂. Each sentence is tested once with each vector."
        measured="The next-token logits for A and B after “The answer is:”. We also count which option is chosen in every trial."
        answer="No. The model chose A in all 128 trials. Its 50% accuracy is a response bias, not evidence that it distinguished v₁ from v₂."
      />

      <section className="result-section" id="result">
        <div className="section-heading">
          <div><span className="section-index">01</span><span className="eyebrow">Result</span><h2>The model always answered A</h2></div>
          <p>Because half the trials have A as the correct answer, always choosing A produces exactly 50% accuracy. Splitting the results by test vector reveals the failure: all v₁ trials are correct and all v₂ trials are wrong.</p>
        </div>

        <div className="metric-grid">
          <MetricCard eyebrow="Overall accuracy" value={pct(results.aggregate.accuracy)} detail={`${results.aggregate.n_correct} / ${results.aggregate.n_trials} · CI ${pct(results.aggregate.ci_95.low)}–${pct(results.aggregate.ci_95.high)}`} tone="ochre" />
          <MetricCard eyebrow="Mean signed margin" value={signed(results.aggregate.mean_margin)} detail={`SD ${results.aggregate.margin_std.toFixed(3)} · logits`} />
          <MetricCard eyebrow="v₁ → A · dax" value={pct(results.aggregate.by_test_vector.v1.accuracy)} detail={`${results.aggregate.by_test_vector.v1.n_correct} / ${results.aggregate.by_test_vector.v1.n_trials} correct`} tone="green" />
          <MetricCard eyebrow="v₂ → B · wug" value={pct(results.aggregate.by_test_vector.v2.accuracy)} detail={`${results.aggregate.by_test_vector.v2.n_correct} / ${results.aggregate.by_test_vector.v2.n_trials} correct`} tone="red" />
        </div>

        <div className="result-grid">
          <article className="paper-panel accuracy-panel">
            <div className="panel-heading"><div><span className="eyebrow">Binomial estimate</span><h3>Overall accuracy</h3></div><span className="panel-note">n = 128</span></div>
            <AccuracyPlot results={results} />
            <ReadingKey>The dot is the observed accuracy. The horizontal line is a 95% binomial confidence interval. The dashed vertical rule marks 50% chance.</ReadingKey>
            <p className="figure-caption">Accuracy alone is ambiguous here. The next panel shows why the 50% result is not successful vector recognition.</p>
          </article>
          <article className="paper-panel bias-panel">
            <div className="panel-heading"><div><span className="eyebrow">Response audit</span><h3>Predicted option</h3></div><span className="panel-note">all trials</span></div>
            <div className="choice-bars">
              <div className="choice-row"><span><b>A</b> dax</span><div><i className="choice-a" style={{ width: `${(predictedA / allTrials.length) * 100}%` }} /></div><strong>{predictedA}</strong></div>
              <div className="choice-row"><span><b>B</b> wug</span><div><i className="choice-b" style={{ width: `${(predictedB / allTrials.length) * 100}%` }} /></div><strong>{predictedB}</strong></div>
            </div>
            <div className="bias-callout"><span>128 : 0</span><p>The option response is invariant to the test vector in this pilot.</p></div>
          </article>
          <article className="paper-panel margin-panel">
            <div className="panel-heading"><div><span className="eyebrow">Paired by carrier</span><h3>Correct-choice margin</h3></div><div className="dot-legend"><span><i className="v1-dot" /> v₁</span><span><i className="v2-dot" /> v₂</span></div></div>
            <PairedMarginPlot episodes={results.episodes} />
            <ReadingKey>Each narrow column is one sentence. Green is the v₁ trial and red is the v₂ trial. Values above zero favor the correct answer; values below zero favor the wrong answer.</ReadingKey>
            <p className="figure-caption">The same A-favoring scores become positive for v₁, where A is correct, and negative for v₂, where B is correct.</p>
          </article>
        </div>
      </section>

      <section className="audit-section" id="audit">
        <div className="section-heading compact-heading">
          <div><span className="section-index">02</span><span className="eyebrow">Trial audit</span><h2>Inspect every intervention and decision</h2></div>
          <p>The matrix shows all 64 paired episodes. Select any cell or table row to inspect its carrier, injection sequence, logits, token site, vector seed, fingerprint, and leading coordinates.</p>
        </div>

        <article className="matrix-panel paper-panel">
          <div className="panel-heading matrix-heading">
            <div><span className="eyebrow">64 episodes · 128 trials</span><h3>Paired outcome matrix</h3></div>
            <div className="matrix-legend"><span><i className="correct-key" /> correct</span><span><i className="error-key" /> incorrect</span><span><i className="selected-key" /> selected</span></div>
          </div>
          <TrialMatrix episodes={results.episodes} selectedId={selectedId} onSelect={setSelectedId} />
        </article>

        <div className="audit-workspace">
          <div className="trial-browser">
            <div className="filter-bar">
              <label className="search-field">
                <span>Search trials</span>
                <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="ID, sentence, label…" />
              </label>
              <label><span>Outcome</span><select value={outcome} onChange={(event) => setOutcome(event.target.value as OutcomeFilter)}><option value="all">All outcomes</option><option value="correct">Correct only</option><option value="incorrect">Errors only</option></select></label>
              <label><span>Test vector</span><select value={vector} onChange={(event) => setVector(event.target.value as VectorFilter)}><option value="all">v₁ + v₂</option><option value="v1">v₁ only</option><option value="v2">v₂ only</option></select></label>
              <label><span>Sort</span><select value={sort} onChange={(event) => setSort(event.target.value as SortKey)}><option value="episode">Episode</option><option value="margin-low">Margin ↑</option><option value="margin-high">Margin ↓</option></select></label>
              <span className="result-count">{filteredTrials.length} shown</span>
            </div>

            <div className="table-shell">
              <table>
                <thead><tr><th>Trial</th><th>Carrier sentence</th><th>Injected</th><th>Expected</th><th>Predicted</th><th>z<sub>A</sub></th><th>z<sub>B</sub></th><th>Margin</th><th>Outcome</th></tr></thead>
                <tbody>
                  {filteredTrials.map((trial) => (
                    <tr key={trial.trial_id} className={selectedId === trial.trial_id ? "selected-row" : ""} onClick={() => setSelectedId(trial.trial_id)}>
                      <td><button className="trial-link" onClick={() => setSelectedId(trial.trial_id)}>{trial.trial_id}</button></td>
                      <td className="sentence-cell">{trial.sentence}</td>
                      <td><span className={`vector-badge ${trial.test_vector}`}>{trial.test_vector === "v1" ? "v₁" : "v₂"}</span></td>
                      <td>{trial.correct_option} · {trial.correct_label}</td>
                      <td>{trial.predicted_option} · {trial.predicted_label}</td>
                      <td className="mono">{trial.logit_A.toFixed(2)}</td>
                      <td className="mono">{trial.logit_B.toFixed(2)}</td>
                      <td className={`mono margin-value ${trial.margin >= 0 ? "positive-margin" : "negative-margin"}`}>{signed(trial.margin)}</td>
                      <td><span className={`outcome-dot ${trial.correct ? "is-correct" : "is-error"}`}><i />{trial.correct ? "Correct" : "Error"}</span></td>
                    </tr>
                  ))}
                  {filteredTrials.length === 0 && <tr><td colSpan={9} className="empty-state">No trials match these filters.</td></tr>}
                </tbody>
              </table>
            </div>
          </div>
          <TrialDetail trial={selectedTrial} episode={selectedEpisode} />
        </div>
      </section>

      <section className="method-section" id="method">
        <div className="section-heading compact-heading">
          <div><span className="section-index">03</span><span className="eyebrow">Exact procedure</span><h2>How the 128 trials were run</h2></div>
          <p>The model was not asked to generate a long response. We supplied the demonstrations and answer prefix, ran the complete conversation once, and compared only the next-token scores for A and B.</p>
        </div>

        <div className="method-flow" aria-label="Experiment workflow">
          <div><span>1</span><b>Choose the scale</b><p>Measure 100 ordinary residual norms; set α to 10% of the median</p></div><i>→</i>
          <div><span>2</span><b>Sample vectors</b><p>Make 64 fresh, perpendicular unit-vector pairs</p></div><i>→</i>
          <div><span>3</span><b>Add them internally</b><p>At the final period in both examples and the test sentence</p></div><i>→</i>
          <div><span>4</span><b>Compare A and B</b><p>Run both possible test vectors for every pair</p></div>
        </div>

        <div className="method-grid">
          <article className="paper-panel prompt-panel">
            <div className="panel-heading"><div><span className="eyebrow">Visible carrier protocol</span><h3>Complete prompt</h3></div><span className="panel-note">episode 01</span></div>
            <PromptPreview sentence={results.episodes[0].sentence} />
          </article>
          <article className="paper-panel specification-panel">
            <div className="panel-heading"><div><span className="eyebrow">Recorded configuration</span><h3>Run specification</h3></div><span className="status-seal">verified</span></div>
            <dl className="spec-list">
              <div><dt>Checkpoint</dt><dd>{results.run.model_id}</dd></div>
              <div><dt>Revision</dt><dd className="mono truncate">{results.run.model_revision}</dd></div>
              <div><dt>Precision</dt><dd>{results.run.model_dtype.toUpperCase()} · unquantized</dd></div>
              <div><dt>GPU</dt><dd>{results.run.gpu}</dd></div>
              <div><dt>Transformer blocks</dt><dd>{results.design.num_layers}</dd></div>
              <div><dt>Injection block</dt><dd>{results.design.layer_index_zero_based} zero-based · middle</dd></div>
              <div><dt>Residual site</dt><dd>{results.design.site}</dd></div>
              <div><dt>Hidden width</dt><dd>{results.design.hidden_size.toLocaleString()}</dd></div>
              <div><dt>Calibration R<sub>ℓ</sub></dt><dd>{results.calibration.median_residual_norm.toFixed(4)}</dd></div>
              <div><dt>Strength α</dt><dd>{results.design.alpha.toFixed(4)} · ρ {results.design.rho}</dd></div>
              <div><dt>Candidate A</dt><dd>token {results.tokenization.A.id} · {JSON.stringify(results.tokenization.A.decoded_piece)}</dd></div>
              <div><dt>Candidate B</dt><dd>token {results.tokenization.B.id} · {JSON.stringify(results.tokenization.B.decoded_piece)}</dd></div>
              <div><dt>Single-token check</dt><dd>{results.tokenization.verified_single_token ? "Passed" : "Failed"}</dd></div>
              <div><dt>Evaluation</dt><dd>eval mode · deterministic · no sampling</dd></div>
              <div><dt>Run seed</dt><dd>{results.run.seed}</dd></div>
              <div><dt>Wall time</dt><dd>{results.run.duration_seconds.toFixed(2)} s</dd></div>
            </dl>
            <div className="artifact-links">
              <a href="/data/experiment-results.json" download>Download audit JSON <span>236 KB</span></a>
              <a href="/data/intervention-vectors-fp32.npz" download>Download full FP32 vectors <span>2.3 MB</span></a>
            </div>
          </article>
        </div>

        <div className="interpretation-note">
          <span>Interpretation boundary</span>
          <p>This no-controls pilot does not establish introspection. Here it does not even show above-chance marker identification: the aggregate 50% result is fully explained by a fixed A/dax response. Controls that permute option order and demonstration mapping are necessary before attributing any future effect to intervention identity.</p>
        </div>
      </section>

      <footer>
        <div><span className="wordmark-mark">H</span><p><b>Hidden Marker Lab</b><br />Experiment 01 · complete audit artifact</p></div>
        <p>Qwen3-14B · {results.aggregate.n_trials} trials · seed {results.run.seed}</p>
      </footer>
    </main>
  );
}
