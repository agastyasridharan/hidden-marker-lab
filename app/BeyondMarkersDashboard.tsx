"use client";

import { useEffect, useMemo, useState } from "react";
import { ResearchHeader } from "./ResearchHeader";

type Variant = "base" | "hidden_marker_lora_seed_20260820";
type AuditKind = "sae" | "ablation" | "jlens" | "verbal" | "evalaware";

type SaeSummary = {
  variant: Variant;
  condition: "positive" | "negative" | "random" | "none";
  n: number;
  accuracy: number;
  accuracy_wilson_95: [number, number];
  mean_correct_margin: number;
};

type AblationSummary = {
  predictor_variant: Variant;
  target_variant: Variant;
  self_prediction: boolean;
  n: number;
  exact_accuracy: number;
  exact_wilson_95: [number, number];
  changed_accuracy: number;
  answer_accuracy: number;
  mean_correct_logprob_margin: number;
};

type LensSummary = {
  variant: Variant;
  suite: string;
  readout: "jacobian" | "vanilla";
  n_items: number;
  n_intermediates: number;
  token_coverage: number;
  median_best_rank: number;
  pass_at_1: number;
  pass_at_5: number;
  pass_at_10: number;
  pass_at_50: number;
};

type Calibration = {
  variant: Variant;
  n: number;
  mean_pearson_sampled_vocab: number;
  mean_spearman_sampled_vocab: number;
  mean_top_10_overlap: number;
  mean_top_50_overlap: number;
};

type AwarenessSummary = {
  variant: Variant;
  readout: "jacobian" | "vanilla" | "explicit_report" | "internal_report_association";
  n: number;
  single_factor_vs_baseline_auc?: number;
  auc_bootstrap_95?: [number, number];
  all_paired_mean_change?: number;
  all_paired_bootstrap_95?: [number, number];
  condition?: string;
};

type VerbalCurve = {
  variant: Variant;
  prefill: "word" | "default";
  strength: number;
  n: number;
  slot_rr_q25: number;
  slot_rr_median: number;
  slot_rr_q75: number;
  slot_rr_mean: number;
  slot_rank_median: number;
  best_other_position_rr_median: number;
  best_other_position_rank_median: number;
  shuffled_slot_rr_median: number;
  shuffled_slot_rank_median: number;
  slot_beats_best_other_rate: number;
  target_beats_shuffled_rate: number;
  slot_top_1_rate: number;
  slot_top_5_rate: number;
  slot_top_10_rate: number;
  slot_top_100_rate: number;
  mean_target_minus_shuffled_logit: number;
};

type Download = { url: string; sha256: string; bytes: number };

export type Results = {
  status: "complete";
  completed_at: string;
  models: { base: string; trained: string; model_revision: string; adapter_sha256: string; dtype: string; quantized: boolean };
  sae: {
    design: { n_items: number; n_options: number; rho: number; residual_norm_median: number; alpha: number };
    summary: SaeSummary[];
    paired_condition_comparison: Array<{ variant: Variant; contrast: string; n: number; accuracy_difference: number; bootstrap_95: [number, number] }>;
  };
  ablation: {
    design: { n_items: number; target_disagreement_n: number };
    summary: AblationSummary[];
    self_other_comparison: Array<{ predictor_variant: Variant; scope: string; n: number; self_minus_other_exact_accuracy: number | null; bootstrap_95: [number, number] | null }>;
  };
  jlens: { summary: LensSummary[]; calibration: Calibration[] };
  verbal_introspection: {
    design: {
      released_concepts: number;
      usable_single_token_concepts: number;
      excluded_concepts: Array<{ surface: string; contextual_suffix_ids: number[] }>;
      block_output_layers_zero_indexed: number[];
      question_token_count: number;
    };
    curves: VerbalCurve[];
  };
  evalaware: {
    design: { lens_truncation_by_condition: Array<{ condition: string; n: number; truncated_n: number }> };
    summary: AwarenessSummary[];
    factor_effects: Array<{ variant: Variant; readout: string; factor: string; valence: string; n: number; paired_mean_change: number; paired_bootstrap_95: [number, number] }>;
  };
  downloads: Record<string, Download>;
  sources: Record<string, string>;
};

type SaeRecord = {
  variant: Variant; condition: string; item_index: number; feature_index: number; reference: string;
  correct_letter: string; prediction_letter: string; is_correct: boolean; correct_margin: number;
  options: string[]; logits: number[]; site: string; alpha: number;
};
type AblationRecord = {
  predictor_variant: Variant; target_variant: Variant; item_index: number; source_index: number;
  question: string; choices: string[]; hint: string; target_changed: boolean;
  target_without_hint_answer: string; prediction_changed: boolean; prediction_answer: string;
  prediction_text: string; exact_correct: boolean; correct_logprob_margin: number; candidate_logprobs: number[];
};
type LensRecord = {
  variant: Variant; suite: string; item_index: number; name?: string; prompt: string; target?: string;
  readout_token: string; intermediates: Array<{ intermediate: string; jacobian_best_rank?: number; jacobian_best_layer?: number; vanilla_best_rank?: number; vanilla_best_layer?: number }>;
  calibration: { spearman_sampled_vocab: number; top_10_overlap: number };
};
type AwarenessRecord = {
  variant: Variant; task_id: string; task_name: string; valence: string; condition: string; prompt: string;
  jacobian: { score: number; best_layer: number }; vanilla: { score: number; best_layer: number };
  yes_logit: number; no_logit: number; report_margin_yes_minus_no: number; report_prediction: string;
  lens_input_tokens: number; lens_input_truncated: boolean;
};
type VerbalRecord = {
  variant: Variant; prefill: "word" | "default"; strength: number; concept_index: number;
  concept: string; surface: string; token_id: number; shuffled_surface: string;
  slot_rank: number; slot_reciprocal_rank: number; best_other_position_rank: number;
  shuffled_slot_rank: number; target_slot_logit: number; shuffled_slot_logit: number;
  target_minus_shuffled_logit: number; report_position: number;
  assistant_position_tokens: string[]; target_ranks_by_assistant_position: number[];
};
type AnyAuditRecord = SaeRecord | AblationRecord | LensRecord | VerbalRecord | AwarenessRecord;

const variants: Variant[] = ["base", "hidden_marker_lora_seed_20260820"];
const conditions: SaeSummary["condition"][] = ["positive", "negative", "random", "none"];
const suites = ["association", "multihop", "multilingual", "order-ops", "poetry", "typo"];
const auditLabels: Record<AuditKind, string> = { sae: "SAE features", ablation: "Input ablation", jlens: "Jacobian lens", verbal: "Injected concepts", evalaware: "Eval awareness" };
const downloadKeys: Record<AuditKind, string> = { sae: "sae_records", ablation: "ablation_records", jlens: "jlens_records", verbal: "verbal_records", evalaware: "evalaware_records" };
const factorLabels: Record<string, { name: string; note: string }> = {
  F1: { name: "Stimulus artificiality", note: "placeholder names, domains, or values" },
  F2: { name: "Scenario coherence", note: "contradictions or overdetermined details" },
  F3: { name: "Prose register", note: "casual user voice versus formal vignette" },
  F4: { name: "Ecological completeness", note: "full context versus a bare request" },
  F5: { name: "Stakes telegraphing", note: "the evaluated dimension is named directly" },
  F6: { name: "Paradigm recognizability", note: "benchmark or jailbreak scaffolding" },
  F7: { name: "Task format", note: "natural prose versus rigid template" },
  F8: { name: "Evaluation disclosure", note: "testing, rubric, or output-convention language" },
};

const pct = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const pp = (value: number) => `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)} pp`;
const num = (value: number, digits = 2) => value.toFixed(digits);
const modelName = (variant: Variant) => variant === "base" ? "Base" : "Hidden-marker LoRA";
const rankPosition = (rank: number) => `${Math.min(100, Math.log10(Math.max(1, rank)) / 4 * 100)}%`;

function findSae(results: Results, variant: Variant, condition: SaeSummary["condition"]) {
  return results.sae.summary.find((row) => row.variant === variant && row.condition === condition)!;
}

function findAblation(results: Results, predictor: Variant, target: Variant) {
  return results.ablation.summary.find((row) => row.predictor_variant === predictor && row.target_variant === target)!;
}

function findVerbal(results: Results, variant: Variant, strength: number, prefill: VerbalCurve["prefill"] = "word") {
  return results.verbal_introspection.curves.find((row) => row.variant === variant && row.prefill === prefill && row.strength === strength)!;
}

function SaePlot({ results }: { results: Results }) {
  return <figure className="bm-plot">
    <div className="bm-axis"><span>0</span><span>25</span><span>50</span><span>75</span><span>100%</span></div>
    {conditions.map((condition) => <div className="bm-dot-row" key={condition}>
      <div><strong>{condition}</strong><small>{condition === "positive" ? "correct SAE direction" : condition === "negative" ? "same direction, sign reversed" : condition === "random" ? "matched-norm random direction" : "no intervention"}</small></div>
      <div className="bm-dot-track">
        <i className="bm-chance-125" />
        {variants.map((variant) => {
          const row = findSae(results, variant, condition);
          return <b key={variant} className={`bm-result-dot ${variant === "base" ? "bm-base" : "bm-trained"}`} style={{ left: `${row.accuracy * 100}%` }} title={`${modelName(variant)} ${pct(row.accuracy)}`} />;
        })}
      </div>
      <output><span>{pct(findSae(results, "base", condition).accuracy)}</span><b>{pct(findSae(results, "hidden_marker_lora_seed_20260820", condition).accuracy)}</b></output>
    </div>)}
    <figcaption><span><i className="bm-key-base" />Base</span><span><i className="bm-key-trained" />Hidden-marker LoRA</span><span><i className="bm-key-chance" />Eight-way chance, 12.5%</span></figcaption>
  </figure>;
}

function AblationMatrix({ results }: { results: Results }) {
  return <div className="bm-matrix">
    <div className="bm-matrix-corner">Predictor ↓<br />Target →</div>
    {variants.map((target) => <div className="bm-matrix-head" key={target}>{modelName(target)}</div>)}
    {variants.flatMap((predictor) => [
      <div className="bm-matrix-head bm-row-head" key={`${predictor}-head`}>{modelName(predictor)}</div>,
      ...variants.map((target) => {
        const row = findAblation(results, predictor, target);
        return <div className={`bm-matrix-value ${predictor === target ? "bm-self-cell" : ""}`} key={`${predictor}-${target}`}>
          <strong>{pct(row.exact_accuracy)}</strong>
          <span>{row.n} items</span>
          <small>change status {pct(row.changed_accuracy)}<br />answer letter {pct(row.answer_accuracy)}</small>
        </div>;
      }),
    ])}
  </div>;
}

function LensPlot({ results }: { results: Results }) {
  return <figure className="bm-plot bm-lens-plot">
    <div className="bm-axis"><span>0</span><span>25</span><span>50</span><span>75</span><span>100%</span></div>
    {suites.map((suite) => {
      const rows = variants.map((variant) => results.jlens.summary.find((row) => row.variant === variant && row.suite === suite && row.readout === "jacobian")!);
      return <div className="bm-bar-row" key={suite}>
        <div><strong>{suite}</strong><small>known intermediate in top 10 at any layer</small></div>
        <div className="bm-pair-bars"><i style={{ width: `${rows[0].pass_at_10 * 100}%` }} /><b style={{ width: `${rows[1].pass_at_10 * 100}%` }} /></div>
        <output><span>{pct(rows[0].pass_at_10)}</span><b>{pct(rows[1].pass_at_10)}</b></output>
      </div>;
    })}
    <figcaption><span><i className="bm-key-base" />Published base-model lens</span><span><i className="bm-key-trained" />Same lens applied with LoRA enabled</span></figcaption>
  </figure>;
}

function VerbalPlot({ results }: { results: Results }) {
  const strengths = [0, 0.005, 0.01, 0.015, 0.02, 0.025];
  return <figure className="bm-plot bm-verbal-plot">
    <div className="bm-axis"><span>rank 1</span><span>10</span><span>100</span><span>1,000</span><span>10,000</span></div>
    {strengths.map((strength) => {
      const rows = variants.map((variant) => findVerbal(results, variant, strength));
      return <div className="bm-dot-row" key={strength}>
        <div><strong>s = {strength.toFixed(3)}</strong><small>median next-token rank; lower is better</small></div>
        <div className="bm-dot-track">{rows.map((row, index) => <b key={row.variant} className={`bm-result-dot ${index === 0 ? "bm-base" : "bm-trained"}`} style={{ left: rankPosition(row.slot_rank_median) }} title={`${modelName(row.variant)} median rank ${row.slot_rank_median}`} />)}</div>
        <output><span>{Math.round(rows[0].slot_rank_median).toLocaleString()}</span><b>{Math.round(rows[1].slot_rank_median).toLocaleString()}</b></output>
      </div>;
    })}
    <figcaption><span><i className="bm-key-base" />Base</span><span><i className="bm-key-trained" />Hidden-marker LoRA</span><span>Logarithmic rank axis. The released “word” prefill is shown; the second prefill is retained in the audit.</span></figcaption>
  </figure>;
}

function AwarenessPlot({ results }: { results: Results }) {
  const readouts: Array<{ key: AwarenessSummary["readout"]; label: string }> = [
    { key: "jacobian", label: "Jacobian-lens internal score" },
    { key: "vanilla", label: "Ordinary logit-lens internal score" },
    { key: "explicit_report", label: "Explicit YES minus NO report" },
  ];
  return <figure className="bm-plot bm-awareness-plot">
    <div className="bm-axis"><span>0</span><span>.25</span><span>.50</span><span>.75</span><span>1.0 AUC</span></div>
    {readouts.map(({ key, label }) => {
      const rows = variants.map((variant) => results.evalaware.summary.find((row) => row.variant === variant && row.readout === key)!);
      return <div className="bm-dot-row" key={key}>
        <div><strong>{label}</strong><small>single-factor variants versus baseline</small></div>
        <div className="bm-dot-track"><i className="bm-chance-50" />{variants.map((variant, index) => <b key={variant} className={`bm-result-dot ${index === 0 ? "bm-base" : "bm-trained"}`} style={{ left: `${Number(rows[index].single_factor_vs_baseline_auc) * 100}%` }} />)}</div>
        <output><span>{num(Number(rows[0].single_factor_vs_baseline_auc))}</span><b>{num(Number(rows[1].single_factor_vs_baseline_auc))}</b></output>
      </div>;
    })}
    <figcaption><span><i className="bm-key-chance" />AUC .50 means no discrimination</span><span>Higher means a larger score for the more evaluation-like rendering.</span></figcaption>
  </figure>;
}

function Findings({ results }: { results: Results }) {
  const saeSpecificity = results.sae.paired_condition_comparison.find((row) => row.variant === "hidden_marker_lora_seed_20260820" && row.contrast === "positive_minus_none")!;
  const selfContrast = results.ablation.self_other_comparison.find((row) => row.predictor_variant === "hidden_marker_lora_seed_20260820" && row.scope === "target_disagreement");
  const baseCal = results.jlens.calibration.find((row) => row.variant === "base")!;
  const trainedCal = results.jlens.calibration.find((row) => row.variant !== "base")!;
  const verbalBase = findVerbal(results, "base", 0.025);
  const verbalTrained = findVerbal(results, "hidden_marker_lora_seed_20260820", 0.025);
  const explicitBase = results.evalaware.summary.find((row) => row.variant === "base" && row.readout === "explicit_report")!;
  const explicitTrained = results.evalaware.summary.find((row) => row.variant !== "base" && row.readout === "explicit_report")!;
  return <div className="bm-finding-grid">
    <article><span>SAE direction specificity</span><strong>{pp(saeSpecificity.accuracy_difference)}</strong><p>Correct SAE direction minus no intervention for the LoRA. Paired 95% interval: {pp(saeSpecificity.bootstrap_95[0])} to {pp(saeSpecificity.bootstrap_95[1])}. The interval includes zero.</p></article>
    <article><span>Privileged self-prediction</span><strong>{selfContrast?.self_minus_other_exact_accuracy === null || !selfContrast ? "not identified" : pp(selfContrast.self_minus_other_exact_accuracy)}</strong><p>LoRA self-target accuracy minus its accuracy for the base target on the {results.ablation.design.target_disagreement_n} items where the target outcomes differ.</p></article>
    <article><span>Lens calibration</span><strong>{(trainedCal.mean_spearman_sampled_vocab - baseCal.mean_spearman_sampled_vocab).toFixed(3)}</strong><p>Change in mean final-layer sampled-vocabulary Spearman correlation when the fixed base-model lens is applied to the LoRA. A large loss limits the mechanistic comparison.</p></article>
    <article><span>Injected-concept report</span><strong>{Math.round(verbalTrained.slot_top_10_rate * verbalTrained.n)} / {verbalTrained.n}</strong><p>LoRA concepts ranked in the top 10 at maximum strength. Base also reached {Math.round(verbalBase.slot_top_10_rate * verbalBase.n)} / {verbalBase.n}. The LoRA median rank was {Math.round(verbalTrained.slot_rank_median).toLocaleString()}.</p></article>
    <article><span>Explicit eval awareness</span><strong>{(Number(explicitTrained.single_factor_vs_baseline_auc) - Number(explicitBase.single_factor_vs_baseline_auc)).toFixed(3)}</strong><p>Change in AUC for the direct YES/NO evaluation-awareness report. This measures sensitivity to benchmark-like cues, not deception.</p></article>
  </div>;
}

function recordIdentity(kind: AuditKind, row: AnyAuditRecord) {
  if (kind === "sae") { const item = row as SaeRecord; return `${modelName(item.variant)} · ${item.condition} · feature ${item.feature_index}`; }
  if (kind === "ablation") { const item = row as AblationRecord; return `${modelName(item.predictor_variant)} predicts ${modelName(item.target_variant)} · item ${item.item_index}`; }
  if (kind === "jlens") { const item = row as LensRecord; return `${modelName(item.variant)} · ${item.suite} · item ${item.item_index}`; }
  if (kind === "verbal") { const item = row as VerbalRecord; return `${modelName(item.variant)} · ${item.surface} · s=${item.strength.toFixed(3)}`; }
  const item = row as AwarenessRecord; return `${modelName(item.variant)} · ${item.condition} · ${item.task_id}`;
}

function recordSearchText(row: AnyAuditRecord) { return JSON.stringify(row).toLowerCase(); }

function AuditDetail({ kind, row }: { kind: AuditKind; row: AnyAuditRecord }) {
  return <aside className="bm-audit-detail">
    <span className="eyebrow">Selected record</span>
    <h3>{recordIdentity(kind, row)}</h3>
    {kind === "sae" && (() => { const item = row as SaeRecord; return <>
      <p>{item.reference}</p><dl><div><dt>Correct option</dt><dd>{item.correct_letter}</dd></div><div><dt>Prediction</dt><dd>{item.prediction_letter}</dd></div><div><dt>Correct margin</dt><dd>{num(item.correct_margin, 3)}</dd></div><div><dt>Injection</dt><dd>{item.site}</dd></div></dl>
      <ol className="bm-option-list">{item.options.map((option, index) => <li className={String.fromCharCode(65 + index) === item.correct_letter ? "bm-correct-option" : ""} key={`${index}-${option}`}><b>{String.fromCharCode(65 + index)}</b><span>{option}</span><code>{num(item.logits[index], 3)}</code></li>)}</ol>
    </>; })()}
    {kind === "ablation" && (() => { const item = row as AblationRecord; return <>
      <p>{item.question}</p><dl><div><dt>Hint</dt><dd>{item.hint}</dd></div><div><dt>Target outcome</dt><dd>{item.target_changed ? "changed" : "unchanged"}; {item.target_without_hint_answer}</dd></div><div><dt>Prediction</dt><dd>{item.prediction_text}</dd></div><div><dt>Exact score</dt><dd>{item.exact_correct ? "correct" : "incorrect"}</dd></div><div><dt>Log-probability margin</dt><dd>{num(item.correct_logprob_margin, 3)}</dd></div></dl>
    </>; })()}
    {kind === "jlens" && (() => { const item = row as LensRecord; return <>
      <p className="bm-prewrap">{item.prompt}</p><dl><div><dt>Readout token</dt><dd>{JSON.stringify(item.readout_token)}</dd></div><div><dt>Calibration ρ</dt><dd>{num(item.calibration.spearman_sampled_vocab, 3)}</dd></div></dl>
      <div className="bm-intermediate-list">{item.intermediates.map((intermediate) => <div key={intermediate.intermediate}><b>{intermediate.intermediate}</b><span>J rank {intermediate.jacobian_best_rank ?? "not tokenized"} · vanilla rank {intermediate.vanilla_best_rank ?? "not tokenized"}</span></div>)}</div>
    </>; })()}
    {kind === "verbal" && (() => { const item = row as VerbalRecord; return <>
      <p>The J-lens direction for <b>{item.surface}</b> was added at every token of the final user question, across the fixed workspace layer band. No Assistant token was directly edited.</p>
      <dl><div><dt>Report-slot rank</dt><dd>{item.slot_rank}</dd></div><div><dt>Best earlier rank</dt><dd>{item.best_other_position_rank}</dd></div><div><dt>Shuffled control</dt><dd>{item.shuffled_surface}, rank {item.shuffled_slot_rank}</dd></div><div><dt>Target / shuffled logits</dt><dd>{num(item.target_slot_logit, 2)} / {num(item.shuffled_slot_logit, 2)}</dd></div><div><dt>Target minus shuffled</dt><dd>{num(item.target_minus_shuffled_logit, 2)}</dd></div><div><dt>Prefill</dt><dd>{item.prefill}</dd></div></dl>
      <div className="bm-intermediate-list">{item.assistant_position_tokens.map((token, index) => <div key={`${index}-${token}`}><b>{JSON.stringify(token)}</b><span>target rank {item.target_ranks_by_assistant_position[index]}{index + 1 === item.assistant_position_tokens.length ? " · scored report position" : ""}</span></div>)}</div>
    </>; })()}
    {kind === "evalaware" && (() => { const item = row as AwarenessRecord; return <>
      <p className="bm-prewrap">{item.prompt}</p><dl><div><dt>Jacobian score</dt><dd>{num(item.jacobian.score, 3)} at layer {item.jacobian.best_layer}</dd></div><div><dt>Vanilla score</dt><dd>{num(item.vanilla.score, 3)} at layer {item.vanilla.best_layer}</dd></div><div><dt>Lens input</dt><dd>{item.lens_input_tokens} tokens{item.lens_input_truncated ? "; truncated" : ""}</dd></div><div><dt>YES / NO logits</dt><dd>{num(item.yes_logit, 2)} / {num(item.no_logit, 2)}</dd></div><div><dt>Explicit prediction</dt><dd>{item.report_prediction}</dd></div></dl>
    </>; })()}
  </aside>;
}

function AuditLedger({ results }: { results: Results }) {
  const [kind, setKind] = useState<AuditKind>("sae");
  const [rows, setRows] = useState<AnyAuditRecord[]>([]);
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<AnyAuditRecord | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const pageSize = 50;

  useEffect(() => {
    let current = true;
    fetch(results.downloads[downloadKeys[kind]].url).then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    }).then((data: AnyAuditRecord[]) => { if (current) { setRows(data); setSelected(data[0] ?? null); } })
      .catch((reason: Error) => { if (current) setError(reason.message); })
      .finally(() => { if (current) setLoading(false); });
    return () => { current = false; };
  }, [kind, results]);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return needle ? rows.filter((row) => recordSearchText(row).includes(needle)) : rows;
  }, [query, rows]);
  const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const visible = filtered.slice(page * pageSize, (page + 1) * pageSize);
  const chooseKind = (next: AuditKind) => {
    setLoading(true);
    setError("");
    setRows([]);
    setSelected(null);
    setPage(0);
    setKind(next);
  };

  return <div className="bm-audit-workspace">
    <div className="bm-ledger">
      <div className="bm-audit-tabs">{(Object.keys(auditLabels) as AuditKind[]).map((key) => <button className={kind === key ? "active" : ""} onClick={() => chooseKind(key)} key={key}>{auditLabels[key]}</button>)}</div>
      <label className="bm-audit-search"><span>Search every field</span><input value={query} onChange={(event) => { setQuery(event.target.value); setPage(0); }} placeholder="feature, prompt, model, condition…" /></label>
      <div className="bm-ledger-count">{loading ? "Loading records…" : error ? `Could not load audit: ${error}` : `${filtered.length.toLocaleString()} of ${rows.length.toLocaleString()} records`}</div>
      {!loading && !error && <div className="bm-table-shell"><table><thead><tr><th>Record</th><th>Condition</th><th>Measured result</th><th>Outcome</th></tr></thead><tbody>{visible.map((row, index) => {
        const identity = recordIdentity(kind, row);
        let condition = ""; let measure = ""; let outcome = "";
        if (kind === "sae") { const item = row as SaeRecord; condition = item.condition; measure = `margin ${num(item.correct_margin, 2)}`; outcome = item.is_correct ? "correct" : "incorrect"; }
        if (kind === "ablation") { const item = row as AblationRecord; condition = `${modelName(item.predictor_variant)} → ${modelName(item.target_variant)}`; measure = `margin ${num(item.correct_logprob_margin, 2)}`; outcome = item.exact_correct ? "correct" : "incorrect"; }
        if (kind === "jlens") { const item = row as LensRecord; condition = item.suite; measure = `${item.intermediates.length} intermediates`; outcome = `ρ ${num(item.calibration.spearman_sampled_vocab, 2)}`; }
        if (kind === "verbal") { const item = row as VerbalRecord; condition = `${item.prefill}, s=${item.strength.toFixed(3)}`; measure = `rank ${item.slot_rank}`; outcome = `control rank ${item.shuffled_slot_rank}`; }
        if (kind === "evalaware") { const item = row as AwarenessRecord; condition = item.condition; measure = `J ${num(item.jacobian.score, 2)}`; outcome = item.report_prediction; }
        return <tr className={selected === row ? "selected-row" : ""} onClick={() => setSelected(row)} key={`${identity}-${index}`}><td><button>{identity}</button></td><td>{condition}</td><td className="mono">{measure}</td><td>{outcome}</td></tr>;
      })}</tbody></table></div>}
      <div className="bm-pagination"><span>Page {page + 1} of {pages}</span><button disabled={page === 0} onClick={() => setPage((value) => value - 1)}>Previous</button><button disabled={page + 1 >= pages} onClick={() => setPage((value) => value + 1)}>Next</button></div>
    </div>
    {selected ? <AuditDetail kind={kind} row={selected} /> : <aside className="bm-audit-detail"><p>Select a record to inspect its prompt, intervention or readout, candidate scores, and outcome.</p></aside>}
  </div>;
}

function CompletedDashboard({ results }: { results: Results }) {
  const verbalMaximum = variants.map((variant) => findVerbal(results, variant, 0.025));
  return <>
    <section className="bm-result-intro result-section">
      <div className="section-heading"><div><span className="section-index">01</span><span className="eyebrow">Measured results</span><h2>Five different questions, not one introspection score</h2></div><p>The panels are kept separate because they test different claims. A model can improve on one without gaining privileged access to its internal state.</p></div>
      <div className="bm-verdict"><b>Result:</b><p>The hidden-marker LoRA did not show broad transfer. It did not reliably identify unseen SAE directions, it predicted the base model’s ablated answers much better than its own, and its sensitivity to evaluation-like cues was unchanged or slightly lower. The fixed Jacobian lens remained calibrated, but changes in recovered intermediate answers were small and inconsistent. Injected-concept reporting improved for a few words rather than across the full set.</p></div>
      <Findings results={results} />
    </section>
    <section className="result-section bm-result-block">
      <SaePlot results={results} />
    </section>
    <section className="result-section bm-result-block bm-tinted">
      <div className="section-heading"><div><span className="section-index">03</span><span className="eyebrow">Behavioral self-prediction</span><h2>Can the model predict its answer without a hint?</h2></div><p>Rows are predictor models and columns are the target policy being predicted. The diagonal is self-prediction. The off-diagonal cells show how well the same predictor forecasts the other model.</p></div>
      <AblationMatrix results={results} />
      <div className="bm-disagreement-note"><b>{results.ablation.design.target_disagreement_n} / {results.ablation.design.n_items}</b><p>items produce different target outcomes for base and LoRA. Only this subset can distinguish privileged self-prediction from learning a generic response rule.</p></div>
    </section>
    <section className="result-section bm-result-block">
      <div className="section-heading"><div><span className="section-index">04</span><span className="eyebrow">Latent computation</span><h2>Does a fixed Jacobian lens recover intermediate answers?</h2></div><p>The plotted endpoint is pass@10 for the published Qwen3-14B lens. The lens was fitted on the base model, so its calibration on the LoRA is part of the result rather than an implementation detail.</p></div>
      <LensPlot results={results} />
      <div className="bm-calibration-grid">{results.jlens.calibration.map((row) => <article key={row.variant}><span>{modelName(row.variant)}</span><strong>ρ {num(row.mean_spearman_sampled_vocab, 3)}</strong><p>Mean final-layer sampled-vocabulary rank correlation. Top-10 overlap: {pct(row.mean_top_10_overlap)}.</p></article>)}</div>
    </section>
    <section className="result-section bm-result-block bm-tinted">
      <div className="section-heading"><div><span className="section-index">05</span><span className="eyebrow">Concept-specific verbal report</span><h2>Can the model name an injected J-space concept?</h2></div><p>We add a published Jacobian-lens direction at every token of the final user question, across Qwen block outputs 15 through 36. The model then receives a teacher-forced answer prefix ending in an open quote. The plot shows the injected word’s median next-token rank. A rank of 1 means it is the model’s highest-scoring next token. No answer-prefix token is edited.</p></div>
      <VerbalPlot results={results} />
      <div className="bm-calibration-grid">{verbalMaximum.map((row) => <article key={row.variant}><span>{modelName(row.variant)} at s = 0.025</span><strong>{Math.round(row.slot_top_10_rate * row.n)} / {row.n} in top 10</strong><p>Median rank {Math.round(row.slot_rank_median).toLocaleString()}; top 5 in {Math.round(row.slot_top_5_rate * row.n)} cases. The report slot beat every earlier Assistant position in {pct(row.slot_beats_best_other_rate)} and beat the shuffled-word control in {pct(row.target_beats_shuffled_rate)}.</p></article>)}</div>
      <div className="bm-disagreement-note"><b>{results.verbal_introspection.design.usable_single_token_concepts} / {results.verbal_introspection.design.released_concepts}</b><p>released concept words are a single contextual Qwen token. All exclusions and token IDs are listed in the result bundle. Earlier Assistant positions and a shuffled concept are scored as controls for report-position selectivity and concept specificity.</p></div>
    </section>
    <section className="result-section bm-result-block">
      <div className="section-heading"><div><span className="section-index">06</span><span className="eyebrow">Evaluation awareness</span><h2>Do evaluation-like cues become more visible?</h2></div><p>EvalAwareBench renders each task once at baseline and eight more times with one factor varied at a time. AUC measures whether a readout assigns a larger awareness score to the single-factor variants. It does not establish that the model is deceptive or strategically hiding behavior.</p></div>
      <AwarenessPlot results={results} />
      <div className="bm-factor-key">{Object.entries(factorLabels).map(([factor, description]) => <article key={factor}><b>{factor} · {description.name}</b><span>{description.note}</span></article>)}</div>
      <div className="bm-truncation-table"><div><b>Lens input coverage</b><span>prompts above the fixed 512-token limit</span></div>{results.evalaware.design.lens_truncation_by_condition.map((row) => <div key={row.condition}><b>{row.condition}</b><span>{row.truncated_n} / {row.n}</span></div>)}</div>
    </section>
    <section className="audit-section bm-audit-section">
      <div className="section-heading"><div><span className="section-index">07</span><span className="eyebrow">Complete audit</span><h2>Inspect every evaluated record</h2></div><p>Search across prompts, features, models, and conditions. Selecting a row exposes the target, intervention or readout, scores, and prediction used in aggregation.</p></div>
      <AuditLedger results={results} />
    </section>
    <section className="method-section bm-method-section">
      <div className="section-heading"><div><span className="section-index">08</span><span className="eyebrow">Methods and limits</span><h2>What was run, and what was deliberately not claimed</h2></div><p>All comparisons use the same pinned Qwen3-14B weights. The trained condition enables the rank-16, alpha-32 hidden-marker LoRA on blocks 20 through 39.</p></div>
      <div className="bm-method-grid">
        <article><span>1</span><h3>SAE concept identification</h3><p>A published Qwen3-14B sparse autoencoder supplies block-20 residual directions. The model sees a neutral carrier and eight independently generated Neuronpedia descriptions. It receives no activation examples for the target feature.</p><dl><div><dt>Features</dt><dd>256</dd></div><div><dt>Site</dt><dd>block 20 resid_post</dd></div><div><dt>Strength</dt><dd>ρ = {results.sae.design.rho}</dd></div><div><dt>Conditions</dt><dd>positive, negative, random, none</dd></div></dl></article>
        <article><span>2</span><h3>Input-ablation prediction</h3><p>The published Qwen3-8B responses are not reused as labels. Base and LoRA targets are recomputed on Qwen3-14B, with and without each hint, then crossed with both predictors.</p><dl><div><dt>Items</dt><dd>400</dd></div><div><dt>Comparisons</dt><dd>2 predictors × 2 targets</dd></div><div><dt>Scoring</dt><dd>eight equal-length sequences</dd></div><div><dt>Key subset</dt><dd>target disagreements</dd></div></dl></article>
        <article><span>3</span><h3>Jacobian-lens readout</h3><p>The released Qwen3-14B lens is held fixed for both model conditions. Known intermediate answers are ranked across source layers on six synthetic task families. Ordinary logit-lens ranks are retained as a baseline.</p><dl><div><dt>Prompts</dt><dd>551 per model</dd></div><div><dt>Source layers</dt><dd>0 through 38</dd></div><div><dt>Primary display</dt><dd>best-layer pass@10</dd></div><div><dt>Validity check</dt><dd>final-layer calibration</dd></div></dl></article>
        <article><span>4</span><h3>Injected J-space concept</h3><p>The released verbal-introspection prompt asks the model to identify a thought injected on the final user turn. Layer-specific directions are derived from the fixed Qwen Jacobian lens. The same vectors and base-model norm calibration are used for both model variants.</p><dl><div><dt>Concepts</dt><dd>{results.verbal_introspection.design.usable_single_token_concepts}</dd></div><div><dt>Layers</dt><dd>15 through 36 resid_post</dd></div><div><dt>Strengths</dt><dd>0 through 0.025</dd></div><div><dt>Controls</dt><dd>earlier positions, shuffled word</dd></div></dl></article>
        <article><span>5</span><h3>EvalAwareBench</h3><p>Two hundred task templates are rendered at baseline and with each of the eight benchmark factors varied separately. We record a fixed internal awareness lexicon score and a separate direct YES/NO report for each rendering.</p><dl><div><dt>Records</dt><dd>1,800 per model</dd></div><div><dt>Internal readouts</dt><dd>Jacobian and vanilla</dd></div><div><dt>External readout</dt><dd>YES minus NO logit</dd></div><div><dt>Endpoint</dt><dd>pooled and factor-wise paired effects</dd></div></dl></article>
      </div>
      <div className="bm-limit-grid">
        <article><b>This is an adaptation, not a direct replication.</b><p>Li et al. used a separately trained explainer for Llama SAE features. This run instead asks the evaluated Qwen model to classify compatible Qwen SAE directions.</p></article>
        <article><b>No compatible NLA reporter was available.</b><p>The released NLA reproduction targets Qwen2.5-7B. Applying that reporter to Qwen3-14B would not be a faithful model comparison, so NLA is discussed but not scored.</p></article>
        <article><b>Deception was not induced.</b><p>Neither model was trained to pursue a hidden deceptive objective. Eval-awareness scores therefore measure recognition of evaluation-like cues, not deceptive alignment or strategic concealment.</p></article>
        <article><b>A fixed lens can become invalid after fine-tuning.</b><p>If lens calibration deteriorates under the LoRA, lower or higher recovered-rank scores cannot be interpreted cleanly as a change in latent computation.</p></article>
        <article><b>Injected-concept reporting is not sufficient evidence of faithful introspection.</b><p>The intervention direction is derived from the target word’s output geometry. Report-position selectivity and shuffled-word specificity constrain a direct-steering explanation, but this task still does not test reports of naturally occurring, unperturbed states.</p></article>
      </div>
      <div className="bm-downloads">{Object.entries(results.downloads).map(([key, artifact]) => <a href={artifact.url} download key={key}><b>{key.replaceAll("_", " ")}</b><span>{(artifact.bytes / 1_000_000).toFixed(1)} MB · SHA-256 {artifact.sha256.slice(0, 12)}…</span></a>)}</div>
      <div className="bm-sources"><span>Primary sources</span>{Object.entries(results.sources).map(([key, url]) => <a href={url} target="_blank" rel="noreferrer" key={key}>{key.replaceAll("_", " ")} ↗</a>)}</div>
      <p className="bm-run-note">Completed {new Date(results.completed_at).toLocaleString("en-US", { timeZone: "UTC", dateStyle: "medium", timeStyle: "short" })} UTC · Qwen3-14B BF16, unquantized · model revision {results.models.model_revision.slice(0, 12)}… · adapter {results.models.adapter_sha256.slice(0, 12)}…</p>
    </section>
  </>;
}

export function BeyondMarkersDashboard({ initialResults }: { initialResults?: Results }) {
  const [results, setResults] = useState<Results | null>(initialResults ?? null);
  const [error, setError] = useState("");
  useEffect(() => {
    fetch("/data/beyond-markers-results.json", { cache: "no-store" }).then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    }).then((data: Results) => setResults(data)).catch((reason: Error) => setError(reason.message));
  }, []);

  return <main className="bm-page">
    <ResearchHeader active="transfer" status={results ? "Evaluation complete" : "GPU evaluation queued"} />
    <section className="bm-hero">
      <div>
        <span className="eyebrow">Experiment 7 · Transfer evaluation</span>
        <h1>What transfers <em>beyond hidden markers?</em></h1>
        <p className="lede">The LoRA was trained on a narrow task: identify which arbitrary residual-stream direction appeared in context. This evaluation asks whether that training changed any broader ability to describe a semantic feature, predict the model’s own behavior, expose an intermediate computation, report a concept injected through a published mechanistic direction, or recognize that a prompt is an evaluation.</p>
      </div>
      <ol className="bm-question-list">
        <li><b>SAE features</b><span>Can the model name the meaning of a previously unseen feature direction?</span></li>
        <li><b>Input ablation</b><span>Can it predict how removing a hint would change its own answer?</span></li>
        <li><b>Jacobian lens</b><span>Are intermediate answers more legible in the model’s computation?</span></li>
        <li><b>Injected concepts</b><span>Can it report a J-space concept added while it reads the final user question?</span></li>
        <li><b>Evaluation awareness</b><span>Can internal or explicit reports distinguish evaluation-like prompts?</span></li>
      </ol>
    </section>
    {results ? <CompletedDashboard results={results} /> : <section className="bm-pending"><span>Run status</span><div><i /><b>Waiting for an Athena H100</b></div><p>The evaluation code and public artifacts have passed their preflight checks. Result panels will appear here only after both model conditions complete and the record counts are audited.{error ? ` The result bundle is not present yet (${error}).` : ""}</p></section>}
  </main>;
}
