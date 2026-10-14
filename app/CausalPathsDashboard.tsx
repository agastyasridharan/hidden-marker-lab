"use client";

import { useMemo, useState } from "react";
import { ResearchHeader } from "./ResearchHeader";

type Effect = {
  n_trials: number;
  baseline_accuracy: number;
  changed_accuracy: number;
  accuracy_drop: number;
  accuracy_drop_ci_95_hierarchical: [number, number];
  mean_margin_drop: number;
  margin_drop_ci_95_hierarchical: [number, number];
  answer_flip_fraction: number;
};

type Restoration = {
  n_trials: number;
  full_accuracy: number;
  clean_accuracy: number;
  restored_accuracy: number;
  mean_marker_margin_effect: number;
  mean_restored_margin_effect: number;
  aggregate_margin_recovery_fraction: number;
};

type GeometryFamily = {
  n_trials: number;
  accuracy: number;
  accuracy_ci_95_hierarchical: [number, number];
  mean_margin: number;
  mean_head_22_17_match_contrast?: number;
  mean_projected_cosine?: number;
  mean_projected_distance?: number;
  max_abs_full_pair_cosine?: number;
};

type Results = {
  status: string;
  model_id: string;
  n_adapters: number;
  episodes_per_adapter: number;
  total_primary_trials: number;
  protocol: { injection: string; candidate_origin: string; confirmation: string; stage_order: string[] };
  transport: {
    conditions: Record<string, Effect | Restoration>;
    heldout_followup: { removal: Record<"k" | "v" | "k+v", Effect>; restoration: Record<"k" | "v" | "k+v", Restoration> };
  };
  binding: { conditions: Record<string, Effect> };
  geometry: { families: Record<string, GeometryFamily>; rowspace: Array<{ matrix_shape: [number, number]; numerical_rank: number }> };
  relays: { conditions: Record<string, Effect> };
  audit: { path: string; transport_rows: number; transport_followup_rows: number; binding_rows: number; geometry_behavior_rows: number; relay_rows: number };
  interpretation_limit: string;
  provenance: Array<{ model_condition: string; adapter_sha256: string; evaluation_seed: number; gpu: string }>;
};

type AuditRow = Record<string, unknown>;
type AuditData = { stage_trials: Record<string, Record<string, unknown>> };

const pct = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const pp = (value: number, digits = 1) => `${(value * 100).toFixed(digits)} pp`;
const num = (value: unknown, digits = 3) => typeof value === "number" ? value.toFixed(digits) : "not recorded";

function Metric({ label, value, detail, tone = "green" }: { label: string; value: string; detail: string; tone?: "green" | "ochre" | "red" }) {
  return <article className={`cp-metric cp-${tone}`}><span>{label}</span><strong>{value}</strong><p>{detail}</p></article>;
}

function AccuracyRows({ rows, chance = true }: { rows: Array<{ label: string; note: string; value: number; low?: number; high?: number; tone?: string }>; chance?: boolean }) {
  return <figure className="cp-accuracy-figure">
    <div className="cp-axis"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
    {rows.map((row) => <div className="cp-accuracy-row" key={row.label}>
      <div><strong>{row.label}</strong><small>{row.note}</small></div>
      <div className="cp-accuracy-track">{chance && <i className="cp-chance" />}<b className={row.tone ?? ""} style={{ width: `${Math.max(0, Math.min(1, row.value)) * 100}%` }} />{row.low !== undefined && row.high !== undefined && <em style={{ left: `${row.low * 100}%`, width: `${(row.high - row.low) * 100}%` }} />}</div>
      <output>{pct(row.value)}</output>
    </div>)}
  </figure>;
}

function EvidenceTable({ rows }: { rows: Array<{ intervention: string; result: string; reading: string; tone?: string }> }) {
  return <div className="cp-evidence-table">
    <div className="cp-evidence-head"><span>Intervention</span><span>Measured result</span><span>What it tells us</span></div>
    {rows.map((row) => <div key={row.intervention}><strong>{row.intervention}</strong><output className={row.tone ?? ""}>{row.result}</output><p>{row.reading}</p></div>)}
  </div>;
}

function auditConditions(audit: AuditData, stage: string): Record<string, AuditRow[]> {
  const raw = audit.stage_trials[stage] ?? {};
  if (stage === "geometry") {
    return Object.fromEntries(Object.entries(raw).map(([key, value]) => [key, ((value as { behavior?: AuditRow[] }).behavior ?? [])]));
  }
  if (stage === "transport_followup") {
    const nested = raw as Record<string, Record<string, AuditRow[]>>;
    return Object.fromEntries(Object.entries(nested).flatMap(([operation, values]) => Object.entries(values).map(([component, rows]) => [`${operation}/${component}`, rows])));
  }
  return raw as Record<string, AuditRow[]>;
}

function rowText(row: AuditRow, key: string) {
  const value = row[key];
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "boolean") return value ? "yes" : "no";
  return value === undefined || value === null ? "not recorded" : String(value);
}

function AuditLedger({ results }: { results: Results }) {
  const [audit, setAudit] = useState<AuditData | null>(null);
  const [loadError, setLoadError] = useState("");
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState("transport_followup");
  const [condition, setCondition] = useState("removal/v");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<AuditRow | null>(null);
  const [page, setPage] = useState(0);

  const conditions = useMemo(() => audit ? auditConditions(audit, stage) : {}, [audit, stage]);
  const conditionNames = Object.keys(conditions);
  const activeCondition = conditionNames.includes(condition) ? condition : (conditionNames[0] ?? "");
  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    const rows = conditions[activeCondition] ?? [];
    if (!query) return rows;
    return rows.filter((row) => ["model_condition", "episode_id", "test_vector", "operation", "trial_index"].some((key) => rowText(row, key).toLowerCase().includes(query)));
  }, [conditions, activeCondition, search]);
  const pageSize = 40;
  const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize));
  const safePage = Math.min(page, pageCount - 1);
  const shown = filtered.slice(safePage * pageSize, (safePage + 1) * pageSize);

  async function loadAudit() {
    setLoading(true);
    setLoadError("");
    try {
      const response = await fetch(results.audit.path);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setAudit(await response.json() as AuditData);
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "Unknown download error");
    } finally {
      setLoading(false);
    }
  }

  if (!audit) return <div className="cp-audit-loader">
    <div><span className="eyebrow">Complete trial ledger</span><h3>Load all 47,232 intervention records</h3><p>The full ledger is 37 MB, so it is not downloaded until you ask for it. It contains every adapter seed, episode, presentation transform, test vector, before-and-after margin, correctness result, and intervention specification.</p>{loadError && <p className="cp-error">The ledger could not be loaded: {loadError}</p>}</div>
    <div><button type="button" onClick={loadAudit} disabled={loading}>{loading ? "Loading…" : "Load trial ledger"}</button><a href={results.audit.path} download>Download JSON instead</a></div>
  </div>;

  return <div className="cp-audit-workspace">
    <div className="cp-ledger">
      <div className="cp-filters">
        <label>Study stage<select value={stage} onChange={(event) => { setStage(event.target.value); setCondition(""); setPage(0); setSelected(null); }}><option value="transport">1. Transport screen</option><option value="transport_followup">1. Held-out transport check</option><option value="binding">2. Binding</option><option value="geometry">3. Geometry</option><option value="relays">4. Relay paths</option></select></label>
        <label>Condition<select value={activeCondition} onChange={(event) => { setCondition(event.target.value); setPage(0); setSelected(null); }}>{conditionNames.map((name) => <option value={name} key={name}>{name}</option>)}</select></label>
        <label>Find trial<input value={search} onChange={(event) => { setSearch(event.target.value); setPage(0); }} placeholder="seed, episode, vector" /></label>
      </div>
      <div className="cp-ledger-count"><span>{filtered.length.toLocaleString()} records</span><span>page {safePage + 1} of {pageCount}</span></div>
      <div className="cp-table-shell"><table className="trial-table"><thead><tr><th>Adapter</th><th>Episode</th><th>Vector</th><th>Before</th><th>After</th><th>Change</th><th>Correct after</th></tr></thead><tbody>{shown.map((row, index) => <tr key={`${rowText(row, "model_condition")}-${rowText(row, "trial_index")}-${index}`} className={selected === row ? "selected-row" : ""} onClick={() => setSelected(row)}><td>{rowText(row, "model_condition").replace("seed-", "")}</td><td>{rowText(row, "episode_id")}</td><td>{rowText(row, "test_vector")}</td><td>{num(row.baseline_margin ?? row.margin)}</td><td>{num(row.changed_margin ?? row.margin)}</td><td>{num(row.margin_drop)}</td><td>{row.changed_correct === undefined ? rowText(row, "correct") : rowText(row, "changed_correct")}</td></tr>)}</tbody></table></div>
      <div className="cp-pagination"><button type="button" disabled={safePage === 0} onClick={() => setPage(Math.max(0, safePage - 1))}>Previous</button><button type="button" disabled={safePage >= pageCount - 1} onClick={() => setPage(Math.min(pageCount - 1, safePage + 1))}>Next</button></div>
    </div>
    <aside className="cp-detail">{selected ? <><span className="eyebrow">Selected intervention</span><h3>{rowText(selected, "operation")}</h3><dl>{Object.entries(selected).map(([key, value]) => <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{Array.isArray(value) ? value.join(", ") : typeof value === "boolean" ? (value ? "yes" : "no") : String(value)}</dd></div>)}</dl></> : <><span className="eyebrow">Trial detail</span><h3>Select a table row</h3><p>The full intervention record will appear here, including the counterbalancing flags and exact adapter hash.</p></>}</aside>
  </div>;
}

export function CausalPathsDashboard({ results: rawResults }: { results: unknown }) {
  const results = rawResults as Results;
  const transport = results.transport.heldout_followup;
  const binding = results.binding.conditions;
  const geometry = results.geometry.families;
  const relays = results.relays.conditions;
  const demoRestore0 = results.transport.conditions["restore:L22:k+v:H3,4:R0"] as Restoration;
  const demoRestore1 = results.transport.conditions["restore:L22:k+v:H3,4:R1"] as Restoration;
  const demoRestoreTest = results.transport.conditions["restore:L22:k+v:H3,4:R2"] as Restoration;

  return <main className="cp-page">
    <ResearchHeader active="paths" status="Four stages · complete" />

    <section className="cp-hero">
      <div><span className="kicker">Causal path analysis</span><h1>How does the hidden marker move through <em>Qwen’s computation?</em></h1><p className="lede">We intervened on keys, values, queries, attention edges, and carefully constructed vector families. The results identify a distributed sequence: a value channel carries the test marker, queries compare it with demonstration keys, and later tokens relay the result toward the answer.</p><p className="cp-hero-note">This is evidence for trained access to an injected activation. It is not evidence that the model can generally inspect its policies, reasoning, or subjective state.</p></div>
      <aside className="cp-hero-verdict"><span className="eyebrow">Best-supported account</span><div><b>carry</b><i>→</i><b>match</b><i>→</i><b>relay</b></div><p>The behavior is not localized to one “introspection head.” Different components perform distinguishable parts of the computation.</p><dl><div><dt>Adapters</dt><dd>3</dd></div><div><dt>Primary trials per condition</dt><dd>{results.total_primary_trials}</dd></div><div><dt>Fresh episodes</dt><dd>{results.n_adapters * results.episodes_per_adapter}</dd></div></dl></aside>
    </section>

    <section className="result-section cp-primer">
      <div className="section-heading"><div><span className="section-index">0</span><span className="eyebrow">What was changed</span><h2>A causal test replaces one internal signal while keeping the prompt fixed</h2></div><p>Every trial starts from the same hidden-marker task used in the fine-tuning study. The marker is injected before block 20 at the final period of each carrier sentence. We then replace a chosen internal quantity with its value from a matched control run and measure the change in answer accuracy and correct-choice logit margin.</p></div>
      <div className="cp-four-steps">{results.protocol.stage_order.map((stage, index) => <article key={stage}><span>{index + 1}</span><b>{stage}</b><p>{index === 0 ? "Separate the marker information stored in attention keys and values." : index === 1 ? "Swap test queries and demonstration keys to test same-marker matching." : index === 2 ? "Construct vectors that isolate or collide in the LoRA input rowspace." : "Block selected attention routes from source tokens to later tokens."}</p></article>)}</div>
      <div className="plain-note"><b>Reading the causal numbers:</b> an accuracy drop measures how many percentage points are lost after removal or replacement. Restoration starts with the marker removed, puts one internal signal back, and measures how much of the original logit-margin effect returns. Confidence intervals resample adapter seeds and episodes hierarchically.</div>
    </section>

    <section className="result-section cp-stage cp-stage-one">
      <div className="section-heading"><div><span className="section-index">1</span><span className="eyebrow">K/V causal transport</span><h2>At block 20, head 5 carries test-marker information in its value vector</h2></div><p>After the broad K/V screen identified block 20, K/V head 5 at the test period, we froze a component-specific follow-up and evaluated it on new episodes. Removing only its value projection damaged performance. Removing only its key projection did not.</p></div>
      <div className="cp-metric-grid"><Metric label="Remove value" value={pp(transport.removal.v.accuracy_drop)} detail={`95% CI ${pp(transport.removal.v.accuracy_drop_ci_95_hierarchical[0])} to ${pp(transport.removal.v.accuracy_drop_ci_95_hierarchical[1])}`} /><Metric label="Remove key" value={pp(transport.removal.k.accuracy_drop)} detail="95% CI includes zero" tone="ochre" /><Metric label="Restore value" value={pct(transport.restoration.v.aggregate_margin_recovery_fraction)} detail={`accuracy rises from ${pct(transport.restoration.v.clean_accuracy)} to ${pct(transport.restoration.v.restored_accuracy)}`} /><Metric label="Restore key" value={pct(transport.restoration.k.aggregate_margin_recovery_fraction)} detail="essentially no margin recovery" tone="ochre" /></div>
      <div className="cp-split-grid">
        <article className="paper-panel cp-explanation"><span className="eyebrow">What “value carries the marker” means</span><h3>Later token positions can read a marker-dependent value from the period token</h3><p>In attention, a key helps determine whether a source position is selected. A value contains the information copied from that source. Here, restoring the marker-dependent value at block 20 head 5 recovers {pct(transport.restoration.v.aggregate_margin_recovery_fraction)} of the original margin effect. Restoring its key recovers only {pct(transport.restoration.k.aggregate_margin_recovery_fraction)}.</p><p>This result was replicated across all three adapters. The confidence interval on the value-removal accuracy drop was {pp(transport.removal.v.accuracy_drop_ci_95_hierarchical[0])} to {pp(transport.removal.v.accuracy_drop_ci_95_hierarchical[1])}.</p></article>
        <article className="paper-panel cp-explanation"><span className="eyebrow">A different role at block 22</span><h3>Demonstration markers are available as keys</h3><p>At block 22, removing head 3’s key at the second demonstration costs {pp((results.transport.conditions["remove:L22:k:H3:R1"] as Effect).accuracy_drop)}. Removing its value costs {pp((results.transport.conditions["remove:L22:v:H3:R1"] as Effect).accuracy_drop)}.</p><p>Restoring both candidate K/V heads at the first and second demonstration recovers {pct(demoRestore0.aggregate_margin_recovery_fraction)} and {pct(demoRestore1.aggregate_margin_recovery_fraction)} of the respective margin effects. The same restoration at the test period recovers only {pct(demoRestoreTest.aggregate_margin_recovery_fraction)}. This role difference motivated the query-key tests in Stage 2.</p></article>
      </div>
    </section>

    <section className="result-section cp-stage cp-stage-two">
      <div className="section-heading"><div><span className="section-index">2</span><span className="eyebrow">Binding and matching</span><h2>Test-period queries are compared with demonstration keys</h2></div><p>We replaced the test-period query with the query produced by the opposite marker, or replaced the demonstration keys with keys from counterfactual markers. Each replacement breaks the match. Replacing both sides with compatible counterfactuals restores most of the behavior.</p></div>
      <AccuracyRows rows={[
        { label: "Unchanged run", note: "reference", value: binding.test_q_matcher_group.baseline_accuracy, tone: "cp-bar-reference" },
        { label: "Swap test queries", note: "six-head matcher group", value: binding.test_q_matcher_group.changed_accuracy, low: 1 - binding.test_q_matcher_group.accuracy_drop_ci_95_hierarchical[1], high: 1 - binding.test_q_matcher_group.accuracy_drop_ci_95_hierarchical[0], tone: "cp-bar-loss" },
        { label: "Swap demonstration keys", note: "matched K/V heads", value: binding.demo_k_matcher_kv.changed_accuracy, low: 1 - binding.demo_k_matcher_kv.accuracy_drop_ci_95_hierarchical[1], high: 1 - binding.demo_k_matcher_kv.accuracy_drop_ci_95_hierarchical[0], tone: "cp-bar-loss" },
        { label: "Swap demonstration values", note: "component control", value: binding.demo_v_matcher_kv.changed_accuracy, tone: "cp-bar-control" },
        { label: "Swap queries and keys together", note: "compatible counterfactual", value: binding.combined_q_kv_counterfactual.changed_accuracy, low: 1 - binding.combined_q_kv_counterfactual.accuracy_drop_ci_95_hierarchical[1], high: 1 - binding.combined_q_kv_counterfactual.accuracy_drop_ci_95_hierarchical[0], tone: "cp-bar-rescue" },
      ]} />
      <div className="cp-binding-note"><strong>The interaction is the key result.</strong><p>Query replacement alone leaves {pct(binding.test_q_matcher_group.changed_accuracy)} accuracy, and demonstration-key replacement alone leaves {pct(binding.demo_k_matcher_kv.changed_accuracy)}. Making both counterfactual changes together raises accuracy to {pct(binding.combined_q_kv_counterfactual.changed_accuracy)}. This is difficult to explain as independent damage. It is what we expect if the query and keys must agree.</p></div>
      <EvidenceTable rows={[
        { intervention: "Demonstration values", result: pp(binding.demo_v_matcher_kv.accuracy_drop), reading: "The marker-specific value at these K/V heads is not needed for the comparison.", tone: "cp-null" },
        { intervention: "Matcher output from label tokens", result: pp(binding.demo_label_z_matcher_group.accuracy_drop), reading: "No evidence that these heads directly bind a marker to the written nonce label.", tone: "cp-null" },
        { intervention: "Label-token residual at block 22", result: pp(binding.demo_label_residual_pre_22.accuracy_drop), reading: "The proposed label-token storage site was not causal at this layer.", tone: "cp-null" },
      ]} />
    </section>

    <section className="result-section cp-stage cp-stage-three">
      <div className="section-heading"><div><span className="section-index">3</span><span className="eyebrow">Low-rank fingerprint geometry</span><h2>The model is not simply reading coordinates in one LoRA input rowspace</h2></div><p>The block 20 attention LoRA updates have a 48-dimensional combined input rowspace inside the 5,120-dimensional residual stream. We constructed marker pairs inside it, perpendicular to it, and with nearly identical projections into it.</p></div>
      <AccuracyRows rows={[
        { label: "Fresh random pair", note: "ordinary evaluation distribution", value: geometry.random.accuracy, low: geometry.random.accuracy_ci_95_hierarchical[0], high: geometry.random.accuracy_ci_95_hierarchical[1] },
        { label: "Inside LoRA rowspace", note: "all marker energy lies in 48 dimensions", value: geometry.rowspace.accuracy, low: geometry.rowspace.accuracy_ci_95_hierarchical[0], high: geometry.rowspace.accuracy_ci_95_hierarchical[1] },
        { label: "Perpendicular to rowspace", note: `projected distance ${geometry.nullspace.mean_projected_distance?.toExponential(2)}`, value: geometry.nullspace.accuracy, low: geometry.nullspace.accuracy_ci_95_hierarchical[0], high: geometry.nullspace.accuracy_ci_95_hierarchical[1] },
        { label: "Same rowspace projection", note: `projected cosine ${geometry.collision.mean_projected_cosine?.toFixed(6)}`, value: geometry.collision.accuracy, low: geometry.collision.accuracy_ci_95_hierarchical[0], high: geometry.collision.accuracy_ci_95_hierarchical[1] },
        { label: "Opposite marker at test", note: "sign-sensitive control", value: geometry.anti_match.accuracy, low: geometry.anti_match.accuracy_ci_95_hierarchical[0], high: geometry.anti_match.accuracy_ci_95_hierarchical[1], tone: "cp-bar-loss" },
      ]} />
      <div className="cp-geometry-cards"><article><b>Nullspace pair</b><strong>{pct(geometry.nullspace.accuracy)}</strong><p>The pair is numerically invisible to the selected LoRA input rowspace, yet the task still works.</p></article><article><b>Collision pair</b><strong>{pct(geometry.collision.accuracy)}</strong><p>The two full vectors are orthogonal, while their rowspace projections are nearly identical. The model still distinguishes them.</p></article><article><b>Match-head contrast</b><strong>{geometry.collision.mean_head_22_17_match_contrast?.toFixed(3)}</strong><p>The same-vector attention signature remains present for the collision construction.</p></article></div>
      <div className="interpretation-note"><span>Conclusion from Stage 3</span><p>These results reject the narrow hypothesis that the marker fingerprint is just its coordinates in the block 20 Q/K/V LoRA-A rowspace. They do not identify the alternative representation. Base-model projections, nonlinear transformations, other LoRA inputs, or a distributed combination remain possible.</p></div>
    </section>

    <section className="result-section cp-stage cp-stage-four">
      <div className="section-heading"><div><span className="section-index">4</span><span className="eyebrow">Direct and relay paths</span><h2>Later tokens relay the test-period result to the answer</h2></div><p>We blocked attention edges from the marker-bearing period tokens. A block affects only the specified source-to-destination routes. This lets us distinguish a direct period-to-answer path from information that first moves through intermediate tokens.</p></div>
      <AccuracyRows rows={[
        { label: "Unchanged run", note: "reference", value: relays.test_source_blackout_all.baseline_accuracy },
        { label: "Block direct test-period to answer edge", note: "layers 20 to 39", value: relays.answer_to_test_period.changed_accuracy },
        { label: "Block all test-period outputs, layers 20 to 22", note: "early source blackout", value: relays.test_source_blackout_20_22.changed_accuracy },
        { label: "Block all test-period outputs, layers 23 to 39", note: "late source blackout", value: relays.test_source_blackout_23_39.changed_accuracy, low: 1 - relays.test_source_blackout_23_39.accuracy_drop_ci_95_hierarchical[1], high: 1 - relays.test_source_blackout_23_39.accuracy_drop_ci_95_hierarchical[0], tone: "cp-bar-loss" },
        { label: "Block all test-period outputs", note: "every layer", value: relays.test_source_blackout_all.changed_accuracy, low: 1 - relays.test_source_blackout_all.accuracy_drop_ci_95_hierarchical[1], high: 1 - relays.test_source_blackout_all.accuracy_drop_ci_95_hierarchical[0], tone: "cp-bar-loss" },
        { label: "Block neighboring control positions", note: "matched spatial control", value: relays.all_period_control_blackout.changed_accuracy, tone: "cp-bar-control" },
      ]} />
      <div className="cp-relay-explanation"><article><span>Direct edge</span><strong>{pp(relays.answer_to_test_period.accuracy_drop)}</strong><p>Removing the direct route into the final answer token lowers confidence by {relays.answer_to_test_period.mean_margin_drop.toFixed(2)} logits, but barely changes accuracy.</p></article><article><span>Late source blackout</span><strong>{pp(relays.test_source_blackout_23_39.accuracy_drop)}</strong><p>Preventing every later token from reading the test period after block 22 reduces performance to {pct(relays.test_source_blackout_23_39.changed_accuracy)}.</p></article><article><span>All-layer source blackout</span><strong>{pct(relays.test_source_blackout_all.changed_accuracy)}</strong><p>Accuracy reaches chance. Neighboring source positions produce no comparable loss.</p></article></div>
    </section>

    <section className="result-section cp-model-section">
      <div className="section-heading"><div><span className="section-index">5</span><span className="eyebrow">Integrated causal model</span><h2>The evidence supports a sequence, not a single introspection component</h2></div><p>The diagram states the narrowest account supported by all four stages. Solid arrows have direct causal evidence. The exact representation between the identified components remains unresolved.</p></div>
      <div className="cp-circuit">
        <article><span>Injected state</span><b>Period token before block 20</b><p>The residual stream receives one arbitrary unit vector.</p></article><i>→</i><article className="cp-circuit-primary"><span>Carry</span><b>Block 20 · K/V head 5 value</b><p>A marker-dependent value becomes available to later token positions.</p></article><i>→</i><article className="cp-circuit-primary"><span>Match</span><b>Block 22 · query/key circuit</b><p>The test-period query is compared with keys at the two demonstration periods.</p></article><i>→</i><article><span>Relay</span><b>Layers 23 to 39</b><p>Intermediate tokens carry the comparison result toward the answer.</p></article>
      </div>
      <div className="cp-claims"><article><span>Supported</span><p>Specific values, queries, keys, and later source routes are causally necessary or sufficient for substantial parts of the trained behavior.</p></article><article><span>Not yet localized</span><p>The marker-to-label binding itself, the exact feature basis used by the comparator, and the complete downstream route remain unresolved.</p></article><article><span>Not established</span><p>{results.interpretation_limit}</p></article></div>
    </section>

    <section className="audit-section cp-audit-section">
      <div className="section-heading"><div><span className="section-index">6</span><span className="eyebrow">Audit the evidence</span><h2>Inspect every causal intervention</h2></div><p>The summaries above pool three independently trained adapters. The ledger exposes every underlying record, including failures and null controls. Loading is manual because the complete audit file is large.</p></div>
      <AuditLedger results={results} />
    </section>

    <section className="method-section cp-method-section">
      <div className="section-heading"><div><span className="section-index">7</span><span className="eyebrow">Protocol and provenance</span><h2>What was fixed, repeated, and reported</h2></div><p>The primary study used the candidate heads fixed by the earlier Experiment 4 discovery split, then evaluated a new episode seed. The block 20 component split was a result-blind follow-up on another new episode seed.</p></div>
      <div className="cp-method-grid"><article><h3>Evaluation unit</h3><dl><div><dt>Model</dt><dd>{results.model_id}</dd></div><div><dt>Adapters</dt><dd>{results.n_adapters} independent LoRA training seeds</dd></div><div><dt>Primary episodes</dt><dd>{results.n_adapters * results.episodes_per_adapter}</dd></div><div><dt>Trials per condition</dt><dd>{results.total_primary_trials}</dd></div><div><dt>Counterbalancing</dt><dd>mapping, demonstration order, option order, both test vectors</dd></div></dl></article><article><h3>Intervention site</h3><dl><div><dt>Marker injection</dt><dd>block 20 residual input</dd></div><div><dt>Token</dt><dd>final period of each carrier sentence</dd></div><div><dt>Strength</dt><dd>ρ = 0.1 of median residual norm</dd></div><div><dt>Precision</dt><dd>BF16 model, FP32 direction sampling</dd></div><div><dt>Generation</dt><dd>single causal forward pass, no sampling</dd></div></dl></article><article><h3>Adapter provenance</h3><dl>{results.provenance.map((row) => <div key={row.model_condition}><dt>{row.model_condition}</dt><dd title={row.adapter_sha256}>{row.adapter_sha256.slice(0, 12)}… · eval {row.evaluation_seed}</dd></div>)}</dl><p>All runs report an NVIDIA H100 NVL. Full hashes and software versions are retained in the result JSON and trial ledger.</p></article></div>
      <div className="artifact-links cp-downloads"><a href="/data/kv-path-study-results.json" download><span>Summary JSON</span><b>Aggregated effects, intervals, and provenance</b></a><a href={results.audit.path} download><span>Complete audit JSON</span><b>47,232 trial-level intervention records</b></a></div>
    </section>

    <footer><div><span className="wordmark-mark">H</span><p><b>Hidden Marker Lab</b><br />Causal path study</p></div><p>Qwen3-14B · 3 adapters · 4 ordered stages · complete trial audit</p></footer>
  </main>;
}
