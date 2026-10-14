import { ResearchHeader } from "./ResearchHeader";

type BaseResults = { aggregate: { accuracy: number; n_trials: number } };
type BinaryResults = { aggregate: { after: { accuracy: number }; n_trials: number } };
type TrainingResults = { aggregate: { base_accuracy: number; fixed_seed_mean_accuracy: number; paired_improvement: { cluster_bootstrap_ci_95: [number, number] } } };
type OodCondition = { id: string; mean_accuracy: number };
type OodResults = { subspace: { mean_holdout_accuracy: number; mean_train_accuracy: number }; fixed_adapter_conditions: OodCondition[] };
type BenchResults = { comparisons: Array<{ target: string; observer: string; average_primary_score: number }> };
type PathResults = { binding: { conditions: { combined_q_kv_counterfactual: { changed_accuracy: number } } } };
type BeyondResults = { verbal_introspection: { curves: Array<{ variant: string; prefill: string; strength: number; n: number; slot_top_10_rate: number }> } };

const pct = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;

function OverviewBar({ label, value, note, tone = "good" }: { label: string; value: number; note: string; tone?: "good" | "mixed" | "null" }) {
  return (
    <div className="overview-bar-row">
      <div><strong>{label}</strong><span>{note}</span></div>
      <div className="overview-bar-track"><i className="chance-rule" /><b className={`overview-bar-${tone}`} style={{ width: `${value * 100}%` }} /></div>
      <output>{pct(value)}</output>
    </div>
  );
}

function Term({ word, children }: { word: string; children: React.ReactNode }) {
  return <div className="term-card"><dt>{word}</dt><dd>{children}</dd></div>;
}

export function StudyOverview({ base: rawBase, binary: rawBinary, training: rawTraining, ood: rawOod, benchmark: rawBenchmark, paths: rawPaths, beyond: rawBeyond }: { base: unknown; binary: unknown; training: unknown; ood: unknown; benchmark: unknown; paths: unknown; beyond: unknown }) {
  const base = rawBase as BaseResults;
  const binary = rawBinary as BinaryResults;
  const training = rawTraining as TrainingResults;
  const ood = rawOod as OodResults;
  const benchmark = rawBenchmark as BenchResults;
  const paths = rawPaths as PathResults;
  const beyond = rawBeyond as BeyondResults;
  const condition = (id: string) => ood.fixed_adapter_conditions.find((row) => row.id === id)!.mean_accuracy;
  const benchmarkBase = benchmark.comparisons.find((row) => row.target === "base" && row.observer === "base")!;
  const benchmarkTrained = benchmark.comparisons.find((row) => row.target === "trained_20260820" && row.observer === "trained_20260820")!;
  const verbalTrained = beyond.verbal_introspection.curves.find((row) => row.variant !== "base" && row.prefill === "word" && row.strength === 0.025)!;

  return <main className="overview-page">
    <ResearchHeader active="overview" status="Seven experiments complete" />

    <section className="overview-hero">
      <div>
        <span className="kicker">A guided report on hidden-state interventions</span>
        <h1>Can a language model notice a change made to its <em>own internal state?</em></h1>
        <p className="lede">We add a small vector to one internal activation while the model reads a sentence. The model must learn an arbitrary name for that vector from examples, then report which vector appeared again.</p>
        <p className="overview-answer"><b>Short answer:</b> the untrained model could not do this. Fine-tuning taught it to solve the task at the location used during training, and the learned rule transferred to new vectors. It did not reliably transfer to new token positions or layers.</p>
      </div>
      <aside className="overview-conclusion">
        <span className="eyebrow">Main empirical conclusion</span>
        <strong>Vectors: yes.<br />Locations: no.</strong>
        <p>The trained model handled entirely new intervention directions, including directions from an exactly orthogonal held-out subspace. Moving the intervention to a new place in the computation returned performance to chance.</p>
      </aside>
    </section>

    <section className="overview-section pale-section">
      <div className="prose-heading">
        <span className="section-index">A</span>
        <div><span className="eyebrow">The basic task</span><h2>What happens in one trial?</h2></div>
        <p>The words shown to the model are deliberately uninformative: the same sentence appears in both examples and in the test. Only the internal intervention changes.</p>
      </div>
      <div className="episode-walkthrough">
        <article><span>Example 1</span><p>The copper lantern stood beside the window.</p><div><i className="vector-chip v1-chip">v₁</i><b>internally added at the final period</b><strong>name: dax</strong></div></article>
        <article><span>Example 2</span><p>The copper lantern stood beside the window.</p><div><i className="vector-chip v2-chip">v₂</i><b>internally added at the final period</b><strong>name: wug</strong></div></article>
        <article className="test-episode"><span>Test</span><p>The copper lantern stood beside the window.</p><div><i className="vector-chip unknown-chip">v₁?</i><b>one vector appears again</b><strong>answer: dax or wug?</strong></div></article>
      </div>
      <div className="plain-note"><b>Important:</b> “dax” and “wug” have no fixed meaning. A fresh vector pair and fresh arbitrary labels are used in each fine-tuning episode. Solving the task requires comparing the intervention in the test sentence with the interventions in the examples.</div>
    </section>

    <section className="overview-section">
      <div className="prose-heading">
        <span className="section-index">B</span>
        <div><span className="eyebrow">The sequence of experiments</span><h2>Each experiment answers one narrower question</h2></div>
        <p>Read the study in order. Each page begins with the question, defines the measurement, shows the result, and states what the result does and does not establish.</p>
      </div>
      <div className="study-roadmap">
        <a href="/label-choice"><span>1</span><div><small>Base model</small><h3>Can Qwen3-14B solve the task without training?</h3><p>No. It answered “A” on every trial, producing 50% accuracy for the wrong reason.</p></div><strong>{pct(base.aggregate.accuracy)}</strong></a>
        <a href="/binary-dax"><span>2</span><div><small>Binary check</small><h3>Does a YES/NO question reveal weaker evidence?</h3><p>No vector-specific signal appeared. Both interventions moved the logits in nearly the same way.</p></div><strong>{pct(binary.aggregate.after.accuracy)}</strong></a>
        <a href="/introspection-training"><span>3</span><div><small>Fine-tuning</small><h3>Can online training teach the comparison rule?</h3><p>Yes at the trained site. Three independent adapters reached a mean of {pct(training.aggregate.fixed_seed_mean_accuracy)} on held-out episodes.</p></div><strong>{pct(training.aggregate.fixed_seed_mean_accuracy)}</strong></a>
        <a href="/ood-generalization"><span>4</span><div><small>Generalization and mechanism</small><h3>What transfers, and which components implement it?</h3><p>New directions transferred; new locations did not. A redundant attention circuit carries part of the comparison.</p></div><strong>{pct(ood.subspace.mean_holdout_accuracy)}</strong></a>
        <a href="/introspectbench"><span>5</span><div><small>Broader policy introspection</small><h3>Does the hidden-marker adapter transfer to IntrospectBench?</h3><p>We compare both self scores and cross-model observers on four published behavioral tasks, with every prompt and output available for audit.</p></div><strong>{pct(benchmarkTrained.average_primary_score)}</strong></a>
        <a href="/causal-paths"><span>6</span><div><small>Causal path analysis</small><h3>How does information about the marker move through the model?</h3><p>Value vectors carry the test marker, queries compare it with demonstration keys, and later tokens relay the result to the answer.</p></div><strong>{pct(paths.binding.conditions.combined_q_kv_counterfactual.changed_accuracy)}</strong></a>
        <a href="/beyond-markers"><span>7</span><div><small>Transfer beyond marker detection</small><h3>Does the adapter change broader forms of self-knowledge?</h3><p>No broad transfer was found across unseen SAE features, behavioral self-prediction, latent intermediate answers, injected concepts, or evaluation awareness.</p></div><strong>{Math.round(verbalTrained.slot_top_10_rate * verbalTrained.n)} / {verbalTrained.n}</strong></a>
      </div>
    </section>

    <section className="overview-section pale-section">
      <div className="prose-heading">
        <span className="section-index">C</span>
        <div><span className="eyebrow">Results at a glance</span><h2>The same 50% baseline can mean very different things</h2></div>
        <p>The vertical dashed rule is chance. Read each row together with its note: in Experiments 1 and 2, 50% came from a fixed response, not successful discrimination.</p>
      </div>
      <figure className="overview-figure">
        <div className="overview-axis"><span>0%</span><span>25%</span><span>50% chance</span><span>75%</span><span>100%</span></div>
        <OverviewBar label="Untrained A/B task" value={base.aggregate.accuracy} note="always chose A" tone="null" />
        <OverviewBar label="Untrained YES/NO task" value={binary.aggregate.after.accuracy} note="always chose NO" tone="null" />
        <OverviewBar label="Fine-tuned, trained location" value={training.aggregate.fixed_seed_mean_accuracy} note="mean across three seeds" />
        <OverviewBar label="Exactly orthogonal vector span" value={ood.subspace.mean_holdout_accuracy} note="never used during training" />
        <OverviewBar label="Different carrier sentences" value={condition("different_carriers")} note="language changed within episode" tone="mixed" />
        <OverviewBar label="Prompt paraphrase" value={condition("prompt_paraphrase")} note="instructions reworded" tone="mixed" />
        <OverviewBar label="Different token position" value={condition("cross_position")} note="test injection moved" tone="null" />
        <OverviewBar label="Different layer" value={condition("layer_24")} note="injection moved from block 20 to 24" tone="null" />
        <figcaption>Accuracy is averaged over the relevant model seeds. Intervals and complete per-seed results appear on the individual experiment pages.</figcaption>
      </figure>
    </section>

    <section className="overview-section">
      <div className="prose-heading">
        <span className="section-index">D</span>
        <div><span className="eyebrow">Vocabulary used throughout</span><h2>A small glossary</h2></div>
        <p>These terms are precise, but none needs to remain mysterious.</p>
      </div>
      <dl className="glossary-grid">
        <Term word="Residual stream">The main vector of information passed from one transformer block to the next. We modify this vector immediately before block 20.</Term>
        <Term word="Direction / vector">A list of 5,120 numbers with length one. Adding it nudges the model’s internal state in a controlled way.</Term>
        <Term word="Injection">The operation <code>hidden state + α × vector</code> at one chosen token and layer.</Term>
        <Term word="Logit">The model’s raw score for a possible next token. The candidate with the larger logit is the predicted answer.</Term>
        <Term word="LoRA adapter">A small set of trainable low-rank weight updates. The original Qwen weights remain frozen.</Term>
        <Term word="Held out / OOD">Data deliberately excluded from training and checkpoint choice, used to test whether a learned rule generalizes.</Term>
        <Term word="Attention head">One component that decides which earlier token positions to read from and what information to move.</Term>
        <Term word="Causal ablation">Disable a component and measure whether the target behavior deteriorates while unrelated language modeling remains intact.</Term>
      </dl>
    </section>

    <section className="overview-section notebook-callout">
      <div><span className="eyebrow">Reproduce the study</span><h2>Run every experiment from scratch</h2><p>The replication notebook explains the code, prompts, intervention hooks, training curriculum, controls, statistical summaries, and mechanistic tests in the order they are used.</p></div>
      <div><a href="/data/hidden-marker-lab-replication.html" target="_blank" rel="noreferrer">Read the rendered notebook</a><a href="/data/hidden-marker-lab-replication.ipynb" download>Download the executable notebook</a><a href="/data/hidden-marker-lab-replication-source.zip" download>Download the runner source bundle</a></div>
    </section>

    <footer><div><span className="wordmark-mark">H</span><p><b>Hidden Marker Lab</b><br />A guided report and complete audit</p></div><p>Qwen3-14B · seven studies · IntrospectBench base {pct(benchmarkBase.average_primary_score)}</p></footer>
  </main>;
}
