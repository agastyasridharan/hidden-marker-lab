export function ExperimentPrimer({
  question,
  idea,
  fixed,
  changed,
  measured,
  answer,
}: {
  question: string;
  idea: string;
  fixed: string;
  changed: string;
  measured: string;
  answer: string;
}) {
  return (
    <section className="experiment-primer">
      <div className="primer-intro">
        <span className="eyebrow">Before reading the graphs</span>
        <h2>{question}</h2>
        <p>{idea}</p>
      </div>
      <div className="primer-grid">
        <article><span>Held fixed</span><p>{fixed}</p></article>
        <article><span>Changed on purpose</span><p>{changed}</p></article>
        <article><span>What we measure</span><p>{measured}</p></article>
        <article className="primer-answer"><span>Answer from this experiment</span><p>{answer}</p></article>
      </div>
    </section>
  );
}

export function ReadingKey({ children }: { children: React.ReactNode }) {
  return <div className="reading-key"><span>How to read this figure</span><p>{children}</p></div>;
}
