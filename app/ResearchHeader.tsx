/* eslint-disable @next/next/no-html-link-for-pages -- vinext experiment routes use full document navigation */

export type ResearchRoute = "overview" | "label" | "binary" | "training" | "ood" | "benchmark" | "paths" | "transfer";

const routes: Array<{ id: ResearchRoute; href: string; label: string }> = [
  { id: "overview", href: "/", label: "Start here" },
  { id: "label", href: "/label-choice", label: "1 · Base model" },
  { id: "binary", href: "/binary-dax", label: "2 · Binary check" },
  { id: "training", href: "/introspection-training", label: "3 · Fine-tuning" },
  { id: "ood", href: "/ood-generalization", label: "4 · Generalization" },
  { id: "benchmark", href: "/introspectbench", label: "5 · IntrospectBench" },
  { id: "paths", href: "/causal-paths", label: "6 · Causal paths" },
  { id: "transfer", href: "/beyond-markers", label: "7 · Beyond markers" },
];

export function ResearchHeader({ active, status }: { active: ResearchRoute; status?: string }) {
  return (
    <header className="site-header research-header">
      <a className="wordmark" href="/" aria-label="Hidden Marker Lab overview">
        <span className="wordmark-mark">H</span>
        <span>Hidden Marker Lab<small>A guided research report</small></span>
      </a>
      <nav aria-label="Study pages">
        {routes.map((route) => (
          <a
            key={route.id}
            className={route.id === active ? "active-experiment" : ""}
            href={route.href}
            aria-current={route.id === active ? "page" : undefined}
          >
            {route.label}
          </a>
        ))}
      </nav>
      <span className="run-status"><i />{status ?? "All runs complete"}</span>
    </header>
  );
}
