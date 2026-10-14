import type { Metadata } from "next";
import results from "../../public/data/introspectbench-results.json";
import { IntrospectBenchDashboard } from "../IntrospectBenchDashboard";

export const metadata: Metadata = {
  title: "IntrospectBench · Hidden Marker Lab",
  description: "A base-versus-LoRA evaluation of Qwen3-14B on four policy-introspection tasks, with cross-model comparisons and complete trial audits.",
  openGraph: {
    title: "IntrospectBench · Hidden Marker Lab",
    description: "K-th-word prediction, ethical calibration, paraphrase recognition, and Heads-Up for base and hidden-marker-trained Qwen3-14B.",
    images: [],
  },
  twitter: {
    title: "IntrospectBench · Hidden Marker Lab",
    description: "A complete, auditable base-versus-LoRA IntrospectBench run.",
    images: [],
  },
};

export default function IntrospectBenchPage() {
  return <IntrospectBenchDashboard results={results} />;
}
