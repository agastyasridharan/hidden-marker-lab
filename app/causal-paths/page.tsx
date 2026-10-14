import type { Metadata } from "next";
import results from "../../public/data/kv-path-study-results.json";
import { CausalPathsDashboard } from "../CausalPathsDashboard";

export const metadata: Metadata = {
  title: "Causal Paths · Hidden Marker Lab",
  description: "Four causal intervention studies tracing how hidden-marker information is carried, matched, and relayed through fine-tuned Qwen3-14B.",
  openGraph: {
    title: "Causal Paths · Hidden Marker Lab",
    description: "Key, value, query, geometry, and attention-edge interventions with a complete trial audit.",
    images: [],
  },
  twitter: {
    title: "Causal Paths · Hidden Marker Lab",
    description: "A causal analysis of hidden-marker transport and matching.",
    images: [],
  },
};

export default function CausalPathsPage() {
  return <CausalPathsDashboard results={results} />;
}
