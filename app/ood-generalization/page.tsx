import type { Metadata } from "next";
import experiment4Results from "../../public/data/experiment4-results.json";
import { OodGeneralizationDashboard } from "../OodGeneralizationDashboard";

export const metadata: Metadata = {
  title: "OOD Generalization & Circuit Analysis · Hidden Marker Lab",
  description: "Disjoint-span generalization, OOD stress tests, and causal localization of a hidden-marker matching circuit in fine-tuned Qwen3-14B.",
  openGraph: {
    title: "OOD Generalization & Circuit Analysis · Hidden Marker Lab",
    description: "Three disjoint-span training runs, complete OOD ledgers, and an executed mechanistic-analysis notebook.",
    images: [],
  },
  twitter: {
    title: "OOD Generalization & Circuit Analysis · Hidden Marker Lab",
    description: "Disjoint-span generalization and causal hidden-marker circuit analysis.",
    images: [],
  },
};

export default function OodGeneralizationPage() {
  return <OodGeneralizationDashboard results={experiment4Results} />;
}

