import type { Metadata } from "next";
import trainingResults from "../../public/data/introspection-training-results.json";
import { IntrospectionTrainingDashboard } from "../IntrospectionTrainingDashboard";

export const metadata: Metadata = {
  title: "Introspection Fine-tuning · Hidden Marker Lab",
  description: "Multi-seed Qwen3-14B fine-tuning and counterfactual evaluation for in-context matching of arbitrary residual-stream interventions.",
  openGraph: {
    title: "Introspection Fine-tuning · Hidden Marker Lab",
    description: "Multi-seed training, controls, and a complete trial audit for arbitrary hidden-marker matching.",
    images: [],
  },
  twitter: {
    title: "Introspection Fine-tuning · Hidden Marker Lab",
    description: "Multi-seed training, controls, and a complete trial audit for arbitrary hidden-marker matching.",
    images: [],
  },
};

export default function IntrospectionTrainingPage() {
  return <IntrospectionTrainingDashboard results={trainingResults} />;
}
