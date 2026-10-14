import type { Metadata } from "next";
import experimentResults from "../../public/data/experiment-results.json";
import { ExperimentDashboard } from "../ExperimentDashboard";

export const metadata: Metadata = {
  title: "Experiment 1: Untrained Model · Hidden Marker Lab",
  description: "Can an untrained Qwen3-14B distinguish two arbitrary residual-stream interventions from in-context examples?",
  openGraph: { title: "Experiment 1: Untrained Model · Hidden Marker Lab", description: "The complete 128-trial untrained-model hidden-marker experiment.", images: [] },
  twitter: { title: "Experiment 1: Untrained Model · Hidden Marker Lab", description: "The complete 128-trial untrained-model hidden-marker experiment.", images: [] },
};

export default function LabelChoicePage() {
  return <ExperimentDashboard results={experimentResults} />;
}
