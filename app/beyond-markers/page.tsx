import type { Metadata } from "next";
import beyondMarkersResults from "../../public/data/beyond-markers-results.json";
import { BeyondMarkersDashboard, type Results } from "../BeyondMarkersDashboard";

export const metadata: Metadata = {
  title: "Beyond hidden markers · Hidden Marker Lab",
  description: "A base-versus-LoRA transfer evaluation using SAE feature naming, input ablation, the Jacobian lens, injected-concept reporting, and EvalAwareBench.",
  openGraph: {
    title: "Beyond hidden markers · Hidden Marker Lab",
    description: "Five tests of whether hidden-marker training transfers to broader forms of self-knowledge.",
    images: [],
  },
  twitter: {
    title: "Beyond hidden markers · Hidden Marker Lab",
    description: "SAE features, self-prediction, latent computations, and evaluation awareness.",
    images: [],
  },
};

export default function BeyondMarkersPage() {
  return <BeyondMarkersDashboard initialResults={beyondMarkersResults as Results} />;
}
