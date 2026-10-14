import type { Metadata } from "next";
import binaryResults from "../../public/data/binary-dax-results.json";
import { BinaryDaxDashboard } from "../BinaryDaxDashboard";

export const metadata: Metadata = {
  title: "Binary Dax Detection · Hidden Marker Lab",
  description: "Before-and-after YES/NO logit audit for 128 binary hidden-marker intervention trials on Qwen3-14B.",
  openGraph: {
    title: "Binary Dax Detection · Hidden Marker Lab",
    description: "Before-and-after YES/NO logit audit for 128 binary hidden-marker intervention trials on Qwen3-14B.",
    images: [],
  },
  twitter: {
    title: "Binary Dax Detection · Hidden Marker Lab",
    description: "Before-and-after YES/NO logit audit for 128 binary hidden-marker intervention trials on Qwen3-14B.",
    images: [],
  },
};

export default function BinaryDaxPage() {
  return <BinaryDaxDashboard results={binaryResults} />;
}
