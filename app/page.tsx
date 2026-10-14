import experimentResults from "../public/data/experiment-results.json";
import binaryResults from "../public/data/binary-dax-results.json";
import trainingResults from "../public/data/introspection-training-results.json";
import oodResults from "../public/data/experiment4-results.json";
import introspectBenchResults from "../public/data/introspectbench-results.json";
import pathResults from "../public/data/kv-path-study-results.json";
import beyondResults from "../public/data/beyond-markers-results.json";
import { StudyOverview } from "./StudyOverview";

export default function Home() {
  return <StudyOverview base={experimentResults} binary={binaryResults} training={trainingResults} ood={oodResults} benchmark={introspectBenchResults} paths={pathResults} beyond={beyondResults} />;
}
