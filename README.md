# Hidden Marker Lab

[Open the research dashboard](https://agastyasridharan.github.io/hidden-marker-lab/)

GitHub Pages edition of https://hidden-marker-lab.therealgasty.chatgpt.site/.
The original React dashboard components and CSS are preserved. The build renders
all eight pages to static HTML and hydrates their interactive controls. Only
site-local links and data requests are prefixed for the GitHub Pages subdirectory.
All downloadable evidence files are copied without modification.

## Build

Use Node.js 24, then run `npm ci` and `npm run build`. The `dist` directory is a
complete static site. Pushes to `main` publish it through GitHub Actions.

## Research

Qwen3-14B learns to associate arbitrary injected residual-stream vectors with
labels supplied in context. Base-model accuracy was 50%; three fine-tuned
adapters averaged 93.25% on held-out matching trials. Generalization to new
vectors was strong, but transfer across injection locations and to broader
self-knowledge tasks was limited. This establishes a narrow learned matching
ability, not general or subjective introspection.

See the dashboard for all seven experiments and the downloadable replication
notebooks and source bundles. `RESEARCH.md` preserves the original project's
research notes; its experiment source paths refer to that original project and
the downloadable source bundle.
