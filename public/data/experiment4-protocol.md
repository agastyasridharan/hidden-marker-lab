# Experiment 4: out-of-distribution access to hidden markers

Protocol frozen before inspecting Experiment 4 outcomes on 2026-08-20.

## Claims and estimands

The existing three fixed-block adapters are evaluated without further tuning. The ID reference and each one-axis OOD condition use 64 fresh episodes, both test vectors, and two counterfactual transformations (256 trials per adapter-condition). The joint-OOD condition uses 128 episodes and four transformations (1,024 trials per adapter). Accuracy is averaged over the three independently trained adapters; Wilson intervals describe trial-level uncertainty and an episode-cluster bootstrap is used for primary comparisons.

The frozen-adapter suite changes:

1. prompt wording;
2. identical versus different demonstration/test carriers;
3. the injection token (period versus first content token, including a cross-position condition);
4. intervention layer (16, 20, or 24);
5. vector family (Gaussian-QR, Student-t(3)-QR, and orthogonalized natural residual-activation differences);
6. pair cosine (0, 0.5, or 0.9);
7. several shifts jointly.

An angle sweep injects `cos(theta) v1 + sin(theta) v2` at test for theta in 0, 15, ..., 90 degrees. Only the endpoints have categorical ground truth; intermediate points report the label-v2 minus label-v1 logit contrast and are not counted as accuracy trials.

## Disjoint-span arm

Three additional adapters are trained independently. All localization, same/different, label-binding, and validation directions lie in the training half of a fixed 5,120-dimensional orthonormal basis. The basis is implemented as a deterministic signed and permuted block Walsh-Hadamard transform; its two halves are exactly orthogonal and dense in the model's residual basis. The primary evaluation compares fresh vectors from the trained half with fresh vectors from the reserved complement, using held-out nonce labels and lexically disjoint carriers.

The strong extrapolation criterion is: each seed must exceed 75% on the reserved complement, and the across-seed mean holdout accuracy must be no more than 10 percentage points below its in-span accuracy. Failure is informative and must not be redescribed as successful arbitrary-direction generalization.

## Mechanistic study

Head discovery uses one adapter and a discovery episode split. Confirmation uses held-out episodes and the other independently trained adapters. Attention routing and adapter-weight norms are candidate-generation statistics only. A putative marker-matching head requires all of the following:

- held-out single-head ablation produces a substantial, reproducible task-specific accuracy or correct-margin loss;
- counterfactual activation patching changes the answer in the predicted direction;
- the effect survives mapping, option, label, carrier, and OOD-direction counterbalancing;
- the same layer/head, or a preregistered small circuit, recurs across independent training seeds;
- effects are larger than matched random-head controls and are not explained by general language-model degradation;
- query/key/value or head-composition analysis supports the proposed information-flow mechanism.

If these criteria fail, the result is reported as no reliable single-head localization. Distributed layer/MLP mediation, seed-specific implementations, and reuse of generic in-context routing heads remain alternative hypotheses.

