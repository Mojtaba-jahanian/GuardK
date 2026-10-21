# Frozen-method specification (documentation, not an implementation)

The base selector standardizes features and constructs candidates by sequential prototype splitting. For each candidate, the documented score is:

G_k = 0.55 S_k + 0.20 H_tilde_k + 0.15 R_k + 0.10 O_k - 0.02 k/K_max.

Eligibility requires stability >= 0.55 and minimum cluster prevalence >= 0.003. Selection chooses the smallest eligible K within 0.008 of the best score. The fallback, when no candidate is eligible, is documented in the manuscript.

The outer layer uses five 80% subsamples. Acceptance requires consistency >= 0.60 and the modal subsample K to equal the original selected K. Score ambiguity (margin <= 0.008) or overlap ambiguity (selected overlap safety < 0.45) requires strong reproducibility (consistency >= 0.80 and matching mode). The guard returns the original full-data partition or abstains; it does not substitute the subsample modal partition.

Definitions, deterministic seeds, fallback behavior, candidate initialization and numerical settings must be taken from the original experimental implementation before this documentation is used to reproduce experiments.
