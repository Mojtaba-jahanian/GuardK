# Reproducibility and source provenance

## Recovered original source

The G10B archive exported on 20 October 2026 contains the actual `src/guardk` tree, experimental scripts and per-run G9 records. The recovered `core.py` and `core_v04.py` match the SHA256 values preserved in `manifests/G9_CURRENT_V04_SHA256.txt`; the v0.4 protocol-lock file also matches. No algorithm source was reconstructed or retuned for this release. The original development headers/version string are retained verbatim even where the code was subsequently frozen for G9.

## Included
- Unmodified source modules and original G6--G9 experiment scripts.
- G9 synthetic generator embedded in its original runner.
- Original per-run records for the 288-case, six-method confirmatory grid.
- G7/G8 records and selected comparator records.
- Frozen source/protocol hash records and an archived environment snapshot.
- Figure reproduction and summary verification tools.

## Verified in preparing this release
Source hashes and aggregate/per-run consistency were checked. The archived original deterministic smoke test was run separately from the confirmatory evaluation. Full G9, modern-comparator and G11 experiments were not rerun to prepare this repository.

## Remaining limits
The available environment file is the archived G2 snapshot, not a newly verified exact lock for every later stage or the isolated DeepDPM environment. Complete G11/G11B per-run outputs and sample-level predictions for every stage are not included. Third-party comparator source is not vendored; follow original repository commits, environment and license records. Raw patient-level data are not redistributed. A successful smoke test does not prove bitwise reproduction of every published result under a different runtime.

## Original launchers
Files under `experiments/original` retain their historical output roots. Do not run them blindly into an existing experiment directory. The G9 wrapper relocates only the output root to a new directory and leaves all scientific settings unchanged.
