# Auxiliary Tuning Measurement Design

## Goal

Allow a tuning-named dataset to run when it is a measurement of an existing mechanical state rather than a new tuning state.

## Design

Campaign YAML gains an optional `aux` mapping.  Each key is a dataset ID and each value names its parent state plus whether automatic simulation comparison applies.  State entries remain the authoritative mechanical sequence.

For `raw_sweep_260721_tune_s003_plungersensitivity`, the parent is `s003` and `cmp: false`.  The runner writes the ordinary analysis and records the campaign/state in the manifest, but skips the `r_c` comparison because plunger sensitivity is not a geometry-state comparison.

## Validation

An auxiliary dataset must reference an existing state, cannot duplicate a completed state's primary dataset, and is checked under `data/raw` when a data root is supplied.
