# Geometry Sweep Axis and Reference Design

## Goal

Process simulation datasets such as `sim_sweep_260728_zlen` with an explicit
geometry sweep axis and reference value, while preserving `NumDepth` as the
Cell/Iris observation-state coordinate.

## Data Contract

- The source remains under `data/sim`; no preprocessed Touchstone copy is
  created.
- `sim_L_c` is the coupler-cell z-length sweep axis in millimetres.
- `sim_NumDepth=1.5` identifies the Cell plane and `sim_NumDepth=2.0`
  identifies the Iris plane through the existing `tune_position` mapping.
- `sim_DepthPlunger` varies because the Cell and Iris states use different
  shorting-plunger positions. It is state metadata, not a second geometry
  sweep axis.
- The simulation dispersion markers remain `f_2pi3`, `f_mean`, and `f_pi2`;
  this workflow does not shift those marker frequencies.

## Interface

The one-folder CLI accepts two optional arguments:

```text
--geometry-sweep-axis sim_L_c
--geometry-sweep-base 29.148
```

The axis and base propagate through `run_folder_analysis()` and
`build_marker_analysis()` into `compute_geometry_phase_response()`. Supplying a
base without an axis is invalid. Existing calls that omit both arguments retain
their current automatic behavior.

## Analysis

For each marker and `sim_L_c`, the geometry response pairs the Cell 1.5 and
Iris 2.0 samples. It records:

- absolute Cell and Iris marker phase;
- wrapped phase pickup relative to `sim_L_c=29.148 mm`;
- wrapped Iris-minus-Cell phase difference and its change from the reference.

The Cell/Iris difference is a comparison of two shorting-reference states. It
must not be labelled as same-family periodic phase advance. The existing
`phase_adv` table may remain empty for this dataset.

## Outputs

The standard runner writes the established files:

- `tables/marker_pts.csv`;
- `tables/geometry_phase_response.csv`;
- `figures/geometry_phase_response/absolute_phase.png`;
- `figures/geometry_phase_response/phase_pickup.png`;
- marker-specific absolute and pickup plots.

The manifest records the requested axis and base under
`geometry_phase_response` so the reference choice is reproducible.

## Validation

- Reject an explicit axis that is missing, non-numeric, or constant.
- Reject a requested base that is not one of the sampled sweep values.
- Preserve the automatic path for existing geometry sweeps.
- Prove CLI parsing, pipeline propagation, manifest provenance, table content,
  and figure creation with focused tests.
- Run the real `sim_sweep_260728_zlen` analysis and verify 22 input files,
  66 marker points, 33 geometry-response rows, the 29.148 mm reference, and
  non-empty generated figures.

## Compatibility

No dataset-name branch is introduced. The feature is generic for any simulation
marker dataset with numeric `sim_*` geometry metadata and paired Cell/Iris
states.
