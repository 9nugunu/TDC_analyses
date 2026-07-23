# Position Phase Bar Figure Design

## Goal

Add a reproducible grouped bar figure that compares the measured wrapped
S-parameter phase at tuning positions 1.0 and 2.0 for the three frequency
markers `f_2pi3`, `f_mean`, and `f_pi2`.

## Data Contract

- Read the existing `marker_pts.csv` product rather than re-reading or
  converting the raw Touchstone files.
- Select exactly one row for every requested `(tune_position, marker_name)`
  pair.
- Preserve the signed wrapped phase in `s_phase_deg`, while deriving a separate
  `phase_0to360_deg` plotting column with modulo-360 arithmetic.
- Retain source identity and marker frequency in the chart-ready comparison
  table.
- Use the polar-reference ideal phases:
  - position 1.0: 180 degrees for all three markers;
  - position 2.0: 60 degrees at `f_2pi3`, 0 degrees at `f_mean`, and
    300 degrees at `f_pi2`.
- Derive the ideal 1.0-to-2.0 phase advances as 240, 180, and 120 degrees,
  respectively.

## Figure Contract

- Write the phase and phase-advance views as two separate figure files.
- In the phase figure, plot measured phase at positions 1.0 and 2.0 with ideal
  phase targets overlaid as horizontal markers.
- Order the mode groups as `f_2pi3`, `f_mean`, and `f_pi2`.
- Within every mode group, order the bars as position 1.0 then position 2.0.
- Use a two-level horizontal axis: individual position labels directly below
  the bars and frequency-mode labels below each pair.
- In the phase-advance figure, compare the measured modulo-360 phase advance
  with the ideal 240, 180, and 120 degree targets.
- In both figures, overlay the measured and ideal bars at exactly the same
  width. Keep the ideal as a gray bar/outline, shade excess above the ideal in
  muted red, and shade any deficit below the ideal in blue.
- Use fixed 0-to-360 degree scales so the backward-propagating `f_pi2` target
  remains visible at 300 degrees rather than being folded to -60 degrees.
- Add one-decimal direct labels to all bars.
- Distinguish positions with both color and hatch, avoiding red/green sign
  semantics.

## Reproducibility

Implement a dataset-agnostic visualization function plus a CLI that accepts
the marker table, requested positions, and output path. The requested result is
written to:

`fig/analyses/raw_sweep_260701_iris_portE/figures/phase_bar/position_1p0_vs_2p0_phase_bars.png`

No dataset-name conditional is allowed.

## Verification

- Unit-test selection, ordering, duplicate/missing-pair validation, and image
  creation.
- Run the CLI against the current analysis table.
- Numerically compare the six plotted values to `marker_pts.csv`.
- Open the final PNG and inspect its labels, signs, zero line, two-level
  horizontal axis, and layout.
