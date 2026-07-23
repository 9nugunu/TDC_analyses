# Tuning comparison figure design

## Purpose

Replace the automatic tuning-run radius-equivalence figure with the two figures
actually used to interpret the experiment:

1. a calibrated experiment-on-simulation `r_c` map; and
2. experiment-versus-simulation phase bars for baseline, current, and response.

The tuned experimental dataset owns these derived outputs. The simulation
dataset remains an input reference and does not own experiment-specific plots.

## Automatic trigger and inputs

The existing exact `_tune_` dataset trigger and campaign lookup remain
unchanged. For every measured family, the campaign supplies:

- a one-dimensional `r_c` simulation reference;
- the experimental anchor and comparison positions; and
- the design radius, currently `56.59 mm`.

Only families containing both configured experimental positions are rendered.
The current iris-only tuned dataset therefore renders iris outputs and records
cell as skipped.

Experimental marker frequencies already contain the measurement-temperature
correction. Experimental and vacuum-simulation phases are paired by marker
identity (`f_2pi3`, `f_mean`, and `f_pi2`); raw traces are not resampled at the
same uncorrected numeric frequency.

Port extension is not reapplied. The anchor position's `f_mean` phase is checked
against 180 degrees and written to `anchor_chk.csv`.

## Radius calibration

The calibrated current state follows the established design-referenced method:

1. evaluate the simulated `f_mean` phase at the design radius;
2. add the current measured `f_mean` phase at the comparison position;
3. invert the monotonic simulated `f_mean(r_c)` curve to obtain the current
   equivalent radius; and
4. hold that current radius fixed while fitting the baseline radius from the
   three measured marker phase changes.

These radii are simulation-equivalent coordinates. They are not absolute
physical cavity radii, bolt travel, or a torque-to-displacement calibration.

## Output contract

All automatic comparison products are stored under the tuned dataset:

```text
fig/analyses/<tuned-dataset>/
  tables/
    rc_fit.csv
    phase_cmp.csv
    anchor_chk.csv
  figures/
    tuning_cmp/
      phase_rc_map.png
      phase_cmp_bars.png
```

The manifest uses `outputs.tuning_cmp` and registers only these concise table
and figure names.

### `phase_rc_map.png`

This reproduces the intent of
`phase_rc_experiment_calibrated_to_design.png`:

- the three simulation phase curves versus `r_c`;
- a baseline fitted-radius vertical line;
- a current fitted-radius vertical line; and
- the ideal/design `r_c=56.59 mm` vertical line.

### `phase_cmp_bars.png`

This reproduces the intent of
`experiment_vs_simulation_phase_bars.png`:

- baseline experiment versus simulation;
- current experiment versus simulation; and
- tuning response, `current - baseline`, for experiment versus simulation.

The response panel is the fitted observable. Absolute-phase panels retain the
reference-plane distinction and must not be presented as the fit objective.

## Concise code names

The reusable public names are:

- `fit_rc_states()` for calibrated baseline/current equivalent radii;
- `plot_phase_rc_map()` for the radius-map figure; and
- `plot_phase_cmp_bars()` for the bar figure.

Longer legacy plotting names are removed after every in-repository caller and
test is migrated. No compatibility aliases are retained.

## Legacy cleanup

After the replacement passes tests and an actual tuning smoke run, remove the
known obsolete generated figures:

- `phase_rc_experiment_calibrated_to_design.png`;
- `experiment_vs_simulation_phase_bars.png`; and
- `iris_phase_radius_equivalence.png`.

Remove empty legacy comparison directories after verifying their resolved paths
remain under `fig/analyses`. Do not delete shared simulation line-scan CSVs,
standard analysis figures, or unrelated user outputs.

## Verification

- Unit-test the design-referenced current inversion and three-marker baseline
  fit with synthetic monotonic phase curves.
- Unit-test both concise figure functions and output filenames.
- Unit-test manifest replacement so verbose comparison names are absent.
- Run the real `raw_sweep_260701_iris_tune_Torque13p5` analysis.
- Confirm the anchor check, fitted radii, bar residuals, image labels, vertical
  lines, and legacy-file removal numerically and visually.
