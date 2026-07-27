# State Temperature Marker Correction Design

## Goal

Record the 30.0 C design-dimension reference and the 23.4 C measurement
temperature for tuning states s003 and s004, then use those metadata whenever
their experimental marker frequencies are sampled.

## Scope

- Add campaign-level marker-correction defaults: design temperature, humidity,
  copper thermal expansion coefficient, and humid-air permittivity.
- Add an optional measurement temperature to each tuning state.
- Resolve an experimental state to a `TemperatureHumidityCorrection` before
  marker analysis; leave unregistered datasets and states without a measured
  temperature on the existing defaults.
- Pass the resolved correction into the existing marker-analysis pipeline.
- Preserve correction provenance in the campaign section of the manifest.
- Regenerate the existing s003 and s004 analysis directories in place after
  verification.

## Data Model

`campaign.marker_correction` holds campaign-wide reference values.  The YAML
uses `design_temp_C` because it refers to the mechanical drawing/CST geometry
reference, not an RF operating set point.  `states.<id>.temp_meas_C` holds the
temperature at that state's RF measurement.

For a state with `temp_meas_C`, the resolved correction is:

```text
TemperatureHumidityCorrection(
    temp_op_C=campaign.marker_correction.design_temp_C,
    temp_meas_C=state.temp_meas_C,
    humidity_fraction=campaign.marker_correction.humidity_fraction,
    thermal_alpha_per_C=campaign.marker_correction.thermal_alpha_per_C,
    eps_air_humid=campaign.marker_correction.eps_air_humid,
)
```

The existing frequency rule remains unchanged:

```text
f_marker = f_dispersion /
           (1 + alpha * (T_meas - T_design)) /
           sqrt(epsilon_air_humid)
```

For `iris_260701`, the campaign records a 30.0 C design reference and states
s003 and s004 record 23.4 C measurements.

## Integration

The tuning-campaign workflow resolves the correction from dataset identity.
The CLI and batch runner resolve it before calling `run_folder_analysis`; the
runner accepts it as an optional argument and forwards it to
`build_marker_analysis`.  This keeps campaign lookup out of the generic runner
and avoids dataset-name branches.

The manifest's `tuning_campaign` section records the resolved correction for a
state, so regenerated tables and figures can be traced to the exact thermal
assumption.

## Validation and Compatibility

- All correction numbers must be finite numeric values.
- A state without `temp_meas_C` resolves to no override and therefore retains
  the current default correction behavior.
- Simulation marker extraction remains uncorrected because the existing
  `marker_role="sim"` contract ignores experimental corrections.
- Existing YAML files without `marker_correction` and state temperatures remain
  valid through defaults matching the current code values.

## Verification

Tests will prove YAML parsing, correction resolution, pipeline forwarding, and
manifest provenance.  Then the focused suite will run before regenerating s003
and s004; regenerated marker tables must report `temp_op_C=30.0` and
`temp_meas_C=23.4`.
