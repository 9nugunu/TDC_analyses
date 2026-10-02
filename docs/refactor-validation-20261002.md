# Refactor validation — 2026-10-02

The baseline was local commit `bf96acf` before the execution, campaign, and polar
refactors. Its complete test suite passed: **455 tests**. The final refactored
suite passed: **495 tests in 35.56 seconds**. These counts include parametrized
cases; they are not a code-coverage measurement or performance comparison.

## Executed test command

Run from the repository root on this workstation:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:MPLBACKEND='Agg'
& 'C:/Users/KST/.conda/envs/lpstomo/python.exe' -m pytest -q -p no:cacheprovider
```

## Output preservation

A source snapshot of the baseline and the final working source were executed in
separate processes against the same isolated input paths. Eight cases covered:

1. Processed experimental S11 data with multiple tuning positions.
2. Simulation grid data with navigator metadata.
3. Direct Y11 data.
4. Direct Z11 data.
5. CST dispersion exports.
6. CST 1D field profile.
7. Paired complex 3D E/H fields.
8. Copies of the real FullBrazing `0.5.S2P` and `1.5.S2P` inputs from
   `data/raw/raw_sweep_260415_sparams_fullbrazing`.

All **57 CSV files matched byte for byte** and all **46 PNG files matched in
dimensions and RGBA pixels**. Output lists, analysis modes, and manifests matched
after normalizing the output directory and excluding the new `figure_cache`
metadata. The first seven cases use controlled test inputs. This does not claim
that every production dataset has been rerun or that every scientific model has
been independently validated.

The local comparison harness, source snapshot, input copies, comparison records,
and execution logs are retained under the ignored `tmp/refactor_20261002/` folder.
Production data and analysis output folders were not regenerated.

## Intentional behavior changes

- Single and batch entrypoints now use the same campaign execution policy.
  Regression tests compare synthetic plunger sensitivity CSV values and manifest
  products in normal and tables-only modes. Batch scheduling remains separate.
- Grid S11 images are reused only for matching data, marker conditions, style,
  code identity, and runtime versions. Tests cover changed numerical inputs,
  marker conditions, settings, missing images, and legacy manifests.
- Tables-only refresh preserves the original signature of a complete existing
  figure group. A partial group loses its signature so the next full run restores
  missing images. This interleaved missing-image case was found in independent
  review, fixed, and included in the final suite.

Scientific calculations and the standard CSV contract are preserved. Existing
nodal-shift tests continue to verify that `f_mean` is excluded from target errors.
