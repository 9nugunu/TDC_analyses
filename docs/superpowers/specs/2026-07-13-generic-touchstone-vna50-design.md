# Generic Touchstone and VNA-50 Transformation Design

## Goal

Load CST `S`, `Y`, and `Z` Touchstone datasets without treating every network
parameter as an S-parameter, then convert a one-port input-impedance sweep into
the VNA-equivalent 50-ohm reflection coefficient

\[
\Gamma_{50}(f)=\frac{Z_{11}(f)-50}{Z_{11}(f)+50}.
\]

The original CST datasets remain unchanged. The derived reflection data is
written as a separate, provenance-bearing `R 50` S1P simulation dataset.

## In-scope datasets

- Raw CST S1P source:
  `data/sim/sim_sweep_260713_coupler_tuner_insertion`
- CST Z1P input-impedance source:
  `data/sim/sim_sweep_260713_coupler_tuner_zin`
- Derived VNA-equivalent output:
  `data/sim/sim_sweep_260713_coupler_tuner_vna50`

The S1P and Z1P source folders each contain four CST runs. Their Navigator
metadata maps the runs to `tuner_insertion_depth = 0, 4, 8, 12`, and their
frequency grids contain 10,001 matching points from 2.6 to 3.0 GHz.

## Physical definition

For this one-port dataset, `Z11` is the input impedance `Zin`. Every exported
complex Z11 sample is converted independently using the fixed real VNA
reference impedance `50.0 ohm`.

The output Touchstone option line is `# GHz S RI R 50` because the written
complex values are the reflection coefficient referenced to 50 ohm. This is a
derived representation of the same simulated one-port network; it does not add
a second physical termination or alter the CST geometry.

This first implementation is intentionally one-port only. For an N-port
dataset, diagonal `Znn` alone does not reproduce a VNA measurement with the
other ports terminated. N-port load reduction is out of scope.

## Architecture

### 1. Generic Touchstone parsing

The low-level Touchstone reader accepts `.sNp`, `.yNp`, and `.zNp` extensions
and preserves the header parameter kind (`S`, `Y`, or `Z`). Its internal value
names are parameter-neutral. Compatibility access for existing S consumers is
kept during the refactor so current S-parameter behavior does not change.

Touchstone discovery recognizes all three supported parameter families. A
dataset folder must contain exactly one network-parameter family. Mixed S/Y/Z
folders stop with a clear error instead of selecting one family implicitly.

### 2. Parameter-specific adapters

The generic parser provides complex network values and metadata. Thin adapters
produce explicit tables:

- S: `s_name`, `s_real`, `s_imag`, `s_db`, `s_phase_deg`
- Y: `y_name`, `y_real_siemens`, `y_imag_siemens`
- Z: `z_name`, `z_real_ohm`, `z_imag_ohm`

`reference_ohm` and S-specific `is_normalized` semantics are not reused to
describe physical Y or Z values. In particular, the Z1P header `R 1` does not
mean that Z11 must be divided by one ohm or treated as a normalized S value.

Existing simulation metadata behavior remains centralized in `SimLoader`:
CST parameter comments, run IDs, `result_navigator.csv`, `sim_*` fields, and
`scan_type` are attached consistently to S, Y, and Z tables.

### 3. Pure VNA transformation

A small analysis function accepts a complex Z11 array and a positive finite
reference impedance and returns complex Gamma. It performs no file I/O and has
no dataset-specific branches.

The production workflow calls it with `reference_ohm=50.0`. It rejects
non-finite input values, non-positive reference impedance, and samples for
which `Z11 + 50` is numerically singular.

### 4. Derived Touchstone dataset writer

The writer creates the output folder without editing either source folder. It
keeps the source run basenames, changes the extension from `.z1p` to `.s1p`,
and writes complex RI values at the original frequencies.

Each output file records:

```text
! Derived from CST Z11
! Gamma50 = (Zin - 50) / (Zin + 50)
! Source dataset: sim_sweep_260713_coupler_tuner_zin
# GHz S RI R 50
```

`result_navigator.csv` is copied unchanged. A `transform_manifest.json`
records the source dataset, formula, reference impedance, parameter family,
run/file mapping, frequency range, point count, and output dataset ID.

The output folder must not already contain conflicting S1P files. Re-running
against an identical complete output may replace only files owned by this
transformation after validating the manifest; otherwise the workflow stops.

## Data flow

```text
sim_sweep_260713_coupler_tuner_zin/*.z1p
        |
        v
generic Touchstone reader -> Z adapter -> Z11 table with Navigator metadata
        |
        v
Gamma50 = (Z11 - 50) / (Z11 + 50)
        |
        v
sim_sweep_260713_coupler_tuner_vna50/*.s1p
        |
        v
existing S loader and existing S11/Smith-chart analysis
```

The raw `R 0` S1P dataset remains available for modal-reference comparisons,
but it is not required to compute Gamma50 once Z11 is supplied directly.

## Naming and routing

All dataset folders follow `data/NAMING.md`:

```text
<layer>_<category>_<date>_<object>_<condition>
```

No hyphens or dataset-specific loader branches are introduced. Simulation
identity remains under `data/sim`; `data/prepro` is not used for this derived
simulation product.

## Errors and validation

The workflow stops with a targeted error when:

- a dataset name violates `data/NAMING.md`;
- a folder mixes S, Y, and Z Touchstone families;
- a Z input contains anything other than one-port `Z11` rows;
- a run ID is missing or duplicated;
- Navigator metadata does not map one-to-one to the four Z files;
- files in one dataset use different frequency grids;
- any Z11 or Gamma50 value is non-finite;
- `Z11 + 50` is numerically singular;
- a conflicting output dataset already exists.

The workflow permits tiny floating-point passivity excursions such as
`abs(Gamma50) = 1 + O(1e-15)` and does not clip complex values to the unit
circle. Larger excursions are reported in the manifest rather than silently
modified.

## Tests

1. Extend low-level parser tests for `.z1p`, `parameter="Z"`, RI values, and
   one-port value count.
2. Extend discovery tests so S/Y/Z files are recognized and mixed families are
   rejected.
3. Preserve the complete existing S loader test suite as a compatibility
   guard.
4. Preserve the existing direct Y11 workflow while routing it through the
   generic parsing boundary.
5. Add Z adapter tests for units, run IDs, Navigator metadata, frequency, and
   complex Z11 columns.
6. Add unit tests for the Gamma50 formula, invalid reference impedance,
   non-finite Z11, and a singular denominator.
7. Add a writer round-trip test: write derived S1P, reload it through the
   existing S loader, and confirm `reference_ohm=50`, complex values, run
   metadata, and frequency grid.
8. Use the existing paired `1DRcFine` Z/Y and CST `50norm` artifacts as a
   numerical regression: the Python-derived Gamma50 must match the CST
   50-ohm result within the precision of the exported files.
9. Run the production smoke check on the four `coupler_tuner_zin` files and
   confirm four output S1P files with 10,001 points each.

## Out of scope

- Modifying or deleting raw CST files.
- Hardcoding a particular dataset ID in reusable code.
- Interpolating mismatched frequency grids.
- N-port termination or load-reduction calculations.
- Cable, adapter, transition, or calibration-error modeling.
- Automatic reference-plane de-embedding.
