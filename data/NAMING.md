# Data Dataset Naming

Dataset folders under `data/raw`, `data/sim`, and `data/prepro` must use:

```text
<layer>_<date>_<category>_<object>_<condition>
```

Required fields:

```text
layer: raw | sim | prepro
date: YYMMDD | undated
category: sweep | grid | dispersion
object/condition: ASCII alphanumeric tokens joined by underscores
```

Examples:

```text
raw_260604_sweep_iris_portE
sim_260526_grid_iris_offset
sim_260505_dispersion_single_cell_step1
sim_260605_sweep_iris_2d_solver_export_nonorm
prepro_260415_sweep_sparams_fullbrazing
```

Use `sweep` for tune-position, line, and 2D sweep datasets. The analysis code
decides later whether each tune-position region follows KYHL-style or nodal
shift logic.

Use `grid` for tuning sensitivity investigations such as parameter grids,
offset checks, field maps, or coupler radius sweeps.

Use `dispersion` for CST dispersion / phase advance / mode-frequency exports.

The folder prefix must match its parent layer:

```text
data/raw/raw_...
data/sim/sim_...
data/prepro/prepro_...
```

Analysis intentionally stops with `ValueError` when a dataset folder does not
follow this rule. Rename the dataset folder before running analysis.
