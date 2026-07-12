# Navigator-driven S11 figure design

## Goal

For simulation datasets with a `result_navigator.csv`, make one S11 figure per
distinct changing parameter point. Put that point in the output filename and
figure title. Do not create the unreadable all-run S11 overview for those
datasets.

## Scope

- Read only metadata already merged by `SimLoader` as `sim_*` columns.
- Treat a Navigator column as a sweep parameter when it has more than one
  non-null value in the S11 table.
- Keep traces together only when all changing Navigator parameter values are
  identical. This preserves legitimate same-point port pairs.
- Keep the existing raw-experiment overview and simulation datasets without a
  changing Navigator parameter unchanged.

## Output convention

Use an ASCII-safe filename with values encoded using `p` for decimal points
and `m` for negative signs. Use a readable title with native parameter names
and values.

Geometry parameters use fixed precision so lexical order matches numeric
order: `sim_r_c` has two decimal places and `sim_w_c` has four. Therefore
`56.10` is written as `56p10`, while `56.01` is written as `56p01`.

```text
s11/r_c_56p09.png
S11 magnitude | r_c = 56.09 mm

s11/r_c_56p09_w_c_19p0224.png
S11 magnitude | r_c = 56.09 mm; w_c = 19.0224 mm
```

The current repository has no unit column in `result_navigator.csv`. The
first implementation therefore displays units only for the known geometry
parameters `r_c` and `w_c`, both in millimetres. Other Navigator parameters
are displayed without an inferred unit.

## Compatibility and errors

- `result_navigator.csv` remains optional.
- Existing run-id matching and marker-point joins remain unchanged.
- No source data files are renamed.
- A repeated parameter tuple is intentionally one figure. Its source traces
  remain visible together, so the result can expose duplicate exports rather
  than silently discard them.

## Tests

Add S11 plan tests for a one-dimensional `r_c` sweep and a two-dimensional
`r_c`/`w_c` sweep. They must assert that no `overview` plan is created and
that filenames and titles contain the actual Navigator values. Retain the
existing raw-data overview tests as the compatibility guard.
