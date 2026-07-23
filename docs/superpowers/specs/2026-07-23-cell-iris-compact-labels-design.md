# Cell-Iris Compact Labels

## Scope

Shorten only the visible titles, axis labels, legend labels, and x-tick labels
of the four `cell_iris_response` figures.  Keep the calculations, ordering,
output filenames, and plotted values unchanged.

## Copy

- Titles: `Iris/Cell |Y11| ratio`, `Iris/Cell $\Delta\phi$ ratio`, `Phase residual`, and
  `Axis error`.
- X axis: `Transition`.
- Y axes: `Iris / Cell`, `$|\Delta\phi_I| / |\Delta\phi_C|$`,
  `$|\phi-\phi_0|$ [deg]`, and `$|\phi-\phi_{axis}|$ [deg]`.
- Legends: `equal`, `Cell`, and `Iris`.
- Tick labels: compact marker/pair notation such as
  `f_mean P1\nC0.5-1.5 | I1-2`.

The shared bold-math style protects `\Delta`, `\phi`, and other existing
commands so LaTex symbols and subscripts render correctly.

## Verification

Capture the labels passed to the shared axis-style helper in a unit test, then
regenerate the existing s004 figures and inspect their text visually.
