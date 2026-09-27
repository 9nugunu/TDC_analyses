# Geometry-response Cell/Iris Color Design

## Goal

Make Cell and Iris traces immediately distinguishable while preserving the
existing frequency-marker color families.

## Visual contract

- Keep the existing frequency hues:
  - `f_2pi3`: red
  - `f_mean`: blue
  - `f_pi2`: green
- Render Cell traces as a lighter, lower-saturation tint of the frequency hue.
- Render Iris traces as the existing darker, higher-saturation frequency hue.
- Retain the current redundant encodings:
  - Cell: circle marker and solid line
  - Iris: square marker and dashed line
- Keep the legend layout as `Cell | Iris`, with matching frequencies on each
  row.
- Draw a gray vertical reference line at the sampled baseline
  \(L_{c,0}=29.148\ \mathrm{mm}\) in both absolute-phase and phase-pickup
  figures.
- Label the absolute-phase vertical axis as \(\phi\) in degrees.
- Label the phase-pickup vertical axis as \(\Delta\phi\) in degrees.

## Phase-reference definition

The phase pickup is not referenced to one common frequency marker. Each marker
and each shorting state uses its own phase at the baseline geometry:

\[
\Delta\phi_{m,s}(L_c)
=
\operatorname{wrap}_{180}
\left[
\phi_{m,s}(L_c)-\phi_{m,s}(L_{c,0})
\right],
\]

where \(m\) is `f_2pi3`, `f_mean`, or `f_pi2`, \(s\) is Cell or Iris, and
\(L_{c,0}=29.148\ \mathrm{mm}\).

## Scope

Apply the state-specific color treatment only to the geometry phase-response
figures. Do not change marker definitions, phase values, units, axes, or other
plot families.

## Implementation

Derive the Cell tint from each `MARKER_COLORS` base color by blending it with
white. Use the unmodified base color for Iris. Centralize the transformation
in the geometry-response plotting module so every combined and per-marker
figure uses the same rule.

## Verification

- A regression test must confirm that paired Cell/Iris traces keep the same hue
  family but use different colors, with Cell lighter than Iris.
- A regression test must confirm the vertical line is located at the tabulated
  `sweep_base` and does not enter the legend.
- A regression test must confirm the two vertical-axis labels are \(\phi\) and
  \(\Delta\phi\), respectively.
- Regenerate all eight geometry-response PNG files from the existing
  `geom_phase.csv`.
- Visually inspect the combined absolute-phase and phase-pickup figures.
- Run the complete test suite before reporting completion.
