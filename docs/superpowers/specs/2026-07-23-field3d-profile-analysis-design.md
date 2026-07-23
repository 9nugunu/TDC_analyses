# Automatic 3D Field Profile Analysis Design

## Goal

Make `run_folder_analysis.py <profile-folder>` automatically recognize CST
three-dimensional complex E/H text exports, analyze them without requiring
YAML or a separate command, save compact tables and figures, and reuse the
saved result when the source files have not changed.

## Routing

The existing dataset-layer and dataset-name routing remains unchanged. Within
the existing `profile` lane, a header-only detector runs before the
one-dimensional profile loader:

- `x [mm] y [mm] z [mm] ExRe ... EzIm` selects the 3D E-field lane.
- `x [mm] y [mm] z [mm] HxRe ... HzIm` selects the 3D H-field lane.
- Existing `#"Z / mm"` exports continue through the current 1D profile lane.

The detector must not read the full file merely to determine its format.

## Pairing and case identity

E/H files are paired by removing the leading `e-field` or `h-field` token from
the filename and matching the remaining text case-insensitively. A suffix after
the CST result token `]` becomes the case identifier. When the suffix is empty,
the case is named `default`. No dataset-specific filename branch is allowed.

All valid pairs in the folder remain visible in the output. Thus a cropped
`default` pair and a full `NumDepth0.5` pair can coexist and serve as a
cross-check.

## Numerical analysis

Files are read in bounded chunks. For each field file the workflow records:

- row count, rectangular grid size, bounds, and uniform spacing;
- component-wise sums of complex magnitude squared;
- export-region electric or magnetic energy using the phasor time-average
  factor \(1/4\);
- the closest-to-axis field values versus \(z\).

For each E/H pair, a plunger tip is detected as the first axial plane followed
by a continuous zero-field region to the exported \(z\)-maximum. `NoPlunger`
is the preferred reference case. The plunger radius is inferred from the
connected zero-field component containing the axis on a plane downstream of a
detected tip.

At every unique detected tip position, the no-plunger field is sampled on its
nearest plane inside the inferred circular disk. The saved Slater quantities
are:

\[
U_E^{\Delta V}=\frac{\epsilon_0}{4}\int_{\Delta V}|\mathbf E|^2\,dV,
\qquad
U_H^{\Delta V}=\frac{\mu_0}{4}\int_{\Delta V}|\mathbf H|^2\,dV,
\]

\[
K_{E-H}=U_E^{\Delta V}-U_H^{\Delta V}.
\]

The first version reports the signed local \(K_{E-H}\), its magnitude
normalized by the no-plunger export-region energy, and ratios between
positions. It does not label the result as an absolute frequency shift because
the sign and absolute Slater prefactor depend on the moving-boundary
convention and the selected stored-energy normalization.

## Outputs

The canonical output remains
`fig/analyses/<dataset>/` and contains:

- `tables/field3d_src.csv`
- `tables/field3d_energy.csv`
- `tables/slater_pos.csv`
- `figures/field3d/field_comp.png`
- `figures/field3d/slater_pos.png`
- `manifest.json`

The source table preserves enough provenance to distinguish cropped and full
exports. Plot labels remain concise while mathematical labels retain LaTeX.

## Automatic reuse

The manifest records workflow schema version plus each input file's name,
size, and nanosecond modification time. A repeated folder run returns the
existing tables and figures immediately when:

- the schema version matches;
- every source signature matches;
- every declared output still exists.

Any input or schema change causes an automatic recomputation. No cache option,
YAML file, or dataset-specific registration is required.

## Validation

Synthetic tests cover header detection, pairing, chunked energy aggregation,
tip/radius inference, Slater disk integration, output contracts, cache reuse,
and runner routing. The real
`sim_profile_260723_3DEMfield` folder is then executed and its CSV values and
both figures are inspected.

