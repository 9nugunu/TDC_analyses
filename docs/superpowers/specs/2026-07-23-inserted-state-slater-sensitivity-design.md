# Inserted-State Slater Sensitivity Design

## Goal

Extend the existing 3D field workflow so that it preserves the current
NoPlunger equal-volume baseline and also calculates the local field term for
the next small motion of each already-inserted plunger state.

The two results answer different questions:

- the NoPlunger baseline isolates the effect of axial position under a common
  unperturbed field;
- the inserted-state calculation includes the field redistribution produced by
  the plunger that is already present.

Neither result is a substitute for a direct resonant-frequency sweep of the
actual geometries.

## Physical definition

For each inserted case, the existing tip detector identifies the first axial
plane inside the axis-connected PEC region. The PEC region continues from that
tip toward the maximum exported \(z\). A further insertion therefore displaces
the vacuum slab immediately before the detected tip:

\[
z_{\mathrm{sample}}=z_{\mathrm{tip}}-\Delta z .
\]

Inside the inferred circular plunger cross-section, calculate

\[
U_{E,\mathrm{ins}}^{\Delta V}
=\frac{\epsilon_0}{4}
\int_{\Delta V_{\mathrm{next}}}|\mathbf E_{\mathrm{ins}}|^2\,dV,
\qquad
U_{H,\mathrm{ins}}^{\Delta V}
=\frac{\mu_0}{4}
\int_{\Delta V_{\mathrm{next}}}|\mathbf H_{\mathrm{ins}}|^2\,dV,
\]

\[
K_{\mathrm{ins}}
=U_{E,\mathrm{ins}}^{\Delta V}
-U_{H,\mathrm{ins}}^{\Delta V}.
\]

Normalize this local term by the same inserted case's export-region energy:

\[
\eta_{\mathrm{ins}}
=\frac{K_{\mathrm{ins}}}
{U_{E,\mathrm{ins}}^{V}+U_{H,\mathrm{ins}}^{V}} .
\]

The signed frequency-shift direction remains convention-dependent. The
workflow therefore stores the signed \(E-H\) field term and normalized value,
not a mechanically signed tuner coefficient.

## Field-basis limitation

The current files are complex CST field monitors named at
\(f=2.857\) GHz. Their filenames do not establish that every inserted case was
evaluated at its own shifted eigenfrequency. The inserted-state result must
therefore be labelled as a normalized fixed-frequency field proxy unless
independent provenance establishes that the fields are same-mode eigenfields
or resonant fields.

The workflow must not claim an absolute \(\mathrm{kHz/mm}\) tuner coefficient
from these files alone. A reader may apply a stated frequency only as an
explicit first-order conversion, with the fixed-frequency limitation retained.

## Output contract

Keep `tables/slater_pos.csv` and its existing baseline columns unchanged for
backward compatibility. Append explicit inserted-state columns:

- `inserted_case_id`
- `inserted_sample_z_mm`
- `inserted_step_mm`
- `inserted_voxel_count`
- `inserted_e_j`
- `inserted_h_j`
- `inserted_k_e_minus_h_j`
- `inserted_k_over_u`
- `inserted_abs_k_over_u`
- `inserted_to_baseline_k_ratio`
- `inserted_field_basis`

The existing `e_j`, `h_j`, `k_e_minus_h_j`, and `abs_k_over_u` columns remain
the NoPlunger equal-volume baseline. Manifest metadata must identify both
quantities and the schema version must change so stale cached results are not
reused.

Update `slater_pos.png` to compare baseline and inserted-state signed
\(K_{E-H}\) at each detected tip position. The plot must distinguish the two
field bases without implying that either bar is a measured or direct
eigenfrequency shift.

## Data flow

1. Summarize every E/H pair exactly as the existing workflow does.
2. Select `NoPlunger` as the common baseline.
3. Detect every inserted tip position and the common equivalent radius.
4. Compute the existing NoPlunger disk term at each tip position.
5. For each inserted case, read its E/H plane one grid step before its tip.
6. Integrate the next vacuum slab inside the common circular cross-section.
7. Normalize by that inserted case's own export-region energy.
8. Join the inserted result to the matching baseline row.
9. Save the comparison table, plot, and updated manifest.

The cropped `default` and full `NumDepth0.5` files share the first position.
The canonical inserted-state calculation must use the full case when duplicate
tip positions exist. Cropped files remain visible in `field3d_src.csv` and
`field3d_energy.csv` as provenance and cross-check data.

## Validation

Synthetic tests must demonstrate that:

- the inserted calculation samples the vacuum-side plane before the tip, not
  the zero-field PEC plane;
- it uses the inserted case field rather than the NoPlunger field;
- it normalizes by the inserted case's own total export-region energy;
- the baseline columns retain their existing values;
- duplicate tip positions select the full inserted case deterministically;
- a case with no preceding vacuum plane raises a clear error;
- the workflow and manifest expose both field bases and invalidate the old
  cache schema.

After focused tests pass, run the real
`sim_profile_260723_3DEMfield` dataset, inspect the numerical comparison, and
verify the revised plot visually. The final interpretation must report the
NoPlunger baseline and inserted-state proxy separately. Direct CST
eigenfrequency or resonance finite differences remain the validation target
for an actual tuner coefficient.
