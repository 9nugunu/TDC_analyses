# Concise `r_c` Phase-State Labels

## Goal

Make the `r_c` phase-map filename and state annotations concise and immediately readable without changing the simulated phase curves, fitted radii, or guide-line positions.

## Design

- Use `rc_phase_states.png` as the concise artifact name for the rendered state-comparison figure.
- Use one state vocabulary consistently: `Before tuning`, `Torque 13.5`, and `Design`.
- Show the radius in each annotation as `r_c = <value> mm`.
- Label the auxiliary zero-crossing guide as `f_mean = 0°`.
- Provide a small reusable `plot_phase_rc_states()` wrapper and CLI so the four-guide figure is reproducible.
- Keep the existing figure title and all numerical calculations unchanged.

## Verification

- Unit tests assert the labels passed by `plot_phase_rc_map()`.
- The targeted plotting test is run after the change.
- The updated figure is regenerated and checked for a non-empty PNG.
