# Runner Structure Refactor Plan

**Goal:** Preserve all analysis behavior while reducing `runner.py` responsibility and eliminating duplicate S-parameter folder reads.

**Constraints:** Preserve the current uncommitted KYHL changes, public imports, runner-level monkeypatch seams, logger name, filenames, and manifests. Add no dependencies and do not commit or push.

## Tasks

1. Add a real integration regression test proving one S-parameter folder is loaded once per run; run it red.
2. Allow `build_marker_analysis` to consume an already-loaded table and pass the runner's table through; run marker/runner tests green.
3. Extract workflow result types, CST profile workflow, CST dispersion workflow, mode detection, and manifest/cache helpers into focused modules while re-exporting existing runner names.
4. After each extraction, run the smallest relevant tests and static checks.
5. Run the full test suite, Ruff on changed files, a real CLI smoke test, and a final code/debug review.

## Safety checks

- Keep `run_folder_analysis`, `detect_analysis_modes`, and `resolve_input_paths` callable from `deflector_tuning.runner`.
- Keep plot and analysis calls in the runner namespace where tests and callers replace them dynamically.
- Keep extracted modules below 250 lines of executable Python.
- If behavior changes or a test exposes an undocumented contract, stop that extraction and retain the existing runner implementation.
