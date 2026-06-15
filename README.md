# Python Deflector Tuning

Clean-room, step-by-step deflector tuning analysis code.

## Design rules

1. Keep the first routing decision centralized and obvious:
   - `data/sim/...` -> simulation/CST data loader
   - `data/raw/...` -> experimental raw data loader
   - `data/prepro/...` -> processed experimental products
2. Do not add dataset-specific `if dataset_name == ...` branches.
3. Prefer small dataclasses, simple functions, and explicit class names.
4. Add behavior one step at a time with tests first.

## Preprocessed data policy

`data/prepro` is a processing-stage folder, not a required format conversion
folder.

If an experimental dataset arrives as usable Touchstone (`.s1p`, `.s2p`, etc.)
under `data/raw`, it can share the same low-level Touchstone parser as
simulation data. It still keeps experiment identity in the dataset metadata.
There is no need to create a preprocessed copy just to make it Touchstone.

Use `data/prepro` only when the experimental raw export is CSV-like or when a
new derived product exists, for example:

- cleaned or filtered CSV tables;
- calibrated or normalized traces;
- manually curated exports;
- merged metadata tables;
- any output that should not be confused with the original raw instrument file.

Step 0 only implements source-layer detection. No S-parameter parser or plotting
is included yet.
