# Automatic Tuning-State Registration

## Goal

Make a newly analysed `*_tune_sNNN` raw dataset immediately usable for its
campaign comparison without requiring a per-dataset YAML edit.  The campaign
YAML remains the single, central experiment ledger; no metadata file is added
inside individual raw-data folders.

## Campaign declaration

One campaign may declare `campaign.auto_states: true`.  A new tuning dataset
can be inferred only when exactly one such campaign exists.  This campaign is
the active campaign until a second campaign is introduced.

## Dataset naming

- `..._tune_sNNN[_tag]` is the primary RF measurement of mechanical state
  `sNNN`; an optional trailing tag such as `broken` is recorded as a quality
  flag only when it can be parsed safely.
- `..._tune_sNNN_aux_plunger` is an auxiliary plunger-sensitivity measurement
  attached to state `sNNN`, not a new state.
- Existing exact YAML state and auxiliary mappings always win over inferred
  matching.  This preserves the legacy
  `raw_sweep_260721_tune_s003_plungersensitivity` auxiliary entry.

## Automatic registration

After ordinary folder analysis succeeds, a new primary state is recorded in
the active campaign YAML as:

```yaml
s004:
  prev: s003
  data: raw_sweep_260722_tune_s004
  meas: done
  flag: auto_pending
```

The runner accepts only the next consecutive numeric state and never overwrites
an existing state ID with another dataset.  A torque token in the folder name
is recorded when present; otherwise torque remains absent.  The campaign match
written to the run manifest is marked as inferred/automatic.

## Follow-up metadata

The user or Codex can later add confirmed torque, rotation angle, notes, and
issues to the same central campaign YAML.  Derived `states.csv` remains an
output view, not the record of authority.

## Failure behaviour

Ambiguous active campaigns, a non-consecutive state, an existing state-ID
collision, or an auxiliary dataset whose parent state is absent are rejected
with an actionable error.  A non-tuning dataset remains unaffected.

## Verification

Tests cover automatic primary-state registration, idempotent re-analysis,
state collision rejection, automatic auxiliary classification, and precedence
of existing explicit S003 mappings.
