# Tuning campaign metadata design

## Goal

Represent a sequence of mechanical tuning states, cross-dataset references,
and experimental issues in one compact human-editable file. The file is the
source of truth for experiment-versus-simulation comparison plots. CSV tables,
when useful, are derived outputs rather than files that a person must maintain.

The normal single-folder analysis workflow remains valid without campaign
metadata. A tuning comparison workflow requires a valid campaign file and must
not silently guess missing state relationships.

Experimental datasets opt into campaign handling through an explicit `_tune_`
token in the folder name. Ordinary and baseline datasets do not search the
campaign registry. A tuning dataset must have exactly one state registration;
otherwise analysis reports a registration error. A readable folder token such
as `Torque13p5` is checked against, but never replaces, the YAML `torque_nm`
value.

## Source file and location

Store one YAML file per campaign under `config/tuning_campaigns/`, for example:

```text
config/tuning_campaigns/iris_260701.yaml
```

A campaign file can reference multiple folders under `data/raw` and one
simulation dataset under `data/sim`. It must not contain copied S-parameter or
cell-measurement values.

YAML is preferred over CSV because state and issue fields are sparse and may
contain nested optional information. The implementation will add PyYAML as an
explicit project dependency.

## Compact schema

State and issue IDs are mapping keys, so entries do not repeat an `id` field.
Keys are short but still recognizable. One-letter keys are not used.

```yaml
schema: 1

campaign:
  id: iris_260701
  sim: sim_grid_260701_1DRcFine
  rc_design_mm: 56.59
  phase:
    port_ext: applied
    ext_mm: null
    verified: false

states:
  s000:
    role: baseline
    data: raw_sweep_260701_iris_portE
    torque_nm: 0
    meas: done

  s001:
    prev: s000
    data: raw_sweep_260701_iris_tune_Torque13p5
    torque_nm: 13.5
    meas: done
    flag: provisional

  s002:
    prev: s001
    meas: pending
    change: tuner_bolt_added_only

issues:
  i001:
    after: s001
    verify: s002
    tag: input.lr_tuner_disconnect
    check: [output]
    note: 튜너 볼트와 내부 Radius 조정 수나사 분리
```

`s000` is the RF baseline: the pre-tuning, Torque 0 measurement. `s001` is the
completed Torque 13.5 measurement. `s002` is the post-intervention mechanical
state in which only the tuner bolt was added; its measurement remains pending
until new data are acquired.

The port-extension block records the present evidence without inventing the
unknown extension length. The existing measurement data appear to have the
correction applied, but the exact value is not yet verified.

## State fields

Only `meas` is required for every state. Other fields are required according to
the state:

- `role`: `baseline` only for the unique RF baseline.
- `prev`: preceding state ID; omitted for the first state.
- `data`: raw dataset folder name. Required when `meas: done`.
- `torque_nm`: measured torque when known.
- `ang_deg`: measured rotation relative to the declared mechanical reference
  when known.
- `meas`: `pending` or `done`.
- `change`: short description of the mechanical change from `prev`.
- `flag`: optional data-quality flag such as `provisional`.
- `unc`: optional numeric uncertainty mapping, for example
  `{phase_deg: 0.3}`. Qualitative uncertainty belongs in `note`.
- `note`: optional free-form note.

State IDs encode sequence only. Torque and angle remain separate values and are
not encoded into the ID.

## Issue fields and defaults

Issue entries are intentionally sparse. The common defaults are supplied by the
loader and are not repeated in YAML:

```yaml
stat: open
sev: warn
action: flag
```

Available fields are:

- `after`: state after which the issue was discovered.
- `verify`: future or completed state intended to test its effect.
- `tag`: compact searchable classification.
- `check`: components or channels whose possible impact must be checked.
- `stat`: `open`, `verify_pending`, or `closed`.
- `sev`: `info`, `warn`, or `critical`.
- `action`: `include`, `flag`, or `exclude`.
- `note`: human-readable observation.

The current input-coupler disconnection does not assert that the output coupler
was affected. `check: [output]` means that output-coupler impact is unknown and
must be evaluated using the `s002` remeasurement. Adding the tuner bolt alone
does not automatically close the issue; it remains open until verification.

## Loading, validation, and analysis behavior

Add small typed campaign, state, and issue models plus one YAML loader. The
loader expands defaults in memory but does not rewrite the source file merely
to materialize them.

Validation fails early when:

- the schema version is unsupported;
- there is no unique baseline;
- a `prev`, `after`, or `verify` reference does not exist;
- the `prev` chain is cyclic;
- a completed state has no dataset reference;
- two completed states reference the same dataset unintentionally;
- the simulation or completed-state dataset cannot be found under its expected
  centralized data layer.

Pending states are valid without a dataset. Ordinary single-folder analysis is
unchanged. Campaign comparison analysis requires an explicit YAML path or an
unambiguous campaign match and reports missing metadata instead of guessing a
baseline or simulation variant.

Plots retain flagged measurements and show their issue/quality marker. A point
is omitted from a fit only when its effective issue action is explicitly
`exclude`; no raw measurement is silently discarded.

When requested, normalized state and issue tables may be exported under
`data/prepro` as derived CSV products. Those CSV files are never the editing
source.

## Conversational maintenance contract

The user does not need to edit YAML directly. When the user reports a new state,
measurement, intervention, or issue in natural language, Codex will:

1. read the current campaign file;
2. choose the next sequence ID when one is not supplied;
3. connect the state or issue to the stated preceding and verification states;
4. preserve uncertain facts as `pending`, `provisional`, or an explicit check,
   rather than inferring that an effect occurred;
5. validate the updated campaign;
6. report exactly which entry was added or changed.

Existing entries are not silently overwritten. Ambiguity that would change the
physical interpretation, such as whether an intervention occurred before or
after a measurement, requires confirmation.

## Tests

Tests will cover:

- loading the compact example and expanding issue defaults;
- allowing `s002` to remain pending without a dataset;
- requiring a dataset for `meas: done`;
- unique-baseline, reference, and cycle validation;
- preserving the unknown output-coupler impact as a verification target;
- retaining flagged points and excluding only explicit `action: exclude`
  points from fitting;
- optional normalized CSV export without treating it as source metadata;
- compatibility of the existing single-folder runner when no campaign file is
  present.
