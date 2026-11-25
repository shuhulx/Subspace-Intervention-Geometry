# Results and archived records

The pilot contains 60 continuations: the same ten prompts under six conditions.
The ten prompts are the first two eligible prompts in each of five categories
in the pinned upstream dataset. They are an exploratory subset, not a random
sample. The prompt is the unit of paired behavioral analysis.

The replay contains 160 additional continuations: 40 prompts under two
conditions and two attention backends. Ten prompts are replay targets;
the other thirty restore the original batch composition. These are not
160 independent target prompts.

## Result files

Paths below are relative to the repository root.

| Record | Files | What it supports |
|---|---|---|
| Effective writer geometry | [summary.csv](../results/summary.csv), six files named CONDITION.writers.json in [results/smoke/](../results/smoke/) | Remaining projection norms relative to baseline, over 53 effective residual-writing matrices |
| Pilot direction identity and overlap | [direction_metadata.json](../results/direction_metadata.json), [recovered_direction_vectors.json](../results/recovered_direction_vectors.json), [direction_geometry.json](../results/smoke/direction_geometry.json) | Recorded FP32 values of the regenerated pilot directions at layers 10 and 15; the S1 comparison with the released layer-10 direction |
| Pilot continuations | Twelve numbered condition files in [results/smoke/](../results/smoke/), [settings.json](../results/smoke/settings.json), [prompt_token_ids.json](../results/smoke/prompt_token_ids.json) | The recorded texts, token arrays, run configuration and common inputs |
| Exploratory behavioral labels | [exploratory_labels.csv](../results/exploratory_labels.csv), [ratings/](../results/ratings/), [rubric](../pain_axis/rubric.md) | Unblinded AI annotations and paired descriptive comparisons; no independent human validation |
| Historical replay | Twenty numbered files and [summary.json](../results/replay/summary.json) in [results/replay/](../results/replay/) | Recorded target-match flags, batch composition and the failed stopping rule |
| Auxiliary inspection | [smoke_inspection.json](../results/smoke_inspection.json), [residual_readouts.json](../results/residual_readouts.json) | Checkpoint coverage, text differences and padded activation readouts |
| Original authors' aggregates | [historical_summary.json](../results/historical_summary.json) | Upstream records and paper-reported counts, not newly scored observations |
| Synthetic checks | [synthetic_geometry.json](../results/synthetic_geometry.json), [tests/](../tests/) | Mathematical examples and a separate random-weight model fixture |

The writer measurements concern effective weight geometry. They do not measure
the percentage of residual activity removed or establish a behavioral effect.
The behavioral comparison is inconclusive. The failed replay stopped the
planned full comparison; the missing historical provenance remains unresolved.
See [Methods](METHODS.md) for the equations and limitations.

## Conditions and units

| Condition key | Intervention |
|---|---|
| baseline | Unedited model |
| s1 | Remove S1 |
| s2 | Remove S2 |
| s1_then_s2 | Remove S1, then S2 |
| s2_then_s1 | Remove S2, then S1 |
| joint | Remove their joint span |

Direction-indexed arrays use the order [S1, S2]. In the pilot these directions
were extracted from the post-block residual stream at zero-based block indices
10 and 15 respectively; the edits cover the model's effective residual writers.

Writer ratios and relative changes in the CSV and JSON files are fractions.
Multiply by 100 to obtain the percentages in the README: a ratio of
0.0022972864 is approximately 0.2297%. The ratio uses each writer's own
unedited projection norm as its denominator; the reported maximum is taken
over all 53 writers.

## Table fields

| File | Fields and meaning |
|---|---|
| results/summary.csv | One row per condition. n_prompts is 10 and n_writers is 53. max_writer_ratio_s1 and max_writer_ratio_s2 are the largest relative projection norms. mean_relative_writer_change is the unweighted mean across writers of the Frobenius-norm weight change divided by the original Frobenius norm. ai_humor_count, ai_coherent_count and ai_relevant_count are sums of the exploratory binary labels. |
| results/exploratory_labels.csv | One row per prompt_id and condition. humor, coherent and relevant are binary rubric scores. rating_provenance is unblinded_AI. |
| results/ratings/ai_unblinded_ratings.csv | One row per anonymous_id, including the upstream prompt text, the model's response continuation and three binary scores. The anonymous IDs do not make these existing ratings blinded. |
| results/ratings/condition_key.csv | Maps each anonymous_id to prompt_id and condition. This is an experimental mapping. Do not give this key or the annotated sheet to a prospective blinded rater. |

The rating procedure and its limits are recorded in
[ratings/method.json](../results/ratings/method.json). In
[ratings/paired.json](../results/ratings/paired.json), each contrast is ordered
from before to after. zero_to_one and one_to_zero count paired changes;
delta is the after-minus-before proportion. The McNemar values are two-sided,
exact, unadjusted exploratory comparisons. The forward-sequential-to-joint
humor comparison has one 1-to-0 change and no 0-to-1 changes, with p = 1.
This does not establish equivalence or absence of an effect.

## Generation and writer records

Pilot batch filenames use a condition key and a zero-based batch index.
Batch 000 has eight rows and batch 001 has two. Each row includes:

| Field | Meaning |
|---|---|
| id, category | Upstream prompt identifier and category |
| prompt_sha256 | SHA-256 of the upstream prompt text |
| generation | Decoded continuation; the prompt itself is excluded |
| generated_ids_with_batch_padding | Generated token IDs, retaining padding after a row has finished |
| fixed_avg_abs_projection_with_batch_padding | Absolute projections on [S1, S2], averaged over post-layer hook calls during generation, including padded steps |

The padded activation readout is a separate diagnostic from the writer ratio.
It should not be interpreted as a clean per-token endpoint or a numerical
replication of the original historical probe.

Each writer file has 53 entries: embed, o_proj_0 through o_proj_25 and
down_proj_0 through down_proj_25. direction_norm and relative_to_original
are two-element arrays in [S1, S2] order. relative_writer_change is the
fractional Frobenius-norm change for that writer. clamped_gain_count records
the number of normalization-gain entries with absolute value below 1e-4;
the audit itself measures the actual learned gains.

The archived smoke runner explicitly uses eager attention. The later
pain_axis/run.py permits backend selection and defaults to eager. The replay
compares default attention, recorded as SDPA, with eager.

## Replay and auxiliary records

Replay filenames contain the backend, condition and original author-batch
index. The included indices are 000, 002, 005, 007 and 010. Each file has
eight rows. The target flag distinguishes the ten scored prompts from
companion rows; historical_exact_match is present for targets only.

The archived target counts are 5/10 baseline and 4/10 sequential for
default/SDPA, and 7/10 baseline and 4/10 sequential for eager. The stopping
rule required all ten targets to match in both default-backend conditions.
The offline verifier counts the stored match flags; it does not recreate
the missing historical environment.

The replay checkpoint JSON files are semantic transcriptions recovered from
file previews. Their archived hashes identify the included records; they
do not assert byte identity with the original runtime files.

Two auxiliary distinctions matter:

- results/historical_summary.json contains original upstream observations
  and paper-reported counts. Do not pool them with this ten-prompt pilot.
- results/synthetic_geometry.json uses a random-weight Gemma model. Its
  released steering bundle has both directions at layer 10, whereas the
  pilot uses layers 10 and 15. Its cosine, 0.606176, and other fixture
  measurements describe that separate setup. The pilot cosine is 0.296582.

The inspection script does not load ratings, so its archived
`ratings_available: false` field is not the status of the completed pilot.
The completed labels and their provenance are in `results/ratings/`; the
inspection checkpoint remains unchanged.

## What can be checked locally

From the repository root, the standard-library checks are:

    python3 scripts/verify_evidence.py
    python3 -m unittest discover -s tests -v

The first checks the historical artifact manifest, recomputes writer ratios
from archived norms and verifies specified relationships among responses,
labels and replay records. The unit tests include small projection examples
and checks for the release tooling. Neither command runs the pretrained model.

[provenance.json](../provenance.json) identifies 61 original archived artifacts
and a separate supplemental record for the recovered direction values. The
documentation, verification support and derived summary tables are separate
from that historical manifest. Two original release transformations are recorded:
direction arrays were omitted from direction_metadata.json, and a local
run directory was omitted from smoke_inspection.json.

The supplemental [recovered_direction_vectors.json](../results/recovered_direction_vectors.json)
preserves the source archive's 2,304 FP32 values for each regenerated pilot
direction. It is byte-identical to that source JSON. These numeric values can
be exported to a new tensor file with
[scripts/materialize_directions.py](../scripts/materialize_directions.py);
the new file will have its own hash. Recovery of the numeric values does not
recover the original serialized .pt file's bytes or its recorded hash.

The release includes generated continuations, upstream prompt text and
exploratory per-answer labels. It omits model weights, the original serialized
pilot tensor file and extraction activation caches. The original paper's
historical S2 layer-15 artifact also remains unavailable; it is distinct from
the recovered regenerated pilot direction. See [Reproduction](REPRODUCE.md)
for rerun boundaries and [third-party notices](../THIRD_PARTY_NOTICES.md)
for attribution and the current licensing scope.
