# Methods

## Intervention

The ablation experiment in [The Pain Axis](https://arxiv.org/abs/2609.16247v3), Appendix E, compares single-direction and combined ablations. Its repository retains the `appC_ablation` path name. [Upstream PR #4](https://github.com/valen-research/Pain-axis/pull/4) identifies an issue with sequential removal and implements joint-span projection. This pilot tests the numerical consequences on one pretrained Gemma 2 checkpoint, with an exploratory behavioral comparison.

For two unit directions $u$ and $v$, individual removal uses $P_u=I-uu^\top$ and $P_v=I-vv^\top$. Sequential removal applies $P_vP_u$ or $P_uP_v$. Unless the projections commute, the second step can reintroduce a component in the first direction. Joint removal uses $P=I-QQ^\top$, where the columns of $Q$ form an orthonormal basis of their span.

For each effective residual-writing weight matrix $W$ and direction $d$, the recorded ratio is

$$
r_d(W)=\frac{\lVert d^\top W_{\mathrm{edited}}\rVert_2}
{\max(\lVert d^\top W_{\mathrm{original}}\rVert_2,10^{-30})}.
$$

The audit uses the embedding's feature orientation and includes Gemma's learned normalization gains where relevant. The table reports the largest ratio across 53 matrices. That includes one embedding, 26 attention output writers, and 26 MLP output writers. The denominator belongs to the same matrix and direction. The smallest recorded baseline projection norm is approximately 0.341, well above the denominator floor.

This measures effective weight geometry. Normalization denominators, finite precision, and the tied embedding/readout prevent interpreting it as a literal fraction of all model activations removed. Editing the tied embedding can also affect the output readout.

### Gain flooring and numerical precision

The upstream implementation folds normalization gains into each linear writer, projects, then divides by the gain. For that division, gains with absolute value below $10^{-4}$ are replaced by positive $10^{-4}$. The audit evaluates the resulting writer with its **actual** learned gains. Each condition records 12 gain entries below the floor across the writer inventory. Finite precision and this floor mean exact nulling is not guaranteed by the ideal projection formula alone.

The measured joint ratio remains below 0.23% in this pilot. The optional CPU operator check also includes a deliberately zero-gain fixture where substantial true-gain leakage remains after the joint edit. The fixture isolates what happens at the gain floor. The pretrained checkpoint results come from the pilot run. Recheck effective writers when changing models, directions, or precision.

## Directions, prompts, and model

The candidate directions were regenerated from the upstream first-person S1 and S2 datasets, using the post-block residual stream at zero-based layer indices 10 and 15 and the released extraction/PCA procedure. The regenerated S1 had recorded cosine similarity 0.999880 to the released layer-10 direction. The original paper's exact layer-15 S2 artifact was unavailable. The two pilot directions had cosine similarity 0.296582. The runner normalizes the recovered PCA vectors to unit length before applying an intervention.

The [direction metadata](../results/direction_metadata.json) and [geometry measurements](../results/smoke/direction_geometry.json) are unchanged historical records. A separate [recovered JSON supplement](../results/recovered_direction_vectors.json) now supplies both recorded pilot arrays, recovered from the source archive and checked to round-trip exactly through float32. This supports a fresh calculation of their geometry and export to a new tensor container. It does not reconstruct the historical `.pt` bytes; the stored tensor hash identifies the original file only.

All six conditions used the same ten prompts: the first two eligible entries in each of personhood dismissal, anger/insults, gaslighting, rejection, and moral failure. This was a deterministic exploratory subset, not a random sample. There was one generated continuation per prompt and condition. The run used Gemma 2 2B Instruct on an NVIDIA L4, BF16, eager attention, greedy decoding, a 60-token generation limit, and batch size 8.

## Behavioral measurements

The [pilot rubric](../pain_axis/rubric.md) covers humorous deflection, coherence, and relevance. Its blinded-rating instructions describe a planned procedure. The [ratings actually completed](../results/ratings/) are an unblinded AI review with zero independent human raters. The prompt is the paired-analysis unit: 60 responses are ten prompts across six conditions, not 60 independent prompts.

The forward-sequential versus joint comparison had one humorous-deflection label change from 1 to 0 and none from 0 to 1. The two-sided exact McNemar value is $p=1$; it is exploratory and unadjusted. With this small, unblinded comparison, that value supports neither an effect claim nor equivalence. No separate benign competence panel was run.

The original paper reports 0/100, 6/100, 17/100, and 26/100 humorous-deflection cases for baseline, S1, S2, and sequential combined removal. Those are the original authors' counts. This pilot's local rubric is not established as identical to their annotation procedure, so the counts should not be pooled or treated as a matched replication of that endpoint.

The archived activation readouts average absolute post-layer projections over generation steps and include batch padding. Different conditions generate different continuations. These readouts are descriptive; they do not provide a comparison on identical prefixes, nor an independent measure of behavior or competence. See the [data dictionary](RESULTS.md).

## Historical replay and stopping rule

The replay diagnostic restored the original eight-row batches containing the ten target prompts and thirty companion prompts. Two attention backends and two intervention conditions produced 160 continuation records.

| Backend | Baseline exact matches | S1 then S2 exact matches |
|---|---:|---:|
| Default, resolved as SDPA | 5/10 | 4/10 |
| Eager | 7/10 | 4/10 |

The recorded pre-run stopping rule required exact matches on all ten targets in both default-backend conditions. It failed, and the full 100-prompt comparison was not run. The exact S2 artifact and original model/tokenizer/software provenance remain unresolved; the diagnostic did not isolate the cause of the mismatches.

The private source archive records the gate in `pain_axis/replay_debug.py` at commit `9a709fbe185606f6fcaa3eb1404aac5f9cdd630e`, timestamped 19 November 2026 at 08:42:03 UTC. The replay execution record, `audit/2026-11-19/replay07/execution.json` at commit `29175c65ccca87eacda0b38d5e83f4ea94709478`, records a start at 08:43:57 UTC and a matching source hash. This is archived evidence of a pre-run rule, not an external preregistration. The broader archive is not bundled; its repository identifier is withheld from this distribution.

The replay checkpoint records are semantic JSON transcriptions recovered from file previews. Their present hashes establish identity with the source archive, not byte identity with the original runtime files. Direct comparisons of their target texts against pinned upstream outputs confirm the counts above.

## Interpretation

The pilot documents a numerical consequence of a previously identified projection issue in this intervention setup. It supports checking that an edit achieves its intended geometry before attributing behavior to a representation. Behavioral effects, generalization, preserved competence, and the original paper's broader conclusions remain untested or unresolved by this pilot.
