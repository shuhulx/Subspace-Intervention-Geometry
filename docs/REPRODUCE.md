# Reproduction

Start with the offline archive checks. To test the projection functions themselves, run the optional CPU fixture. A fresh model run needs the pinned inputs and GPU setup below.

## 1. Verify the supplied records

From the repository root, with Python 3.11 or later and Git:

```bash
python3 scripts/verify_evidence.py
python3 -m unittest discover -s tests -v
```

The main verifier and the default checks use the standard library. They need no model, API key, network, or GPU. Tensor-export round-trip tests are explicitly skipped if PyTorch is absent. `--summary-only` performs a smaller derived-table check and does not verify the archive.

Full verification checks the 61 original archived artifacts against their content hashes, the recovered numeric-vector supplement, 60 pilot responses, 318 writer records, 60 ratings, all 18 paired-statistic objects, descriptive readout means, and 160 replay records. It rejects unexpected evidence batches and checks types, dimensions, finite values, and cross-file correspondence. The [results guide](RESULTS.md) defines each field and its limitations.

The replay checks establish agreement between archived flags, mismatch lists, counts, and the recorded stopping rule. The release audit additionally compared target generation strings directly with the pinned upstream historical output file. Offline checks cannot recover unbundled model weights, original tensor-container bytes, or historical execution conditions.

## 2. Optional CPU projection check

The three original geometry examples are mathematical illustrations. `scripts/check_upstream.py` additionally executes the exact upstream functions used by the pilot on a small CPU writer fixture. It checks source-file hashes before extracting those functions, including embedding orientation, learned gains, finite precision, sequential order, joint removal, and restoration. It also exposes the gain-floor boundary described in [Methods](METHODS.md).

This requires PyTorch and the two pinned upstream source files/checkouts below. It does not load a pretrained model. The [recorded checks](VERIFICATION.md) used PyTorch 2.8.0+cpu, separately from the historical CUDA environment.

For an isolated Linux CPU environment:

```bash
python3 -m venv .venv-cpu
. .venv-cpu/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 numpy
```

After obtaining the pinned sources in section 3:

```bash
python3 scripts/check_upstream.py external/pain-original external/pain-pr4
```

## 3. Pinned upstream inputs

Run these once before optional CPU integration or GPU work:

```bash
mkdir -p external
git clone --filter=blob:none --no-checkout https://github.com/valen-research/Pain-axis.git external/pain-original
git -C external/pain-original checkout --detach 213fb6c89c96004dbe93d1704c636ad8c9d801f3

git clone --filter=blob:none --no-checkout https://github.com/valen-research/Pain-axis.git external/pain-pr4
git -C external/pain-pr4 fetch origin pull/4/head
git -C external/pain-pr4 checkout --detach a0a955052852f31afbc2b8ddfbd508d60a7b432b
```

Use the exact commits. If a commit or required data artifact is unavailable, resolve that missing input before running the experiment. The local helpers extract selected upstream functions; they do not execute the unrelated upstream entrypoints that clear model caches.

## 4. Recorded GPU environment

| Item | Recorded setting |
|---|---|
| Python | 3.12 |
| Model | `google/gemma-2-2b-it` |
| Model revision | `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` |
| Hardware and precision | NVIDIA L4, native BF16, CUDA 12.8 |
| PyTorch | 2.8.0+cu128 |
| Transformers | 4.55.4 |
| TransformerLens, extraction only | 2.16.1 |
| Generation | Seed 42, greedy, 60 new tokens, batch size 8, compilation disabled |
| Pilot attention | Eager, explicitly selected in the frozen smoke runner |
| Replay attention | Default resolved as SDPA, and a separate eager run |

[requirements-recorded.txt](../requirements-recorded.txt) records the historical package versions. It is not a complete lockfile or an installer validated on a new GPU machine. Prepare an isolated compatible environment and review those versions. The release audit did not reinstall the CUDA environment or repeat pretrained inference.

`pain_axis/smoke_runner.py` is the exact archived runner. `pain_axis/run.py` is a later version that adds attention-backend selection and also defaults to eager. All original `pain_axis/` files retain their archived bytes. Their historical assertions and filename-based resume behavior are reasons to use the fresh-run wrapper below.

## 5. Use the recovered pilot directions

The numerical values of the two pilot directions are included in `results/recovered_direction_vectors.json`. With CPU PyTorch available, export a new tensor container:

```bash
python3 scripts/materialize_directions.py local-runs/recovered-directions.pt
```

The helper validates the source JSON and refuses to overwrite an existing output. It preserves the float32 vector values and the metadata needed by the frozen runner. The new file has its own SHA-256; it must not be claimed to match the historical `.pt` hash. The original paper's unavailable S2 artifact is a different object and remains unavailable.

## 6. Optional fresh ten-prompt GPU run

Accept the Gemma model terms and authenticate with Hugging Face before downloading the gated checkpoint. The commands below require CUDA with native BF16 support and the recorded model/runtime dependencies.

```bash
python3 scripts/run_fresh.py smoke external/pain-original external/pain-pr4 local-runs/recovered-directions.pt local-runs/smoke-001 --dry-run
python3 scripts/run_fresh.py smoke external/pain-original external/pain-pr4 local-runs/recovered-directions.pt local-runs/smoke-001
python3 -m pain_axis.inspect_run local-runs/smoke-001 external/pain-original local-runs/smoke-001/inspection.json
```

The dry run checks pinned checkout revisions, actual consumed source files, input paths, and a new output location. It starts no inference and creates no files. It does not validate a CUDA installation or deserialize the direction tensor. The actual launch checks runtime availability, creates the output directory exclusively, saves a preflight receipt, and delegates to the unchanged smoke runner with `--smoke`.

Every launch requires a new directory. After interruption or failure, inspect the saved files and use another directory for a rerun. The wrapper deliberately disables automatic resume because the frozen runner skips existing batch filenames without validating their contents. Run without Python optimization (`-O` or `PYTHONOPTIMIZE`), which would remove checks in the frozen scripts.

A fresh generation is a new experiment with the recorded pilot directions. Even with the same vector values, output identity depends on model, software, hardware, and execution details.

### Optional independent direction extraction

To rerun the extraction procedure instead of using the recovered values:

```bash
python3 scripts/run_fresh.py regenerate external/pain-original local-runs/extraction-001 --dry-run
python3 scripts/run_fresh.py regenerate external/pain-original local-runs/extraction-001
```

This requires TransformerLens, the extraction dataset, the released steering artifact, and GPU resources. It writes `local-runs/extraction-001/directions.pt` plus extraction activation caches. A fresh extraction need not return numerically identical directions. Keep those outputs separate from the recorded pilot values.

### Optional replay diagnostic

After a complete fresh smoke run using the same vector file:

```bash
python3 scripts/run_fresh.py replay external/pain-original external/pain-pr4 local-runs/recovered-directions.pt local-runs/smoke-001 local-runs/replay-001 --dry-run
python3 scripts/run_fresh.py replay external/pain-original external/pain-pr4 local-runs/recovered-directions.pt local-runs/smoke-001 local-runs/replay-001
```

This runs the two recorded replay conditions and two backends. It does not launch the full 100-prompt panel. The historical replay gate failed; a new run does not resolve the original provenance gap by itself.

## Archive and release boundaries

The 61 original artifacts remain byte-identical to the initial release, including the two documented transformations from the source archive: omitted arrays in `direction_metadata.json` and an omitted local path in `smoke_inspection.json`. The recovered full vector JSON is a separate supplement, with its own source locator and hash. [provenance.json](../provenance.json) retains each source commit, relative path and content hash. The private source repository's identifier is withheld and represented by the explicit label `private-source-archive`; the author retains the original manifest.

The frozen `pain_axis/blind.py` records the original rating/export workflow. Its general-purpose paired-input path does not reject every duplicate key, and its export can leave an incomplete output pair after a file collision. Use the release verifier for the supplied ratings. For a new blinded rating study, create a separate validated export and keep its condition key from the raters; do not reuse the published unblinded labels as independent ratings.

Model weights, extraction activation caches, the original serialized direction file, and the original paper's exact historical S2 artifact are not bundled. Licensing and third-party boundaries are recorded in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
