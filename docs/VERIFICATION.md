# Verification

The checks below were recorded on 24 November 2026, using Python 3.12.14. They verify the supplied archive and supporting tools. No new pretrained-model inference was run during those checks.

## Recorded checks

| Check | Recorded result |
|---|---|
| Full archive verifier, including optimised Python | Passed |
| Test suite without PyTorch | 41 passed; 2 optional tests skipped |
| Test suite with PyTorch 2.8.0+cpu | All 43 passed |
| Pinned upstream projection functions on CPU fixtures | Passed |
| Export of recovered direction values and reload through the original reader | Exact float32 values preserved |
| Smoke and extraction command checks on pinned sources | Passed in dry-run mode; no inference |
| Original archived files | All 61 preserved byte for byte |
| Replay target texts compared with pinned upstream outputs | All 40 recorded match flags confirmed |

The full verifier checks the historical file hashes, the recovered direction values, 60 pilot responses, 318 writer records, 60 ratings, 18 paired-statistic objects, descriptive activation readouts and 160 replay records. Tests use modified temporary copies to check that inconsistent or incomplete records are rejected.

The GitHub Actions workflow runs the offline checks on Python 3.11 and 3.12. Hosted CI status is separate from the recorded local results above.

## Direction values and operator checks

The recovered JSON contains two 2,304-element float32 arrays from the pilot. Their content hash and source identifiers are in [provenance.json](../provenance.json). Exporting these values creates a new tensor file with a new hash; the historical tensor-file bytes have not been recovered.

The CPU checks exercise the actual pinned upstream functions with overlapping directions, learned gains, embeddings, finite precision, different direction orders and restoration. A zero-gain fixture exposes a limitation of the implementation's gain floor. This fixture is a numerical check, not a new pretrained-model result. See [Methods](METHODS.md).

## Using the historical scripts

The original runners skip any batch whose output file already exists. The wrapper checks the inputs and requires a fresh output directory. Use it for direction extraction, pilot runs or replay diagnostics. The 100-prompt comparison remains unrun.

The original rating helper has permissive duplicate-key handling and can leave a partial export after a file collision. The supplied labels pass the separate verifier. A new rating study needs a separately checked export procedure.

Some archived comments describe a planned blinded or human-rating workflow. The ratings actually completed for this pilot were unblinded AI annotations with zero independent human raters. The replay script's use of "preregistered" refers to the recorded pre-run stopping rule, not an external preregistration. These historical files remain unchanged.

## Limits

The behavioural comparison remains inconclusive: ten deterministic prompts, one response per condition, unblinded AI labels and no competence panel. The original paper's precise S2 artifact and historical execution environment remain unresolved, and exact replay failed.

Replay checkpoint files were recovered as semantic JSON transcriptions from historical previews. Their hashes identify the supplied archive, not the original runtime bytes. The recorded CUDA environment was not freshly installed or tested during the checks above.

The broader source archive is private. Its repository identifier is withheld in this distribution; the original manifest remains with the author. Source commits, relative paths and content hashes are retained under the explicit label `private-source-archive`.

See [Results](RESULTS.md) for the data dictionary and [Reproduction](REPRODUCE.md) for executable commands.
