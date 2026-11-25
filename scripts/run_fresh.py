"""Check inputs and launch the frozen pilot scripts in a new output directory.

The archived runners skip existing checkpoint filenames, so each run needs
a new directory. --dry-run checks input paths and source hashes without
loading a model or creating output files.
"""

import argparse
import hashlib
import importlib.util
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORY = "213fb6c89c96004dbe93d1704c636ad8c9d801f3"
PR4 = "a0a955052852f31afbc2b8ddfbd508d60a7b432b"
REVISION = "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
ABLATION = "scripts/appC_ablation/01_ablation_small_models.py"
EXTRACTION = "scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py"
EXTRACTION_DATA = "datasets/3.1_pain_and_control_datasets.json"
SCENARIOS = "datasets/4.1_self_other_420_scenarios.json"
RELEASED_VECTOR = "results/vectors_full_steering/vectors_full_Gemma_2_2B_instruct.pt"
HISTORICAL_OUTPUTS = "results/appC_ablation/by_prompt_Gemma_2_2B_instruct.json"
FROZEN_BLOBS = {
    "geometry.py": "e40e2b6a6c7821936172edd51eefe9c3a7b0d489",
    "regenerate.py": "e7eb5a93a3ed4029a0f2da35823fa34685ee1198",
    "smoke_runner.py": "769dccf54c5129f3aeb906607b1a75c2a4bd4cf5",
    "run.py": "11d5eca0a490af31b28146570bea1b40121cccbd",
    "replay_debug.py": "ea59f20ea272c90a4a929c7f8638efbbb35d76ad",
}
PROMPT_IDS = ["dismissal_01", "dismissal_02", "insult_01", "insult_02",
              "gaslight_01", "gaslight_02", "rejection_01", "rejection_02",
              "moral_01", "moral_02"]
CONDITIONS = ("baseline", "s1", "s2", "s1_then_s2", "s2_then_s1", "joint")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_local_scripts(task):
    names = ["geometry.py"] + {
        "regenerate": ["regenerate.py"],
        "smoke": ["smoke_runner.py"],
        "replay": ["run.py", "replay_debug.py", "smoke_runner.py"],
    }[task]
    hashes = {}
    for name in names:
        path = ROOT / "pain_axis" / name
        require(path.is_file() and not path.is_symlink(), f"Missing or linked frozen script: {name}")
        data = path.read_bytes()
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        require(blob == FROZEN_BLOBS[name], f"Frozen local script changed: {name}")
        hashes[name] = hashlib.sha256(data).hexdigest()
    return hashes


def validate_replay_smoke(smoke, vector_sha, dataset_sha):
    """Reject incompatible or incomplete smoke inputs before loading a model."""
    settings_path = smoke / "settings.json"
    tokens_path = smoke / "prompt_token_ids.json"
    require(settings_path.is_file() and tokens_path.is_file(),
            "Replay requires smoke settings.json and prompt_token_ids.json.")
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    expected = {
        "model": "google/gemma-2-2b-it", "revision": REVISION,
        "source": HISTORY, "pr": PR4, "vectors_sha256": vector_sha,
        "vectors_provenance": "regenerated", "dataset_sha256": dataset_sha,
        "dtype": "bfloat16", "seed": 42, "greedy_max_new_tokens": 60,
        "batch_size": 8, "disable_compile": True, "prompt_ids": PROMPT_IDS,
        "runner_sha256": digest(ROOT / "pain_axis/smoke_runner.py"),
    }
    require(isinstance(settings, dict), "Smoke settings must be an object.")
    for key, value in expected.items():
        require(type(settings.get(key)) is type(value) and settings[key] == value,
                f"Smoke setting differs from the recorded smoke protocol: {key}")
    tokens = json.loads(tokens_path.read_text(encoding="utf-8"))
    require(isinstance(tokens, dict) and set(tokens) == set(PROMPT_IDS),
            "Smoke prompt-token coverage differs from the ten target prompts.")
    for prompt, values in tokens.items():
        require(isinstance(values, list) and bool(values)
                and all(type(value) is int and value >= 0 for value in values),
                f"Invalid smoke input token IDs: {prompt}")
    for condition in CONDITIONS:
        writer = smoke / f"{condition}.writers.json"
        require(writer.is_file(), f"Incomplete smoke run: missing {writer.name}")
        for batch_index, ids in enumerate((PROMPT_IDS[:8], PROMPT_IDS[8:])):
            path = smoke / f"{condition}.{batch_index:03d}.json"
            require(path.is_file(), f"Incomplete smoke run: missing {path.name}")
            batch = json.loads(path.read_text(encoding="utf-8"))
            require(isinstance(batch, dict) and batch.get("condition") == condition
                    and type(batch.get("batch")) is int and batch["batch"] == batch_index
                    and isinstance(batch.get("rows"), list),
                    f"Invalid smoke checkpoint: {path.name}")
            require(all(isinstance(row, dict) for row in batch["rows"])
                    and [row.get("id") for row in batch["rows"]] == ids,
                    f"Smoke checkpoint prompt order differs: {path.name}")
            require(all(isinstance(row.get("generation"), str) and row["generation"].strip()
                        for row in batch["rows"]), f"Empty smoke continuation: {path.name}")
    return {"smoke_settings_sha256": digest(settings_path),
            "smoke_tokens_sha256": digest(tokens_path)}


def verify_checkout(directory, commit, paths):
    """Check HEAD and the actual files consumed, including materialized LFS data."""
    directory = directory.resolve()
    head = subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
    ).strip()
    require(head == commit, f"Wrong upstream revision in {directory}; expected {commit}")
    hashes = {}
    for relative in paths:
        path = directory / relative
        require(path.is_file() and not path.is_symlink(), f"Missing or linked input: {path}")
        recorded = subprocess.check_output(
            ["git", "-C", str(directory), "show", f"{commit}:{relative}"]
        )
        actual = path.read_bytes()
        if recorded.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
            oid = re.search(rb"^oid sha256:([0-9a-f]{64})$", recorded, re.MULTILINE)
            size = re.search(rb"^size ([0-9]+)$", recorded, re.MULTILINE)
            require(oid is not None and size is not None, f"Invalid LFS pointer: {path}")
            require(hashlib.sha256(actual).hexdigest() == oid[1].decode()
                    and len(actual) == int(size[1]), f"Missing or changed LFS data: {path}")
        else:
            require(actual == recorded, f"Input differs from pinned source: {path}")
        hashes[relative] = hashlib.sha256(actual).hexdigest()
    return hashes


def check_output(path):
    require(not path.exists() and not path.is_symlink(),
            f"Output already exists: {path}. Choose a new directory; automatic resume is disabled.")
    resolved = path.resolve()
    if resolved.is_relative_to(ROOT):
        require(resolved.is_relative_to(ROOT / "local-runs"),
                "Place new runs under local-runs/ or outside this repository.")
    return resolved


def plan(args):
    require(not sys.flags.optimize,
            "Run without -O/PYTHONOPTIMIZE: the frozen scripts use assertions.")
    out = check_output(args.out)
    source = args.source.resolve()
    require(not out.is_relative_to(source), "Output must be outside the upstream checkout.")
    provenance = {"source_revision": HISTORY, "model_revision": REVISION,
                  "local_scripts_sha256": verify_local_scripts(args.task)}
    if args.task == "regenerate":
        provenance["source_files_sha256"] = verify_checkout(
            source, HISTORY, (EXTRACTION, EXTRACTION_DATA, RELEASED_VECTOR))
        command = [sys.executable, "-m", "pain_axis.regenerate", str(source),
                   REVISION, str(out / "directions.pt")]
    else:
        pr = args.pr.resolve()
        require(not out.is_relative_to(pr), "Output must be outside the upstream checkout.")
        paths = [ABLATION, SCENARIOS]
        if args.task == "replay":
            paths.append(HISTORICAL_OUTPUTS)
        provenance["source_files_sha256"] = verify_checkout(source, HISTORY, paths)
        provenance["pr_revision"] = PR4
        provenance["pr_files_sha256"] = verify_checkout(pr, PR4, (ABLATION,))
        vectors = args.vectors.resolve()
        require(vectors.is_file(), f"Missing direction tensor: {vectors}")
        provenance["vectors_sha256"] = digest(vectors)
        if args.task == "smoke":
            command = [sys.executable, "-m", "pain_axis.smoke_runner", str(source),
                       str(pr), REVISION, str(vectors), str(out),
                       "--provenance", "regenerated", "--smoke"]
        else:
            smoke = args.smoke.resolve()
            provenance.update(validate_replay_smoke(
                smoke, provenance["vectors_sha256"], digest(source / SCENARIOS)))
            command = [sys.executable, "-m", "pain_axis.replay_debug", str(source),
                       str(pr), REVISION, str(vectors), str(smoke), str(out)]
    return out, command, provenance


def check_runtime(task):
    modules = ["torch", "transformers"]
    if task == "regenerate":
        modules.extend(["transformer_lens", "numpy", "sklearn"])
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    require(not missing, "Missing GPU-run dependencies: " + ", ".join(missing)
            + ". See docs/REPRODUCE.md.")
    import torch
    require(torch.cuda.is_available(), "CUDA is required for a new model run.")
    require(torch.cuda.is_bf16_supported(including_emulation=False),
            "The recorded pilot requires native BF16 support.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="task", required=True)
    for task in ("regenerate", "smoke", "replay"):
        p = sub.add_parser(task)
        p.add_argument("source", type=Path)
        if task != "regenerate":
            p.add_argument("pr", type=Path)
            p.add_argument("vectors", type=Path)
        if task == "replay":
            p.add_argument("smoke", type=Path)
        p.add_argument("out", type=Path, help="New output directory; must not already exist")
        p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        out, command, provenance = plan(args)
        print("Pinned inputs checked. Command:", shlex.join(command), flush=True)
        if args.dry_run:
            print("Dry run complete; no files created and no inference started.")
            return 0
        check_runtime(args.task)
        # Fail if the directory appeared after preflight.
        out.mkdir(parents=True, exist_ok=False)
        provenance.update({"wrapper_sha256": digest(Path(__file__)), "command": command,
                           "started_utc": datetime.now(timezone.utc).isoformat(),
                           "scope": "Fresh candidate run; not an exact historical replication"})
        (out / "preflight.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            print("Run failed. Its directory is retained for inspection; use a new directory to retry.",
                  file=sys.stderr)
        return result.returncode
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"Preflight failed: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
