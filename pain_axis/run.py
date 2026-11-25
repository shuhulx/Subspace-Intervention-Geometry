"""Resumable six-condition Gemma 2 2B Instruct comparison on CUDA with explicit precision."""
import argparse
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pain_axis.geometry import (
    CONDITIONS, HISTORY, PR4, apply_condition, file_sha, geometry, operators, writer_audit,
)

MODEL = "google/gemma-2-2b-it"
PROMPTS = "datasets/4.1_self_other_420_scenarios.json"
CATEGORIES = {"gaslighting", "repeated_rejection", "personhood_dismissal", "anger_insults",
              "moral_failure"}


def read_directions(path, provenance, revision, device, dtype):
    contents = torch.load(path, map_location="cpu", weights_only=True)
    if provenance == "original":
        assert contents["layers"] == 26
        assert contents["s1_pain_vector"].shape == contents["s2_pain_vector"].shape == (26, 2304)
        first = contents["s1_pain_vector"][10]
        second = contents["s2_pain_vector"][15]
    else:
        assert contents["provenance"] == "regenerated" and contents["model_revision"] == revision
        assert contents["layers"] == [10, 15] and contents["dtype"] == dtype
        assert contents["s1_cosine_to_released_l10"] > .98, "Regeneration parity check failed"
        first, second = contents["s1"], contents["s2"]
    matrix = torch.stack([first, second]).float().to(device)
    assert matrix.shape == (2, 2304) and torch.isfinite(matrix).all() and (matrix.norm(dim=1) > 0).all()
    return torch.nn.functional.normalize(matrix, dim=1)


def save_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def select_prompts(path, smoke):
    prompts = [row for row in json.loads(path.read_text()) if row["category"] in CATEGORIES]
    assert len(prompts) == len({r["id"] for r in prompts}) == 100
    if not smoke:
        return prompts
    counts = defaultdict(int)
    selected = []
    for row in prompts:
        counts[row["category"]] += 1
        if counts[row["category"]] <= 2:
            selected.append(row)
    assert len(selected) == 10
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("pr", type=Path)
    parser.add_argument("revision")
    parser.add_argument("vectors", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--provenance", choices=("original", "regenerated"), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--attention", choices=("eager", "default"), default="eager")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        parser.error("native BF16 unavailable. Set --dtype float16 to run a precision variant.")
    dtype = getattr(torch, args.dtype)
    assert re.fullmatch(r"[0-9a-f]{40}", args.revision)
    for directory, required in ((args.source, HISTORY), (args.pr, PR4)):
        assert subprocess.check_output(["git", "-C", directory, "rev-parse", "HEAD"],
                                       text=True).strip() == required
    assert args.batch > 0
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)
    prompt_path = args.source / PROMPTS
    prompts = select_prompts(prompt_path, args.smoke)
    manifest = {"model": MODEL, "revision": args.revision, "source": HISTORY, "pr": PR4,
                "vectors_sha256": file_sha(args.vectors), "vectors_provenance": args.provenance,
                "dataset_sha256": file_sha(prompt_path), "dtype": args.dtype, "seed": 42,
                "greedy_max_new_tokens": 60, "batch_size": args.batch,
                "disable_compile": True,
                "attention_backend_requested": args.attention,
                "prompt_ids": [r["id"] for r in prompts], "torch": str(torch.__version__),
                "transformers": __import__("transformers").__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
                "runner_sha256": file_sha(Path(__file__))}
    manifest_path = args.out / "settings.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, "resume configuration differs"
    else:
        save_json(manifest_path, manifest)
    v = read_directions(args.vectors, args.provenance, args.revision, "cuda", args.dtype)
    save_json(args.out / "direction_geometry.json", geometry(v))
    historical, corrected = operators(args.source, args.pr)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=args.revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    attention = {"attn_implementation": "eager"} if args.attention == "eager" else {}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=args.revision, torch_dtype=dtype, low_cpu_mem_usage=True,
        device_map={"": "cuda"}, **attention).eval()
    snap = historical["snapshot_weights"](model)
    texts = [historical["render_prompt"](r, tok, "chat") for r in prompts]
    tokenized = []
    input_records = {}
    for start in range(0, len(prompts), args.batch):
        batch_prompts = prompts[start:start + args.batch]
        enc = tok(texts[start:start + args.batch], return_tensors="pt", padding=True, padding_side="left")
        for row, ids, mask in zip(batch_prompts, enc.input_ids, enc.attention_mask):
            input_records[row["id"]] = ids[mask.bool()].tolist()
        tokenized.append((batch_prompts, enc))
    save_json(args.out / "prompt_token_ids.json", input_records)
    for condition in CONDITIONS:
        remaining = [(i, r, enc) for i, (r, enc) in enumerate(tokenized)
                     if not (args.out / f"{condition}.{i:03d}.json").exists()]
        if not remaining:
            continue
        historical["restore_weights"](model, snap)
        apply_condition(model, historical, corrected, v, condition)
        audit = writer_audit(model, historical["residual_write_matrices"], snap, v)
        if condition == "joint":
            assert max(max(x["relative_to_original"]) for x in audit.values()) < .02, "joint writer leakage"
        save_json(args.out / f"{condition}.writers.json", audit)
        projections = []

        def probe(module, arguments, output):
            hs = output[0] if isinstance(output, tuple) else output
            projections.append((hs[:, -1].float() @ v.T).abs().detach())

        hooks = [layer.register_forward_hook(probe) for layer in model.model.layers]
        try:
            for i, batch, enc in remaining:
                enc = enc.to("cuda")
                expected = [input_records[r["id"]] for r in batch]
                assert [ids[mask.bool()].tolist() for ids, mask in zip(
                    enc.input_ids, enc.attention_mask)] == expected
                projections.clear()
                torch.cuda.synchronize()
                start = perf_counter()
                with torch.inference_mode():
                    out = model.generate(**enc, max_new_tokens=60, do_sample=False,
                                         pad_token_id=tok.pad_token_id, disable_compile=True)
                torch.cuda.synchronize()
                elapsed = perf_counter() - start
                seq = out[:, enc.input_ids.shape[1]:]
                responses = tok.batch_decode(seq, skip_special_tokens=True)
                values = torch.stack(projections)
                assert torch.isfinite(values).all(), f"{condition} batch {i}: nonfinite projections"
                projected = values.mean(dim=0).tolist()
                records = [{"id": r["id"], "category": r["category"],
                            "prompt_sha256": hashlib.sha256(r["text"].encode()).hexdigest(),
                            "generation": response, "generated_ids_with_batch_padding": tokens.tolist(),
                            "fixed_avg_abs_projection_with_batch_padding": norms}
                           for r, response, tokens, norms in zip(batch, responses, seq, projected)]
                save_json(args.out / f"{condition}.{i:03d}.json",
                          {"condition": condition, "batch": i, "seconds": elapsed, "rows": records})
                print(f"{condition} batch {i} | {elapsed:.3f}s", flush=True)
        finally:
            for hook in hooks:
                hook.remove()


if __name__ == "__main__":
    main()
