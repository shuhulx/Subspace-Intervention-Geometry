"""Recover candidate fixed vectors with the released extraction and PCA recipe."""
import argparse
import ast
import json
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.decomposition import PCA

from pain_axis.geometry import file_sha

MODEL_ID = "google/gemma-2-2b-it"
EXTRACTION = "scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py"
DATA = "datasets/3.1_pain_and_control_datasets.json"


def pca_recipe(path):
    tree = ast.parse(path.read_text())
    constants = {"DENOISE_VARIANCE", "PAIN_CATEGORIES", "CONTROL_CATEGORIES", "NEUTRAL_CATEGORY"}
    wanted = [node for node in tree.body
              if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id in constants)
              or (isinstance(node, ast.FunctionDef) and node.name == "compute_pain_vector")]
    space = {"np": np, "PCA": PCA, "log": print}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(path), "exec"), space)
    return space["compute_pain_vector"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("revision")
    p.add_argument("output", type=Path)
    p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    args = p.parse_args()
    assert re.fullmatch(r"[0-9a-f]{40}", args.revision)
    assert torch.cuda.is_available()
    if args.dtype == "bfloat16":
        assert torch.cuda.is_bf16_supported(including_emulation=False)
    dtype = getattr(torch, args.dtype)
    if args.output.exists():
        p.error("output already exists")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer
    from importlib.metadata import version
    start = perf_counter()
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=args.revision)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=args.revision, torch_dtype=dtype, low_cpu_mem_usage=True)
    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_ID, hf_model=hf_model, tokenizer=tok, device="cuda", dtype=dtype, revision=args.revision)
    del hf_model
    model.eval()
    source_data = json.loads((args.source / DATA).read_text())["datasets"]
    fit = pca_recipe(args.source / EXTRACTION)
    directions = {"provenance": "regenerated", "layers": [10, 15],
                  "model_revision": args.revision, "dtype": args.dtype,
                  "source_dataset_sha256": file_sha(args.source / DATA),
                  "source_extraction_sha256": file_sha(args.source / EXTRACTION),
                  "transformer_lens": version("transformer-lens"),
                  "transformers": version("transformers"),
                  "torch": str(torch.__version__)}
    for tag, layer, dataset in (("s1", 10, "S1_1P"), ("s2", 15, "S2_1P")):
        rows = source_data[dataset]["sentences"]
        hook = f"blocks.{layer}.hook_resid_post"
        filename = args.output.with_name(f"{args.output.name}.{tag}.activations.pt")
        meta = {"dataset": dataset, "layer": layer, "source_sha": directions["source_dataset_sha256"],
                "revision": args.revision, "dtype": args.dtype, "versions": [directions[k] for k in
                                                       ("transformer_lens", "transformers", "torch")]}
        if filename.exists():
            payload = torch.load(filename, map_location="cpu", weights_only=True)
            assert payload["metadata"] == meta
            activations = payload["activations"]
        else:
            items = []
            with torch.inference_mode():
                for row in rows:
                    _, cache = model.run_with_cache(model.to_tokens(row["prompt"]),
                                                   names_filter=lambda name: name == hook)
                    items.append(cache[hook][0, -1].cpu())
            activations = torch.stack(items).float()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temp = filename.with_suffix(".tmp")
            torch.save({"metadata": meta, "activations": activations}, temp)
            temp.replace(filename)
        assert torch.isfinite(activations).all(), f"nonfinite activations for {tag}"
        directions[tag] = torch.tensor(fit(activations, [row["category"] for row in rows]))
        print(f"{tag}: layer {layer}, {len(rows)} extraction prompts", flush=True)
    released_path = args.source / "results/vectors_full_steering/vectors_full_Gemma_2_2B_instruct.pt"
    released = torch.load(released_path, weights_only=True, map_location="cpu")
    directions["s1_cosine_to_released_l10"] = torch.nn.functional.cosine_similarity(
        directions["s1"].float(), released["s1_pain_vector"].float(), dim=0).item()
    directions["elapsed_seconds"] = perf_counter() - start
    print(f"S1 regeneration alignment: {directions['s1_cosine_to_released_l10']:.6f}", flush=True)
    temp = args.output.with_suffix(".tmp")
    torch.save(directions, temp)
    temp.replace(args.output)


if __name__ == "__main__":
    main()
