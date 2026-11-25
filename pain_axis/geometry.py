"""Run pinned upstream operators and measure effective Gemma writer leakage."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

HISTORY = "213fb6c89c96004dbe93d1704c636ad8c9d801f3"
PR4 = "a0a955052852f31afbc2b8ddfbd508d60a7b432b"
SCRIPT = "scripts/appC_ablation/01_ablation_small_models.py"
CONDITIONS = {
    "baseline": (), "s1": (0,), "s2": (1,), "s1_then_s2": (0, 1),
    "s2_then_s1": (1, 0), "joint": (0, 1),
}


def upstream(path, names):
    tree = ast.parse(path.read_text())
    subset = [node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name in names]
    assert {node.name for node in subset} == names
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=subset, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def operators(source, pr):
    common = {"get_decoder_layers", "residual_write_matrices"}
    historical = upstream(source / SCRIPT, common | {
        "orthogonalize_single", "snapshot_weights", "restore_weights", "parse_turns", "render_prompt"})
    corrected = upstream(pr / SCRIPT, common | {"orthogonalize_directions", "orthogonal_basis"})
    return historical, corrected


def apply_condition(model, historical, corrected, vectors, condition):
    indices = CONDITIONS[condition]
    if condition == "joint":
        corrected["orthogonalize_directions"](model, [vectors[i] for i in indices])
    else:
        for i in indices:
            historical["orthogonalize_single"](model, vectors[i])


def file_sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def geometry(rows):
    v = torch.nn.functional.normalize(rows.double(), dim=1)
    cosine = (v[0] @ v[1]).clamp(-1, 1)
    # Squared overlap is sign-invariant. The norm is on the two-vector span.
    return {"cosine": cosine.item(),
            "principal_angle_degrees": torch.rad2deg(torch.acos(cosine.abs())).item(),
            "squared_overlap": cosine.square().item(),
            "sequential_minus_joint_operator_norm": cosine.abs().item()
            if abs(cosine.item()) < 1 - 1e-12 else 0.0}


@torch.no_grad()
def writer_audit(model, matrices, original, vectors):
    metrics = {}
    for name, weight, scale in matrices(model):
        # For embedding the feature dimension is the final axis. For the
        # linear writers use true learned gains, including unclamped entries.
        matrix = weight.float()
        reference = original[name].to(weight.device).float()
        if name == "embed":
            matrix, reference = matrix.T, reference.T
        if scale is not None:
            matrix = matrix * scale[:, None]
            reference = reference * scale[:, None]
        per_direction = torch.linalg.vector_norm(vectors @ matrix, dim=1)
        baseline = torch.linalg.vector_norm(vectors @ reference, dim=1)
        change = torch.linalg.vector_norm(matrix - reference) / torch.linalg.vector_norm(reference)
        metrics[name] = {
            "direction_norm": per_direction.tolist(),
            "relative_to_original": (per_direction / baseline.clamp_min(1e-30)).tolist(),
            "relative_writer_change": change.item(),
            "clamped_gain_count": 0 if scale is None else int((scale.abs() < 1e-4).sum()),
        }
    return metrics


def rms_demonstration():
    dtype = torch.float64
    e1 = torch.tensor([1., 0., 0.], dtype=dtype)
    e2 = torch.tensor([0., 1., 0.], dtype=dtype)
    directions = torch.stack([(e1 + e2) / 2**0.5, e2])
    basis = torch.linalg.svd(directions, full_matrices=False).Vh
    p = torch.eye(3, dtype=dtype) - basis.T @ basis
    p1 = torch.eye(3, dtype=dtype) - torch.outer(directions[0], directions[0])
    p2 = torch.eye(3, dtype=dtype) - torch.outer(directions[1], directions[1])
    # A single oblique direction shows the effect of nonuniform normalization gains.
    x = torch.tensor([1., 3., 2.], dtype=dtype)
    gain = torch.tensor([0.7, 1.6, 1.2], dtype=dtype)

    def rms(y):
        return y * gain / (y.square().mean() + 1e-6).sqrt()

    effective = rms(x)
    weight_edit = rms((p1 @ (gain * x)) / gain)
    literal = p1 @ effective
    values = {"forward_reintroduced_S1": abs(directions[0] @ p2 @ p1 @ x).item(),
              "reverse_reintroduced_S2": abs(directions[1] @ p1 @ p2 @ x).item(),
              "joint_max_abs_leak": (directions @ p @ x).abs().max().item(),
              "plain_pre_norm_leak": abs(directions[0] @ rms(p1 @ x)).item(),
              "gain_folded_weight_edit_leak": abs(directions[0] @ weight_edit).item(),
              "post_norm_projection_leak": abs(directions[0] @ literal).item(),
              "gain_folded_vs_literal_l2": (weight_edit - literal).norm().item()}
    assert values["joint_max_abs_leak"] < 1e-12
    assert values["gain_folded_weight_edit_leak"] < 1e-12
    assert values["plain_pre_norm_leak"] > 0.1 and values["gain_folded_vs_literal_l2"] > 1e-3
    return values


def main():
    from transformers import Gemma2Config, Gemma2ForCausalLM

    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("pr", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    torch.manual_seed(1701)
    torch.set_num_threads(4)
    historical, corrected = operators(args.source, args.pr)
    path = args.source / "results/vectors_full_steering/vectors_full_Gemma_2_2B_instruct.pt"
    released = torch.load(path, weights_only=True, map_location="cpu")
    assert released["layer"] == 10
    out = {"torch": torch.__version__, "seed": 1701, "rms_demonstration": rms_demonstration(),
           "released_steering_bundle": {
               "sha256": file_sha(path), "layers": [10, 10],
               "identity": "released Section 4.1 vectors, not Appendix C S1 L10 / S2 L15",
               **geometry(torch.stack([released["s1_pain_vector"], released["s2_pain_vector"]]))},
           "fixture": "random-weight model instantiated from HF Gemma2ForCausalLM",
           "writer_audit": {}, "random_model_forward": {}}
    e = torch.eye(24)
    vectors = torch.stack([e[0], (e[0] + e[1]) / 2**0.5])
    for dtype in (torch.float32, torch.bfloat16):
        model = Gemma2ForCausalLM(Gemma2Config(
            vocab_size=128, hidden_size=24, intermediate_size=48, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=6,
            query_pre_attn_scalar=6, max_position_embeddings=128)).to(dtype).eval()
        for layer in model.model.layers:
            for norm in (layer.post_attention_layernorm, layer.post_feedforward_layernorm):
                norm.weight.data.copy_(torch.linspace(-.4, .6, 24))
        original = historical["snapshot_weights"](model)
        matrices = historical["residual_write_matrices"]
        out["writer_audit"][str(dtype)] = audits = {}
        out["random_model_forward"][str(dtype)] = forwards = {}
        ids = torch.tensor([[1, 40, 77, 3], [1, 32, 17, 4]])
        input_hash = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
        for condition in CONDITIONS:
            historical["restore_weights"](model, original)
            apply_condition(model, historical, corrected, vectors, condition)
            audits[condition] = writer_audit(model, matrices, original, vectors)
            with torch.inference_mode():
                out_model = model(input_ids=ids, output_hidden_states=True, use_cache=False)
                generated = model.generate(input_ids=ids, max_new_tokens=4, do_sample=False,
                                           eos_token_id=2, pad_token_id=0)
            residual = out_model.hidden_states[1].float() @ vectors.T
            assert torch.isfinite(out_model.logits).all()
            assert torch.equal(generated[:, :ids.shape[1]], ids)
            forwards[condition] = {
                "same_input_token_ids_sha256": input_hash,
                "block0_max_abs_direction_projection": residual.abs().amax(dim=(0, 1)).tolist(),
                "generated_token_ids_random_fixture": generated[:, ids.shape[1]:].tolist(),
            }
            if condition == "joint":
                tolerance = 1e-5 if dtype == torch.float32 else .008
                assert all(max(x["relative_to_original"]) < tolerance
                           for x in audits[condition].values())
        assert audits["s1_then_s2"]["embed"]["relative_to_original"][0] > .1
        assert audits["s2_then_s1"]["embed"]["relative_to_original"][1] > .1
        assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    out["wall_seconds"] = perf_counter() - start
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
