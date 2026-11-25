"""Exercise pinned upstream projection operators on a small CPU writer fixture.

Requires torch, but no Transformers, model download, CUDA, or pretrained inference.
The exact upstream source blobs are checked before their functions are selected.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = 'scripts/appC_ablation/01_ablation_small_models.py'
HISTORICAL_BLOB = 'e85747ed605c300e70404b0ce119cceeb891d878'
JOINT_BLOB = '74cf8055064d193ccd7a4468d6738f3ef3ae5ee2'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_sources(source, corrected):
    for root, expected in ((source, HISTORICAL_BLOB), (corrected, JOINT_BLOB)):
        path = root / SCRIPT
        require(path.is_file(), f'Missing pinned source: {path}')
        data = path.read_bytes()
        blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        require(blob == expected, f'Upstream source blob differs: {path}')


def run_checks(source, corrected):
    check_sources(source, corrected)
    try:
        import torch
    except ImportError as error:
        raise ValueError('The optional CPU operator check requires torch; no model weights are needed.') from error
    sys.path.insert(0, str(ROOT))
    from pain_axis.geometry import CONDITIONS, apply_condition, operators, rms_demonstration, writer_audit

    torch.manual_seed(1701)
    torch.set_num_threads(2)
    historical, joint_operator = operators(source, corrected)

    class Fixture:
        def __init__(self, dtype=torch.float32, dim=24, zero_gain=False):
            self.embed = torch.nn.Embedding(48, dim, dtype=dtype)
            self.head = torch.nn.Linear(dim, 48, bias=False, dtype=dtype)
            self.head.weight = self.embed.weight
            layers = []
            for _ in range(2):
                gains = torch.linspace(-.4, .6, dim, dtype=dtype)
                if zero_gain:
                    gains[0] = -1
                layers.append(SimpleNamespace(
                    self_attn=SimpleNamespace(o_proj=torch.nn.Linear(dim, dim, bias=False, dtype=dtype)),
                    mlp=SimpleNamespace(down_proj=torch.nn.Linear(2 * dim, dim, bias=False, dtype=dtype)),
                    post_attention_layernorm=SimpleNamespace(weight=gains.clone()),
                    post_feedforward_layernorm=SimpleNamespace(weight=gains.clone())))
            self.model = SimpleNamespace(layers=layers)

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return self.head

    def weights(model):
        return {name: weight.detach().clone()
                for name, weight, _ in historical['residual_write_matrices'](model)}

    first = torch.nn.functional.normalize(torch.randn(24), dim=0)
    second = torch.randn(24)
    second = torch.nn.functional.normalize(second - (second @ first) * first, dim=0)
    vectors = torch.stack([first, .4 * first + (1 - .4 ** 2) ** .5 * second])
    result = {'scope': 'CPU tensor-operator fixture; no pretrained inference or historical GPU replay',
              'torch': str(torch.__version__), 'seed': 1701,
              'source_blobs': {'historical': HISTORICAL_BLOB, 'joint': JOINT_BLOB}, 'conditions': {}}
    for dtype in (torch.float32, torch.bfloat16):
        model = Fixture(dtype)
        snapshot = historical['snapshot_weights'](model)
        audits = {}
        for condition in CONDITIONS:
            historical['restore_weights'](model, snapshot)
            apply_condition(model, historical, joint_operator, vectors, condition)
            audits[condition] = writer_audit(model, historical['residual_write_matrices'], snapshot, vectors)
        maxima = {condition: [max(row['relative_to_original'][i] for row in audit.values()) for i in range(2)]
                  for condition, audit in audits.items()}
        tolerance = 1e-5 if dtype == torch.float32 else .008
        require(max(maxima['joint']) < tolerance, f'Joint leakage in {dtype}: {maxima["joint"]}')
        require(audits['s1_then_s2']['embed']['relative_to_original'][0] > .1,
                f'Forward sequential reintroduction missing in {dtype}')
        require(audits['s2_then_s1']['embed']['relative_to_original'][1] > .1,
                f'Reverse sequential reintroduction missing in {dtype}')
        require(model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr(),
                'Tied embedding/readout storage was broken')

        historical['restore_weights'](model, snapshot)
        joint_operator['orthogonalize_directions'](model, [vectors[0], vectors[1]])
        joint = weights(model)
        historical['restore_weights'](model, snapshot)
        joint_operator['orthogonalize_directions'](model, [vectors[1], vectors[0], vectors[0], -vectors[1]])
        duplicate = weights(model)
        delta = max(float((joint[name].float() - duplicate[name].float()).abs().max()) for name in joint)
        require(delta < (3e-6 if dtype == torch.float32 else .032), f'Order/sign/duplicate invariance: {dtype}')

        historical['restore_weights'](model, snapshot)
        historical['orthogonalize_single'](model, vectors[1])
        single = weights(model)
        historical['restore_weights'](model, snapshot)
        joint_operator['orthogonalize_directions'](model, [vectors[1]])
        require(all(torch.equal(single[name], weight) for name, weight in weights(model).items()),
                'Single-direction arithmetic changed')
        historical['restore_weights'](model, snapshot)
        require(all(torch.equal(snapshot[name], weight) for name, weight in weights(model).items()),
                'Snapshot restore changed weights')
        result['conditions'][str(dtype)] = {
            'max_ratio_each_direction': maxima, 'duplicate_sign_order_max_weight_delta': delta,
            'single_operator_exact_match': True, 'restore_exact': True, 'embedding_head_tied': True}

    # Measure leakage with the true gains. Upstream uses a +1e-4 floor for zero gains.
    model = Fixture(zero_gain=True)
    snapshot = historical['snapshot_weights'](model)
    eye = torch.eye(24)
    boundary_vectors = torch.stack([(eye[0] + eye[1]) / 2 ** .5, eye[2]])
    joint_operator['orthogonalize_directions'](model, list(boundary_vectors))
    audit = writer_audit(model, historical['residual_write_matrices'], snapshot, boundary_vectors)
    boundary = {'max_ratio_each_direction': [max(r['relative_to_original'][i] for r in audit.values()) for i in range(2)],
                'clamped_gain_count': sum(r['clamped_gain_count'] for r in audit.values()),
                'interpretation': 'Expected limitation: flooring zero gains does not guarantee exact true-gain removal.'}
    require(boundary['clamped_gain_count'] == 4 and boundary['max_ratio_each_direction'][0] > .1,
            'Gain-floor boundary is not exposed by the true-gain audit')
    result['zero_gain_boundary'] = boundary
    rms = rms_demonstration()
    require(rms['joint_max_abs_leak'] < 1e-12 and rms['gain_folded_weight_edit_leak'] < 1e-12 and
            rms['plain_pre_norm_leak'] > .1 and rms['gain_folded_vs_literal_l2'] > 1e-3,
            'RMS normalization demonstration failed')
    result['rms_demonstration'] = rms
    result['status'] = 'PASS'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Historical upstream checkout at the recorded commit')
    parser.add_argument('pr', type=Path, help='Upstream joint-correction checkout at the recorded PR commit')
    args = parser.parse_args()
    try:
        result = run_checks(args.source, args.pr)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
