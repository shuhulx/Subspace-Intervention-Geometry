"""Check archive identity and internal consistency without loading the model.

Hash checks pin released bytes; semantic checks recompute derived relationships.
Neither check proves that the historical GPU experiment can replay.
"""
import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('baseline', 's1', 's2', 's1_then_s2', 's2_then_s1', 'joint')
PROMPTS = ('dismissal_01', 'dismissal_02', 'insult_01', 'insult_02',
           'gaslight_01', 'gaslight_02', 'rejection_01', 'rejection_02',
           'moral_01', 'moral_02')
METRICS = ('humor', 'coherent', 'relevant')
SMOKE_SHA = '8778b1403ffa7bc450d48775cbe1263ff2c319bf5a40531dac1a58f01f1e55b1'
REPLAY_SHA = 'b6a5575f76a44abe1cf57649f5ab693736172fec3023bba213735c8d61f9627c'
PAIRS = (('baseline', 's1'), ('baseline', 's2'), ('baseline', 's1_then_s2'),
         ('baseline', 'joint'), ('s1_then_s2', 's2_then_s1'), ('s1_then_s2', 'joint'))
REPLAY_BATCHES = (0, 2, 5, 7, 10)
REPLAY_MATCHES = {('default', 'baseline'): 5, ('default', 's1_then_s2'): 4,
                  ('eager', 'baseline'): 7, ('eager', 's1_then_s2'): 4}
TRANSFORMATIONS = {
    'results/direction_metadata.json': 'Numeric direction arrays omitted; original recorded tensor hash retained.',
    'results/smoke_inspection.json': 'Local run directory omitted.',
}
SMOKE_FILES = ({f'{c}.{b:03d}.json' for c in CONDITIONS for b in range(2)} |
               {f'{c}.writers.json' for c in CONDITIONS} |
               {'settings.json', 'direction_geometry.json', 'prompt_token_ids.json'})
REPLAY_FILES = ({f'{backend}.{c}.{b:03d}.json' for backend, c in REPLAY_MATCHES
                 for b in REPLAY_BATCHES} | {'summary.json'})
RATING_FILES = {'ai_unblinded_ratings.csv', 'condition_key.csv', 'method.json', 'paired.json'}
ARCHIVE_PATHS = ({f'pain_axis/{name}' for name in (
    'blind.py', 'geometry.py', 'historical.py', 'inspect_run.py', 'regenerate.py',
    'replay_debug.py', 'rubric.md', 'run.py', 'smoke_runner.py')} |
    {'requirements-recorded.txt'} |
    {f'results/{name}.json' for name in ('direction_metadata', 'historical_summary',
                                       'residual_readouts', 'smoke_inspection', 'synthetic_geometry')} |
    {f'results/smoke/{name}' for name in SMOKE_FILES} |
    {f'results/replay/{name}' for name in REPLAY_FILES} |
    {f'results/ratings/{name}' for name in RATING_FILES})
WRITERS = {'embed'} | {f'{kind}_{i}' for kind in ('o_proj', 'down_proj') for i in range(26)}
RECOVERED_PATH = 'results/recovered_direction_vectors.json'
RECOVERED_BLOB = 'e618abf662a78dae629516069f3e55397b55b63d'
RECOVERED_SHA256 = 'ab04c16c30c61c5bbfb2c1f2560d2998cad10d082149081aeceec5cde8cf03fb'


def check(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            check(key not in result, f'Duplicate JSON key: {key}')
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f'Nonfinite JSON value in {path.name}: {value}')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_object,
                      parse_constant=invalid_constant)


def rows(path):
    with path.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        check(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)),
              f'CSV columns: {path.name}')
        result = list(reader)
    check(all(None not in row and all(v is not None for v in row.values()) for row in result),
          f'CSV row shape: {path.name}')
    return result


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blob_hash(path):
    data = path.read_bytes()
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


def close(a, b):
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7)


def number(value, label, nonnegative=True):
    check(type(value) in (int, float) and math.isfinite(value), f'Finite number: {label}')
    check(not nonnegative or value >= 0, f'Nonnegative number: {label}')
    return value


def integer(value, label, minimum=0):
    check(type(value) is int and value >= minimum, f'Integer: {label}')
    return value


def vector(value, label):
    check(isinstance(value, list) and len(value) == 2, f'Two-value vector: {label}')
    for item in value:
        number(item, label)
    return value


def tokens(value, label, maximum=None):
    check(isinstance(value, list) and bool(value), f'Nonempty token list: {label}')
    check(maximum is None or len(value) <= maximum, f'Token length: {label}')
    for item in value:
        integer(item, f'token {label}')


def paired_statistics(panel):
    """Recompute all specified contrasts from an already validated rating panel."""
    result = {}
    for first, second in PAIRS:
        contrast = {}
        for metric in METRICS:
            before = [int(panel[first, p][metric]) for p in PROMPTS]
            after = [int(panel[second, p][metric]) for p in PROMPTS]
            positive = sum(a < b for a, b in zip(before, after))
            negative = sum(a > b for a, b in zip(before, after))
            discordant = positive + negative
            probability = min(1, 2 * sum(math.comb(discordant, j)
                              for j in range(min(positive, negative) + 1)) / 2 ** discordant)
            contrast[metric] = {'before': sum(before), 'after': sum(after),
                                'zero_to_one': positive, 'one_to_zero': negative, 'n': len(PROMPTS),
                                'delta': (positive - negative) / len(PROMPTS),
                                'mcnemar_exact_unadjusted_exploratory': probability}
        result[f'{first} -> {second}'] = contrast
    return result


def check_paired(recorded, expected):
    check(set(recorded) == set(expected), 'Paired comparison coverage')
    for comparison, metrics in expected.items():
        check(set(recorded[comparison]) == set(METRICS), f'Paired metrics: {comparison}')
        for metric, values in metrics.items():
            actual = recorded[comparison][metric]
            check(set(actual) == set(values), f'Paired fields: {comparison}/{metric}')
            for field, value in values.items():
                label = f'{comparison}/{metric}/{field}'
                if field in ('delta', 'mcnemar_exact_unadjusted_exploratory'):
                    number(actual[field], label, nonnegative=field != 'delta')
                    check(close(actual[field], value), f'Paired statistic: {label}')
                else:
                    integer(actual[field], label)
                    check(actual[field] == value, f'Paired statistic: {label}')


def check_recovered_values(recovered, metadata, geometry):
    """Validate the recorded pilot's unnormalized PCA values; the runner normalizes them."""
    expected_metadata = {k: v for k, v in metadata.items() if k != 'release_note'}
    check(set(recovered) == set(expected_metadata) | {'s1', 's2'}, 'Recovered direction fields')
    check(all(recovered[k] == v for k, v in expected_metadata.items()), 'Recovered direction metadata')
    norms = []
    for tag in ('s1', 's2'):
        values = recovered[tag]
        check(isinstance(values, list) and len(values) == 2304, f'Recovered direction shape: {tag}')
        for value in values:
            number(value, f'recovered {tag}', nonnegative=False)
            check(abs(value) <= 3.4028234663852886e38, f'Recovered FP32 range: {tag}')
            check(struct.unpack('f', struct.pack('f', value))[0] == value, f'Recovered FP32 value: {tag}')
        norm = math.sqrt(math.fsum(v * v for v in values))
        check(norm > 0 and math.isfinite(norm), f'Recovered positive norm: {tag}')
        norms.append(norm)
    cosine = math.fsum(a * b for a, b in zip(recovered['s1'], recovered['s2'])) / math.prod(norms)
    number(geometry['cosine'], 'recorded direction cosine', nonnegative=False)
    check(abs(cosine - geometry['cosine']) <= 1e-8, 'Recovered normalized direction cosine')
    check(close(geometry['squared_overlap'], cosine * cosine) and
          close(geometry['principal_angle_degrees'], math.degrees(math.acos(min(1., abs(cosine))))) and
          close(geometry['sequential_minus_joint_operator_norm'], abs(cosine)), 'Recovered direction geometry')
    return norms


def check_recovered(manifest, metadata, geometry):
    entries = manifest.get('supplemental_artifacts')
    check(isinstance(entries, list) and len(entries) == 1, 'Supplemental artifact coverage')
    entry = entries[0]
    expected = {'path': RECOVERED_PATH, 'git_blob_sha1': RECOVERED_BLOB,
                'source_git_blob_sha1': RECOVERED_BLOB, 'transformation': 'none',
                'source_repository': 'private-source-archive',
                'source_commit': '29175c65ccca87eacda0b38d5e83f4ea94709478',
                'source_path': 'audit/2026-11-19/smoke06/regenerated_vectors.json'}
    check(all(entry.get(k) == v for k, v in expected.items()), 'Recovered source provenance')
    path = (ROOT / RECOVERED_PATH).resolve()
    check(ROOT.resolve() in path.parents and path.is_file(), 'Recovered direction file missing')
    check(blob_hash(path) == RECOVERED_BLOB and sha256(path) == RECOVERED_SHA256, 'Recovered direction source hash')
    return check_recovered_values(load(path), metadata, geometry)


def check_tables():
    labels = rows(ROOT / 'results/exploratory_labels.csv')
    label_map = {(r['condition'], r['prompt_id']): r for r in labels}
    expected = {(c, p) for c in CONDITIONS for p in PROMPTS}
    check(len(labels) == 60 and set(label_map) == expected, 'Label coverage')
    check(all(r['rating_provenance'] == 'unblinded_AI' and
              all(r[m] in ('0', '1') for m in METRICS) for r in labels), 'Label values/provenance')
    table = rows(ROOT / 'results/summary.csv')
    summary = {r['condition']: r for r in table}
    check(len(table) == 6 and set(summary) == set(CONDITIONS), 'Summary coverage')
    for c, r in summary.items():
        check(int(r['n_prompts']) == 10 and int(r['n_writers']) == 53, f'Scope: {c}')
        for field in ('max_writer_ratio_s1', 'max_writer_ratio_s2', 'mean_relative_writer_change'):
            number(float(r[field]), f'summary {c}/{field}')
        for m in METRICS:
            check(int(r[f'ai_{m}_count']) == sum(int(label_map[c, p][m]) for p in PROMPTS), f'Count: {c}/{m}')
    check(all(float(summary['joint'][f'max_writer_ratio_{s}']) < .0023 for s in ('s1', 's2')), 'Joint bound')
    return summary, label_map


def verify():
    summary, labels = check_tables()
    manifest = load(ROOT / 'provenance.json')
    archive = manifest['artifacts']
    check(len(archive) == len(ARCHIVE_PATHS) == 61 and
          {r['path'] for r in archive} == ARCHIVE_PATHS, 'Archive manifest coverage')
    for r in archive:
        p = (ROOT / r['path']).resolve()
        check(ROOT.resolve() in p.parents and p.is_file(), f'Missing file: {r["path"]}')
        check(blob_hash(p) == r['git_blob_sha1'], f'Archive hash: {r["path"]}')
        check(r['transformation'] == TRANSFORMATIONS.get(r['path'], 'none'),
              f'Archive transformation: {r["path"]}')
        if r['transformation'] == 'none':
            check(r['git_blob_sha1'] == r['source_git_blob_sha1'], f'Source hash: {r["path"]}')
    for folder, expected in (('smoke', SMOKE_FILES), ('replay', REPLAY_FILES), ('ratings', RATING_FILES)):
        actual = {p.name for p in (ROOT / 'results' / folder).iterdir()
                  if p.is_file() and p.suffix in ('.json', '.csv')}
        check(actual == expected, f'Unexpected or missing evidence files: {folder}')
    check(sha256(ROOT / 'pain_axis/smoke_runner.py') == SMOKE_SHA, 'Original runner hash')
    check(sha256(ROOT / 'pain_axis/replay_debug.py') == REPLAY_SHA, 'Replay runner hash')
    res = ROOT / 'results'
    smoke, ratings, replay = (res / n for n in ('smoke', 'ratings', 'replay'))
    settings = load(smoke / 'settings.json')
    check(settings['runner_sha256'] == SMOKE_SHA and settings['prompt_ids'] == list(PROMPTS), 'Smoke settings')
    check(settings['dtype'] == 'bfloat16' and type(settings['batch_size']) is int and
          settings['batch_size'] == 8 and type(settings['greedy_max_new_tokens']) is int and
          settings['greedy_max_new_tokens'] == 60 and settings['disable_compile'] is True, 'Run configuration')
    prompt_tokens = load(smoke / 'prompt_token_ids.json')
    check(set(prompt_tokens) == set(PROMPTS), 'Prompt token coverage')
    for prompt, values in prompt_tokens.items():
        tokens(values, f'prompt {prompt}')
    direction = load(res / 'direction_metadata.json')
    alignment = number(direction['s1_cosine_to_released_l10'], 'direction cosine', nonnegative=False)
    check(direction['layers'] == [10, 15] and .98 < alignment <= 1 and
          direction['tensor_file_sha256'] == settings['vectors_sha256'] and
          direction['model_revision'] == settings['revision'] and
          direction['dtype'] == settings['dtype'] and
          direction['provenance'] == settings['vectors_provenance'] == 'regenerated', 'Direction provenance')
    check_recovered(manifest, direction, load(smoke / 'direction_geometry.json'))
    baseline = load(smoke / 'baseline.writers.json')
    check(set(baseline) == WRITERS, 'Writer coverage')
    outputs, prompt_hashes, categories = {}, {}, {}
    readouts = load(res / 'residual_readouts.json')
    check(set(readouts) == set(CONDITIONS) | {'scope'}, 'Residual readout coverage')
    max_error = 0.
    for c in CONDITIONS:
        writers = load(smoke / f'{c}.writers.json')
        check(set(writers) == WRITERS, f'Writers: {c}')
        for name, writer in writers.items():
            vector(writer['direction_norm'], f'{c}/{name} direction norm')
            vector(writer['relative_to_original'], f'{c}/{name} ratio')
            number(writer['relative_writer_change'], f'{c}/{name} change')
            integer(writer['clamped_gain_count'], f'{c}/{name} clamped gains')
        for i, direction_name in enumerate(('s1', 's2')):
            ratios = []
            for name, w in writers.items():
                recorded = w['relative_to_original'][i]
                check(baseline[name]['direction_norm'][i] > 0, f'Positive baseline norm: {name}')
                ratio = w['direction_norm'][i] / max(baseline[name]['direction_norm'][i], 1e-30)
                max_error = max(max_error, abs(ratio - recorded))
                check(close(ratio, recorded), f'Writer ratio: {c}/{name}/{direction_name}')
                ratios.append(recorded)
            check(close(max(ratios), float(summary[c][f'max_writer_ratio_{direction_name}'])), f'Summary ratio: {c}')
        check(close(mean(w['relative_writer_change'] for w in writers.values()),
                    float(summary[c]['mean_relative_writer_change'])), f'Mean writer change: {c}')
        generations = []
        for b in range(2):
            batch = load(smoke / f'{c}.{b:03d}.json')
            check(batch['condition'] == c and type(batch['batch']) is int and
                  batch['batch'] == b, f'Smoke batch: {c}/{b}')
            number(batch['seconds'], f'smoke seconds {c}/{b}')
            check([r['id'] for r in batch['rows']] == list(PROMPTS[b * 8:(b + 1) * 8]),
                  f'Smoke batch prompt order/coverage: {c}/{b}')
            generations.extend(batch['rows'])
        check(len(generations) == 10 and {r['id'] for r in generations} == set(PROMPTS), f'Generation coverage: {c}')
        for r in generations:
            p = r['id']
            check(prompt_hashes.setdefault(p, r['prompt_sha256']) == r['prompt_sha256'], f'Prompt mismatch: {p}')
            check(isinstance(r['category'], str) and r['category'] and
                  categories.setdefault(p, r['category']) == r['category'], f'Prompt category: {p}')
            check(isinstance(r['generation'], str) and r['generation'].strip(), f'Output: {c}/{p}')
            tokens(r['generated_ids_with_batch_padding'], f'{c}/{p}', 60)
            vector(r['fixed_avg_abs_projection_with_batch_padding'], f'projection {c}/{p}')
            outputs[c, p] = r['generation']
        recorded = readouts[c]
        check(integer(recorded['n'], f'readout n/{c}') == len(generations), 'Residual readout count')
        values = vector(recorded['mean_absolute_projection_each_direction'], f'readout mean/{c}')
        for i in range(2):
            check(close(values[i], mean(r['fixed_avg_abs_projection_with_batch_padding'][i]
                                       for r in generations)), f'Residual readout mean: {c}/{i}')
        check(integer(recorded['maximum_generated_ids_including_padding'], 'readout token maximum') ==
              max(len(r['generated_ids_with_batch_padding']) for r in generations), 'Residual readout token maximum')
    method = load(ratings / 'method.json')
    check(method['blinded'] is False and type(method['independent_humans']) is int and
          method['independent_humans'] == 0, 'Rating provenance')
    check(method['rubric_sha256'] == sha256(ROOT / 'pain_axis/rubric.md'), 'Rubric hash')
    raw = rows(ratings / 'ai_unblinded_ratings.csv')
    aliases = {r['anonymous_id']: r for r in raw}
    keys = rows(ratings / 'condition_key.csv')
    check(len(raw) == len(aliases) == len(keys) == len({r['anonymous_id'] for r in keys}) == 60, 'Rating coverage')
    check(set(aliases) == {r['anonymous_id'] for r in keys}, 'Rating alias correspondence')
    check({(r['condition'], r['prompt_id']) for r in keys} == set(outputs), 'Condition key coverage')
    for k in keys:
        c, p = k['condition'], k['prompt_id']
        r = aliases[k['anonymous_id']]
        check(r['response'] == outputs[c, p], f'Rating response: {c}/{p}')
        check(hashlib.sha256(r['prompt'].encode()).hexdigest() == prompt_hashes[p], f'Prompt hash: {p}')
        check(all(r[m] == labels[c, p][m] for m in METRICS), f'Label table: {c}/{p}')
    panel = {(r['condition'], r['prompt_id']): aliases[r['anonymous_id']] for r in keys}
    check_paired(load(ratings / 'paired.json'), paired_statistics(panel))
    replay_summary = load(replay / 'summary.json')
    check(replay_summary['runner_sha256'] == REPLAY_SHA and
          replay_summary['vectors_sha256'] == settings['vectors_sha256'] and
          replay_summary['revision'] == settings['revision'], 'Replay provenance')
    check(replay_summary['target_ids'] == sorted(PROMPTS) and
          replay_summary['author_batch_indices'] == list(REPLAY_BATCHES) and
          replay_summary['disable_compile'] is True, 'Replay cohort settings')
    check(set(replay_summary['backends']) == {'default', 'eager'}, 'Replay backend coverage')
    for backend, attention in (('default', 'sdpa'), ('eager', 'eager')):
        item = replay_summary['backends'][backend]
        check(item['actual_attention'] == attention and
              set(item['conditions']) == {'baseline', 's1_then_s2'}, 'Replay conditions/backend')
    cohorts = {}
    total = 0
    for (backend, c), count in REPLAY_MATCHES.items():
        generations = []
        for b in REPLAY_BATCHES:
            batch = load(replay / f'{backend}.{c}.{b:03d}.json')
            check(batch['backend'] == backend and batch['condition'] == c and
                  type(batch['author_batch']) is int and batch['author_batch'] == b and
                  len(batch['rows']) == 8, 'Replay batch')
            number(batch['seconds'], 'replay seconds')
            ids = [r['id'] for r in batch['rows']]
            check(cohorts.setdefault(b, ids) == ids, f'Replay batch cohort: {b}')
            generations.extend(batch['rows'])
        for row in generations:
            check(isinstance(row['id'], str) and row['id'], 'Replay ID')
            check(type(row['target']) is bool and row['target'] == (row['id'] in PROMPTS), 'Replay target flag')
            check(isinstance(row['generation'], str) and row['generation'].strip(), 'Replay output text')
            tokens(row['generated_ids_with_batch_padding'], f'replay {backend}/{c}/{row["id"]}', 60)
            if row['target']:
                check(type(row['historical_exact_match']) is bool, 'Replay match flag')
            else:
                check('historical_exact_match' not in row, 'Replay companion score')
        targets = [r for r in generations if r['target']]
        check(len(generations) == len({r['id'] for r in generations}) == 40 and
              len(targets) == 10 and {r['id'] for r in targets} == set(PROMPTS), 'Replay coverage')
        matches = sum(r['historical_exact_match'] for r in targets)
        recorded = replay_summary['backends'][backend]['conditions'][c]
        integer(recorded['exact_matches'], 'replay exact matches')
        check(matches == count == recorded['exact_matches'], 'Replay counts')
        check(recorded['mismatch_ids'] == [r['id'] for r in targets if not r['historical_exact_match']],
              f'Replay mismatch IDs: {backend}/{c}')
        total += len(generations)
    gate = all(c['exact_matches'] == len(PROMPTS)
               for c in replay_summary['backends']['default']['conditions'].values())
    check(type(replay_summary['full_panel_gate_passed']) is bool and
          replay_summary['full_panel_gate_passed'] == gate and not gate, 'Replay gate')
    print(f'PASS: {len(archive)} archived files, {len(outputs)} responses, 318 writer records, 60 ratings, {total} replay generations.')
    print(f'Maximum writer-ratio error: {max_error:.3g}. All 18 paired metrics and six residual readouts recomputed.')
    print('Recovered direction supplement: source bytes and 2 x 2304 FP32 values verified; original .pt bytes are not reconstructed.')
    print('Historical replay gate failed as recorded; no confirmatory panel was run.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-only', action='store_true', help='Check only the derived tables, not the archive')
    if parser.parse_args().summary_only:
        check_tables()
        print('PASS: six summary conditions and 60 exploratory labels (archive not checked).')
    else:
        verify()
