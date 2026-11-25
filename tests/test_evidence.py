"""Test archive checks using modified temporary copies.

Updating each copy's manifest lets the tests check inconsistent contents
after its file hashes have been updated.
"""
import contextlib
import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import verify_evidence as evidence

SOURCE = Path(__file__).resolve().parents[1]


class EvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='pilot-evidence-')
        self.root = Path(self.temp.name) / 'repo'
        shutil.copytree(SOURCE, self.root, ignore=shutil.ignore_patterns(
            '.git', '__pycache__', 'external', 'local-runs'))
        self.addCleanup(self.temp.cleanup)

    def verify(self, summary_only=False):
        with patch.object(evidence, 'ROOT', self.root), contextlib.redirect_stdout(io.StringIO()) as output:
            if summary_only:
                evidence.check_tables()
            else:
                evidence.verify()
        return output.getvalue()

    def refresh(self, relative):
        path = self.root / 'provenance.json'
        manifest = json.loads(path.read_text())
        entry = next(r for r in manifest['artifacts'] if r['path'] == relative)
        entry['git_blob_sha1'] = evidence.blob_hash(self.root / relative)
        if entry['transformation'] == 'none':
            entry['source_git_blob_sha1'] = entry['git_blob_sha1']
        path.write_text(json.dumps(manifest))

    def reject_json(self, relative, mutate, message):
        path = self.root / relative
        original = path.read_bytes()
        data = json.loads(original)
        mutate(data)
        path.write_text(json.dumps(data))
        self.refresh(relative)
        try:
            with self.assertRaisesRegex(ValueError, message):
                self.verify()
        finally:
            path.write_bytes(original)
            self.refresh(relative)

    def summary_value(self, value):
        path = self.root / 'results/summary.csv'
        with path.open() as handle:
            reader = csv.DictReader(handle)
            fields, records = reader.fieldnames, list(reader)
        records[0]['max_writer_ratio_s1'] = value
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

    def test_complete_archive(self):
        self.assertIn('61 archived files, 60 responses, 318 writer records', self.verify())
        self.verify(summary_only=True)

    def test_missing_and_corrupt_archive_bytes(self):
        path = self.root / 'results/smoke/baseline.000.json'
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing file'):
            self.verify()
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Archive hash'):
            self.verify()

    def test_manifest_requires_exact_paths(self):
        path = self.root / 'provenance.json'
        manifest = json.loads(path.read_text())
        manifest['artifacts'][0]['path'] = 'unrelated.txt'
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Archive manifest coverage'):
            self.verify()

    def test_archive_symlink_cannot_escape_root(self):
        path = self.root / 'pain_axis/blind.py'
        outside = Path(self.temp.name) / 'outside.py'
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'Missing file'):
            self.verify()

    def test_unexpected_batch_is_not_silently_ignored(self):
        shutil.copyfile(self.root / 'results/smoke/baseline.000.json',
                        self.root / 'results/smoke/baseline.002.json')
        with self.assertRaisesRegex(ValueError, 'Unexpected or missing evidence files: smoke'):
            self.verify()

    def test_summary_numeric_validation_in_both_modes(self):
        for value in ('nan', 'inf', '-1'):
            self.summary_value(value)
            for summary_only in (False, True):
                with self.subTest(value=value, summary_only=summary_only):
                    with self.assertRaisesRegex(ValueError, 'Finite number|Nonnegative number'):
                        self.verify(summary_only=summary_only)

    def test_wrong_finite_derived_ratio(self):
        self.summary_value('0.125')
        with self.assertRaisesRegex(ValueError, 'Summary ratio: baseline'):
            self.verify()

    def test_prompt_tokens_require_nonempty_integer_arrays(self):
        for value in ([], [-1], ['17'], [True]):
            with self.subTest(value=value):
                self.reject_json('results/smoke/prompt_token_ids.json',
                                 lambda d: d.__setitem__('dismissal_01', value),
                                 'Nonempty token list|Integer: token')

    def test_generated_tokens_require_nonempty_integer_arrays(self):
        for value in ([], [-1], ['17'], [True], [0] * 61):
            with self.subTest(value=value[:3]):
                self.reject_json('results/smoke/baseline.000.json',
                                 lambda d: d['rows'][0].__setitem__('generated_ids_with_batch_padding', value),
                                 'Nonempty token list|Integer: token|Token length')

    def test_projection_dimensions_values_and_finiteness(self):
        for value in ([0.0], [-1.0, 2.0], [float('nan'), 0.0], [False, 0.0]):
            with self.subTest(value=value):
                self.reject_json('results/smoke/baseline.000.json',
                                 lambda d: d['rows'][0].__setitem__('fixed_avg_abs_projection_with_batch_padding', value),
                                 'Two-value vector|Nonnegative number|Nonfinite JSON|Finite number')

    def test_batch_prompt_order_is_checked(self):
        self.reject_json('results/smoke/baseline.000.json', lambda d: d['rows'].reverse(),
                         'Smoke batch prompt order/coverage')

    def test_elapsed_time_cannot_be_negative(self):
        self.reject_json('results/smoke/baseline.000.json', lambda d: d.__setitem__('seconds', -1),
                         'Nonnegative number: smoke seconds')

    def test_writer_ratio_array_requires_two_values(self):
        self.reject_json('results/smoke/joint.writers.json',
                         lambda d: d['embed'].__setitem__('relative_to_original', []), 'Two-value vector')

    def test_all_eighteen_paired_metrics_are_recomputed(self):
        for first, second in evidence.PAIRS:
            comparison = f'{first} -> {second}'
            for metric in evidence.METRICS:
                with self.subTest(comparison=comparison, metric=metric):
                    self.reject_json('results/ratings/paired.json',
                                     lambda d: d[comparison][metric].__setitem__('n', 999), 'Paired statistic')

    def test_paired_delta_and_exact_probability_are_recomputed(self):
        for field in ('delta', 'mcnemar_exact_unadjusted_exploratory'):
            with self.subTest(field=field):
                self.reject_json('results/ratings/paired.json',
                                 lambda d: d['baseline -> s1']['humor'].__setitem__(field, .123),
                                 'Paired statistic')

    def test_duplicate_rating_key_alias_rejected(self):
        path = self.root / 'results/ratings/condition_key.csv'
        with path.open() as handle:
            records = list(csv.reader(handle))
        with path.open('a', newline='') as handle:
            csv.writer(handle).writerow(records[1])
        self.refresh('results/ratings/condition_key.csv')
        with self.assertRaisesRegex(ValueError, 'Rating coverage'):
            self.verify()

    def test_residual_readout_means_are_recomputed(self):
        self.reject_json('results/residual_readouts.json',
                         lambda d: d['baseline'].__setitem__('mean_absolute_projection_each_direction', [0.0, 0.0]),
                         'Residual readout mean')

    def test_replay_target_and_match_flags_must_be_booleans(self):
        for field in ('target', 'historical_exact_match'):
            with self.subTest(field=field):
                self.reject_json('results/replay/default.baseline.000.json',
                                 lambda d: d['rows'][0].__setitem__(field, 1), 'Replay target flag|Replay match flag')

    def test_replay_mismatch_ids_are_recomputed(self):
        self.reject_json('results/replay/summary.json',
                         lambda d: d['backends']['default']['conditions']['baseline'].__setitem__('mismatch_ids', []),
                         'Replay mismatch IDs')

    def test_replay_companion_cohorts_must_agree(self):
        self.reject_json('results/replay/eager.baseline.000.json',
                         lambda d: d['rows'][2].__setitem__('id', 'unexpected_companion'), 'Replay batch cohort')

    def test_replay_tokens_are_validated(self):
        self.reject_json('results/replay/default.baseline.000.json',
                         lambda d: d['rows'][0].__setitem__('generated_ids_with_batch_padding', [-1]), 'Integer: token')

    def test_checks_survive_optimized_python(self):
        command = [sys.executable, '-O', str(self.root / 'scripts/verify_evidence.py')]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.summary_value('nan')
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('Finite number: summary', completed.stderr)

    def test_recovered_supplement_keeps_exact_source_bytes(self):
        path = self.root / evidence.RECOVERED_PATH
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Recovered direction source hash'):
            self.verify()

    def test_recovered_supplement_metadata_matches_release(self):
        recovered = json.loads((SOURCE / evidence.RECOVERED_PATH).read_text())
        metadata = json.loads((SOURCE / 'results/direction_metadata.json').read_text())
        geometry = json.loads((SOURCE / 'results/smoke/direction_geometry.json').read_text())
        recovered['model_revision'] = '0' * 40
        with self.assertRaisesRegex(ValueError, 'Recovered direction metadata'):
            evidence.check_recovered_values(recovered, metadata, geometry)

    def test_recovered_values_preserve_pilot_unnormalized_vectors(self):
        recovered = json.loads((SOURCE / evidence.RECOVERED_PATH).read_text())
        metadata = json.loads((SOURCE / 'results/direction_metadata.json').read_text())
        geometry = json.loads((SOURCE / 'results/smoke/direction_geometry.json').read_text())
        norms = evidence.check_recovered_values(recovered, metadata, geometry)
        self.assertGreater(norms[0], 12)
        self.assertGreater(norms[1], 24)
        for name, value, message in (
                ('s1', [], 'Recovered direction shape'),
                ('s1', [0.] * 2304, 'Recovered positive norm'),
                ('s1', [float('nan')] * 2304, 'Finite number'),
                ('s1', [0.1] * 2304, 'Recovered FP32 value'),
                ('s2', [-x for x in recovered['s2']], 'Recovered normalized direction cosine')):
            with self.subTest(message=message):
                changed = {**recovered, name: value}
                with self.assertRaisesRegex(ValueError, message):
                    evidence.check_recovered_values(changed, metadata, geometry)


if __name__ == '__main__':
    unittest.main()
