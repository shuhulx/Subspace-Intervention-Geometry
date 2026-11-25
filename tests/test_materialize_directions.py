"""Check recovered-value identity, safe output handling, and runner compatibility."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'scripts/materialize_directions.py'
SPEC = importlib.util.spec_from_file_location('materialize_directions', HELPER)
materializer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(materializer)
TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


class MaterializeDirectionsTests(unittest.TestCase):
    def test_help_works_without_site_packages(self):
        result = subprocess.run([sys.executable, '-S', str(HELPER), '--help'],
                                capture_output=True, text=True, check=True)
        self.assertIn('OUTPUT.pt', result.stdout)

    def test_tampered_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'changed.json'
            source.write_bytes(materializer.SOURCE.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'pinned SHA256'):
                materializer.load_source(source)

    def test_non_finite_and_inexact_values_are_rejected(self):
        original = materializer.load_source()
        for value in (float('nan'), float('inf'), .1, True):
            with self.subTest(value=value):
                changed = copy.deepcopy(original)
                changed['s1'][0] = value
                with self.assertRaises(ValueError):
                    materializer.validate_source(changed)

    def test_existing_output_is_preserved_before_torch_import(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'existing.pt'
            output.write_bytes(b'keep this file')
            with patch.object(materializer.importlib, 'import_module',
                              side_effect=AssertionError('PyTorch must not be imported')):
                with self.assertRaises(FileExistsError):
                    materializer.materialize(output)
            self.assertEqual(output.read_bytes(), b'keep this file')

    def test_dangling_symlink_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'existing-link.pt'
            try:
                output.symlink_to(Path(directory) / 'absent.pt')
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f'Symlinks are unavailable: {exc}')
            with self.assertRaises(FileExistsError):
                materializer.materialize(output)
            self.assertTrue(output.is_symlink())

    @unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is optional; CPU round-trip requires it')
    def test_cpu_roundtrip_and_archived_reader_compatibility(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'nested/recovered.pt'
            report = materializer.materialize(output)
            payload = torch.load(output, map_location='cpu', weights_only=True)
            source = materializer.load_source()
            for name in ('s1', 's2'):
                expected = torch.tensor(source[name], dtype=torch.float32)
                self.assertTrue(torch.equal(payload[name], expected))
                self.assertEqual(payload[name].device.type, 'cpu')
                self.assertEqual(tuple(payload[name].shape), (2304,))
            self.assertNotIn('tensor_file_sha256', payload)
            self.assertEqual(payload['historical_tensor_file_sha256'], materializer.HISTORICAL_TENSOR_SHA256)
            self.assertEqual(payload['recovered_from_json_sha256'], materializer.SOURCE_SHA256)
            self.assertEqual(report['output_sha256'], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertNotEqual(report['output_sha256'], materializer.HISTORICAL_TENSOR_SHA256)
            self.assertEqual(payload['dtype'], 'bfloat16')

            # Load just the archived reader to avoid loading the model.
            tree = ast.parse((ROOT / 'pain_axis/smoke_runner.py').read_text())
            reader = next(node for node in tree.body
                          if isinstance(node, ast.FunctionDef) and node.name == 'read_directions')
            namespace = {'torch': torch}
            exec(compile(ast.Module(body=[reader], type_ignores=[]), 'archived_reader', 'exec'), namespace)
            vectors = namespace['read_directions'](output, 'regenerated', materializer.MODEL_REVISION,
                                                   'cpu', 'bfloat16')
            self.assertEqual(tuple(vectors.shape), (2, 2304))
            torch.testing.assert_close(vectors.norm(dim=1), torch.ones(2))

    @unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is optional; serialization failure test requires it')
    def test_serialization_failure_leaves_no_output(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'failed.pt'
            with patch.object(torch, 'save', side_effect=OSError('simulated write failure')):
                with self.assertRaises(OSError):
                    materializer.materialize(output)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
