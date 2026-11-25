"""Preflight failures must be detected before loading a model or reusing data."""

import argparse
import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "scripts/run_fresh.py"
SPEC = importlib.util.spec_from_file_location("run_fresh", PATH)
fresh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fresh)


class FreshRunTests(unittest.TestCase):
    def test_modified_local_frozen_script_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pain_axis").mkdir()
            (root / "pain_axis/geometry.py").write_text("changed\n", encoding="utf-8")
            with patch.object(fresh, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "Frozen local script changed"):
                    fresh.verify_local_scripts("smoke")

    def test_replay_rejects_bad_metadata_tokens_and_incomplete_smoke(self):
        archived = fresh.ROOT / "results/smoke"
        settings = json.loads((archived / "settings.json").read_text())
        for mutation in ("seed", "runner", "tokens", "missing_checkpoint", "wrong_order"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                smoke = Path(tmp) / "smoke"
                shutil.copytree(archived, smoke)
                if mutation in ("seed", "runner"):
                    changed = dict(settings)
                    changed["seed" if mutation == "seed" else "runner_sha256"] = 0
                    (smoke / "settings.json").write_text(json.dumps(changed))
                elif mutation == "tokens":
                    tokens = json.loads((smoke / "prompt_token_ids.json").read_text())
                    tokens[fresh.PROMPT_IDS[0]] = [True]
                    (smoke / "prompt_token_ids.json").write_text(json.dumps(tokens))
                elif mutation == "missing_checkpoint":
                    (smoke / "joint.001.json").unlink()
                else:
                    path = smoke / "baseline.000.json"
                    batch = json.loads(path.read_text())
                    batch["rows"].reverse()
                    path.write_text(json.dumps(batch))
                with self.assertRaises(ValueError):
                    fresh.validate_replay_smoke(smoke, settings["vectors_sha256"],
                                                settings["dataset_sha256"])

    def test_complete_smoke_preflight_uses_recorded_values(self):
        smoke = fresh.ROOT / "results/smoke"
        settings = json.loads((smoke / "settings.json").read_text())
        result = fresh.validate_replay_smoke(smoke, settings["vectors_sha256"],
                                           settings["dataset_sha256"])
        self.assertEqual(result["smoke_tokens_sha256"], fresh.digest(smoke / "prompt_token_ids.json"))

    def test_existing_output_and_dangling_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "used"
            path.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                fresh.check_output(path)
            link = Path(tmp) / "link"
            link.symlink_to(Path(tmp) / "missing")
            with self.assertRaisesRegex(ValueError, "already exists"):
                fresh.check_output(link)

    def test_archived_results_cannot_be_output(self):
        with self.assertRaisesRegex(ValueError, "local-runs"):
            fresh.check_output(fresh.ROOT / "results/accidental-run")

    def test_modified_pinned_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "input.py"
            file.write_text("original\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "add", "input.py"], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Test",
                            "-c", "user.email=test@localhost", "commit", "-qm", "fixture"], check=True)
            sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            self.assertEqual(fresh.verify_checkout(root, sha, ["input.py"]),
                             {"input.py": fresh.digest(file)})
            file.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from pinned"):
                fresh.verify_checkout(root, sha, ["input.py"])

    def test_wrong_upstream_revision_rejected(self):
        with patch.object(fresh.subprocess, "check_output", return_value="wrong\n"):
            with self.assertRaisesRegex(ValueError, "Wrong upstream revision"):
                fresh.verify_checkout(Path("."), fresh.HISTORY, [])

    def test_smoke_plan_is_fixed_ten_prompt_run_and_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vectors = root / "directions.pt"
            vectors.write_bytes(b"fixture-only; dry-run does not load a tensor")
            args = argparse.Namespace(task="smoke", source=root / "source", pr=root / "pr",
                                      vectors=vectors, out=root / "new")
            with patch.object(fresh, "verify_checkout", return_value={}):
                out, command, metadata = fresh.plan(args)
            self.assertFalse(out.exists())
            self.assertIn("pain_axis.smoke_runner", command)
            self.assertIn("--smoke", command)
            self.assertEqual(command[-3:], ["--provenance", "regenerated", "--smoke"])
            self.assertEqual(metadata["vectors_sha256"], fresh.digest(vectors))


if __name__ == "__main__":
    unittest.main()
