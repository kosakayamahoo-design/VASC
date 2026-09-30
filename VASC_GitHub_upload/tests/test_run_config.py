from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vasc_config_entry", ROOT / "evaluate.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class RunConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "run.json"
        self.example = json.loads((ROOT / "configs/local.example.json").read_text())

    def write_config(self, data=None):
        self.path.write_text(json.dumps(self.example if data is None else data), encoding="utf-8")
        return self.path

    def dry_run(self, *extra):
        output = io.StringIO()
        with redirect_stdout(output):
            entry.main(["--config", str(self.path), "--dry-run", *extra])
        return json.loads(output.getvalue())

    def test_config_paths_are_repository_relative(self):
        self.write_config()
        defaults = entry.load_run_config(self.path)
        self.assertEqual(defaults["weights"], ROOT / "checkpoints/vggt/model.pt")
        self.assertEqual(defaults["data_root"], ROOT / "data/nrgbd")
        self.assertTrue(defaults["structural_allocation"])
        self.assertEqual(defaults["gpu"], "0")

    def test_cli_overrides_config(self):
        self.write_config()
        record = self.dry_run("--mode", "topk", "--sparsity", ".65", "--gpu", "2",
                              "--no-structural-allocation", "--output", str(self.root / "custom"))
        self.assertEqual(record["allocation"]["requested_sparsity"], .65)
        self.assertFalse(record["allocation"]["enabled"])
        self.assertEqual(record["environment"]["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(record["environment"]["SPARSE_VGGT_COSA_VALUE_RISK_PAIR_DEBT"], "0")
        self.assertIn(str(self.root / "custom" / "metrics.json"), record["command"])
        self.assertFalse((self.root / "custom").exists())

    def test_both_allocation_interfaces_override_config(self):
        for initial in (False, True):
            self.write_config({**self.example, "structural_allocation": initial})
            for flags, expected in ((("--structural-allocation",), True),
                                    (("--no-structural-allocation",), False),
                                    (("--allocation", "paper"), True),
                                    (("--allocation", "uniform"), False)):
                self.assertEqual(self.dry_run(*flags)["allocation"]["enabled"], expected)

    def test_structural_default_and_existing_opt_out(self):
        for value in (None, False, True):
            data = dict(self.example)
            if value is None:
                data.pop("structural_allocation")
            else:
                data["structural_allocation"] = value
            self.write_config(data)
            settings = self.dry_run()["allocation"]
            self.assertEqual(settings["enabled"], value is not False)
            self.assertEqual(settings["applied"], value is not False)

    def test_auto_output_and_provenance(self):
        self.write_config({**self.example, "output_root": str(self.root / "outputs")})
        first, second = self.dry_run(), self.dry_run()
        self.assertNotEqual(first["command"], second["command"])
        metrics = Path(first["command"][first["command"].index("--metrics_json") + 1])
        self.assertEqual(metrics.parent.parent, self.root / "outputs")
        self.assertIn("vggt_nrgbd_vasc_s0.75_paper_", metrics.parent.name)
        self.assertFalse((self.root / "outputs").exists())
        self.assertEqual(first["config"]["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_pose_only_can_be_disabled_from_cli(self):
        self.write_config({**self.example, "pose_only": True})
        self.assertIn("--pose_only", self.dry_run()["command"])
        self.assertNotIn("--pose_only", self.dry_run("--no-pose-only")["command"])

    def test_bad_config_values(self):
        for key, value in (("structural_allocation", "false"), ("gpu", 0),
                           ("weights", ""), ("weights", None), ("frame_limit", 2.5),
                           ("stride", True), ("pose_only", "yes"), ("mode", "unknown"),
                           ("mode", "current"), ("mode", "memory"),
                           ("sparsity", True), ("sparisty", .75), ("allocation", "paper")):
            with self.subTest(key=key, value=value):
                self.write_config({**self.example, key: value})
                with self.assertRaises(ValueError):
                    entry.load_run_config(self.path)

    def test_non_object_and_malformed_json(self):
        for text in ("[]", "null", "{broken"):
            self.path.write_text(text)
            with self.assertRaises(ValueError):
                entry.load_run_config(self.path)

    def test_missing_explicit_config_fails(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.dry_run()

    def test_init_never_overwrites(self):
        with redirect_stdout(io.StringIO()):
            entry.main(["--init-config", "--config", str(self.path)])
        before = self.path.read_bytes()
        self.assertEqual(json.loads(before), self.example)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            entry.main(["--init-config", "--config", str(self.path)])
        self.assertEqual(self.path.read_bytes(), before)

    def test_init_rejects_run_flags(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            entry.main(["--init-config", "--config", str(self.path), "--mode", "dense"])
        self.assertFalse(self.path.exists())

    def test_default_local_config(self):
        config = self.root / "configs" / "local.json"
        config.parent.mkdir()
        config.write_text(json.dumps(self.example))
        output = io.StringIO()
        with patch.object(entry, "ROOT", self.root), redirect_stdout(output):
            entry.main(["--dry-run"])
        record = json.loads(output.getvalue())
        self.assertEqual(Path(record["config"]["path"]), config.resolve())
        self.assertIn(str(self.root / "checkpoints/vggt/model.pt"), record["command"])

    def test_no_config_gives_setup_message(self):
        output = io.StringIO()
        with patch.object(entry, "ROOT", self.root), redirect_stderr(output), self.assertRaises(SystemExit):
            entry.main([])
        self.assertIn("--init-config", output.getvalue())

    def test_plain_command_execution_and_no_overwrite(self):
        # Exercise the real subprocess/log/metadata path with a tiny fake backend.
        config = self.root / "configs/local.json"
        config.parent.mkdir()
        data = {**self.example, "output": "outputs/result"}
        config.write_text(json.dumps(data))
        (self.root / "checkpoints/vggt").mkdir(parents=True)
        (self.root / "checkpoints/vggt/model.pt").touch()
        for folder in ("data/nrgbd", "external/StreamVGGT/src/eval/mv_recon",
                       "external/vggt/vggt", "third_party/spargeattn/spas_sage_attn", "evaluation"):
            (self.root / folder).mkdir(parents=True)
        (self.root / "evaluation/eval_7scenes_sparse_vggt.py").write_text(
            "import json, pathlib, sys\n"
            "path=pathlib.Path(sys.argv[sys.argv.index('--metrics_json')+1])\n"
            "path.write_text(json.dumps({'fixture_only': True}))\n"
            "print('fixture evaluation finished')\n", encoding="utf-8")
        with patch.object(entry, "ROOT", self.root), redirect_stdout(io.StringIO()):
            entry.main([])
        out = self.root / "outputs/result"
        self.assertEqual(json.loads((out / "metrics.json").read_text()), {"fixture_only": True})
        self.assertIn("fixture evaluation finished", (out / "run.log").read_text())
        launch = json.loads((out / "launch.json").read_text())
        self.assertTrue(launch["allocation"]["applied"])
        snapshot = {p.name: p.read_bytes() for p in out.iterdir()}
        with patch.object(entry, "ROOT", self.root), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            entry.main([])
        self.assertEqual({p.name: p.read_bytes() for p in out.iterdir()}, snapshot)


if __name__ == "__main__":
    unittest.main()
