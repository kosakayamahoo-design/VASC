import ast
import hashlib
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vasc_evaluate", ROOT / "evaluate.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class ReleaseTests(unittest.TestCase):
    def arguments(self, *extra):
        return entry.parser().parse_args([
            "--backbone", "vggt", "--dataset", "nrgbd", "--weights", "model.pt",
            "--data-root", "data/nrgbd", "--output", "outputs/test", *extra])

    def test_main_matrix(self):
        for model in ("vggt", "pi3"):
            for mode in entry.MODES:
                for sparsity in entry.CONFIG["sparsities"]:
                    with self.subTest(model=model, mode=mode, sparsity=sparsity):
                        args = self.arguments("--backbone", model, "--mode", mode,
                                              "--sparsity", str(sparsity))
                        cmd, env = entry.build_launch(args)
                        self.assertEqual(cmd[cmd.index("--kf_every") + 1], "10")
                        self.assertNotIn("--num_frames", cmd)
                        self.assertEqual(env["SPARSE_VGGT_QK_PV_THRESHOLD"], "1e10")
                        self.assertEqual("--layer_sparsity" in cmd,
                                         mode != "dense" and sparsity > .70)
                        self.assertTrue(entry.allocation_settings(args)["enabled"])
                        if model == "pi3":
                            self.assertEqual(cmd[cmd.index("--max_points") + 1], "200000")
                        else:
                            self.assertEqual(cmd[cmd.index("--max_points") + 1], "999999")

    def test_profiles(self):
        for model, count, low in (("vggt", 24, 11), ("pi3", 18, 12)):
            for sparse in (.75, .85):
                values = list(map(float, entry.layer_sparsity(model, sparse, "paper").split(",")))
                self.assertEqual(len(values), count)
                self.assertAlmostEqual(sum(values) / count, sparse, places=9)
                self.assertAlmostEqual(values[low], sparse - .05)
            self.assertIsNone(entry.layer_sparsity(model, .65, "paper"))
            self.assertIsNone(entry.layer_sparsity(model, .85, "uniform"))

    def test_structural_switch_matrix(self):
        for model in ("vggt", "pi3"):
            for mode in entry.MODES:
                for sparsity in entry.CONFIG["sparsities"]:
                    for enabled in (False, True):
                        with self.subTest(model=model, mode=mode, sparsity=sparsity,
                                          enabled=enabled):
                            base = ("--backbone", model, "--mode", mode,
                                    "--sparsity", str(sparsity))
                            flag = "--structural-allocation" if enabled else "--no-structural-allocation"
                            args = self.arguments(*base, flag)
                            legacy = self.arguments(*base, "--allocation",
                                                    "paper" if enabled else "uniform")
                            self.assertEqual(entry.build_launch(args), entry.build_launch(legacy))
                            settings = entry.allocation_settings(args)
                            self.assertEqual(settings["enabled"], enabled)
                            self.assertEqual(settings["applied"], enabled and mode != "dense" and sparsity > .70)
                            ratios = settings["layer_sparsities"]
                            self.assertEqual(len(ratios), entry.CONFIG["structural"][model]["layers"])
                            self.assertAlmostEqual(sum(ratios) / len(ratios),
                                                   0 if mode == "dense" else sparsity, places=9)
                            if not settings["applied"]:
                                self.assertEqual(len(set(ratios)), 1)

    def test_allocation_changes_only_layer_schedule(self):
        for model in ("vggt", "pi3"):
            for mode in entry.MODES:
                with self.subTest(model=model, mode=mode):
                    base = ("--backbone", model, "--mode", mode)
                    on, on_env = entry.build_launch(self.arguments(*base, "--structural-allocation"))
                    off, off_env = entry.build_launch(self.arguments(*base, "--no-structural-allocation"))
                    self.assertEqual(on_env, off_env)
                    if "--layer_sparsity" in on:
                        index = on.index("--layer_sparsity")
                        del on[index:index + 2]
                    self.assertEqual(on, off)

    def test_allocation_config_default_and_override(self):
        for default in (False, True):
            with patch.dict(entry.CONFIG["structural"], {"enabled": default}):
                settings = entry.allocation_settings(self.arguments())
                self.assertEqual(settings["enabled"], default)
                self.assertEqual(settings["applied"], default)
                for flags, enabled in ((("--structural-allocation",), True),
                                       (("--no-structural-allocation",), False),
                                       (("--allocation", "paper"), True),
                                       (("--allocation", "uniform"), False)):
                    self.assertEqual(entry.allocation_settings(self.arguments(*flags))["enabled"], enabled)

    def test_allocation_cli_conflicts(self):
        for flag in ("--structural-allocation", "--no-structural-allocation"):
            for legacy in ("paper", "uniform"):
                for args in ((flag, "--allocation", legacy), ("--allocation", legacy, flag)):
                    with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        self.arguments(*args)

    def test_allocation_threshold_and_extreme_sparsity(self):
        for model in ("vggt", "pi3"):
            for sparsity in (0, .70, .7001, .85):
                args = self.arguments("--backbone", model, "--sparsity", str(sparsity),
                                      "--structural-allocation")
                self.assertEqual(entry.allocation_settings(args)["applied"], sparsity > .70)
            with self.assertRaises(ValueError):
                entry.build_launch(self.arguments("--backbone", model, "--sparsity", ".99",
                                                  "--structural-allocation"))
            for extra in (("--no-structural-allocation",), ("--mode", "dense", "--structural-allocation")):
                args = self.arguments("--backbone", model, "--sparsity", ".99", *extra)
                command, _ = entry.build_launch(args)
                self.assertNotIn("--layer_sparsity", command)

    def test_dry_run_records_allocation(self):
        for extra, policy in ((("--structural-allocation",), "paper"),
                              (("--no-structural-allocation",), "uniform"),
                              (("--structural-allocation", "--mode", "dense"), "dense"),
                              (("--structural-allocation", "--sparsity", ".65"), "uniform")):
            args = self.arguments("--dry-run", *extra)
            output = io.StringIO()
            with patch.object(entry, "parser") as parser_mock, redirect_stdout(output):
                parser_mock.return_value.parse_args.return_value = args
                entry.main()
            record = json.loads(output.getvalue())
            self.assertEqual(record["allocation"], entry.allocation_settings(args))
            self.assertEqual(record["allocation"]["effective_policy"], policy)
            command = record["command"]
            if record["allocation"]["applied"]:
                self.assertEqual(list(map(float, command[command.index("--layer_sparsity") + 1].split(","))),
                                 record["allocation"]["layer_sparsities"])
            else:
                self.assertNotIn("--layer_sparsity", command)

    def test_environment_isolation(self):
        with patch.dict(os.environ, {"SPARSE_VGGT_BAD_EXPERIMENT": "1", "PYTHONPATH": "old"}):
            _, env = entry.build_launch(self.arguments())
        self.assertNotIn("SPARSE_VGGT_BAD_EXPERIMENT", env)
        self.assertNotEqual(env["PYTHONPATH"], "old")

    def test_supported_modes(self):
        self.assertEqual(entry.MODES, ("dense", "topk", "vasc"))
        full = entry.method_environment("vggt", "vasc")
        self.assertEqual(full["SPARSE_VGGT_COSA_VALUE_RISK_PAIR_DEBT"], "1")
        for mode in ("current", "memory"):
            with self.assertRaises(ValueError):
                entry.method_environment("vggt", mode)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.arguments("--mode", mode)

    def test_bad_inputs(self):
        for extra in (("--stride", "0"), ("--sparsity", "nan"), ("--sparsity", "1"),
                      ("--sequence", "seq-01"), ("--frame-limit", "0")):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                entry.build_launch(self.arguments(*extra))

    def test_sources_parse(self):
        for path in ROOT.rglob("*.py"):
            if any(p in {"external", "build", ".venv", "outputs"} for p in path.parts):
                continue
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_source_manifest(self):
        manifest = json.loads((ROOT / "source_manifest.json").read_text())
        for item in manifest["files"]:
            with self.subTest(path=item["path"]):
                actual = hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
                self.assertEqual(actual, item["sha256"])

    def test_pair_selector_has_no_execution_variant(self):
        source = (ROOT / "src/sparse_vggt/utils/sparse_wrapper.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        selector = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "get_cosa_value_risk_pair_debt_mask")
        self.assertNotIn("execution_mode", [arg.arg for arg in selector.args.kwonlyargs])
        self.assertNotIn("SPARSE_VGGT_COSA_ABLATE_VALUE", source)
        scheduler = (ROOT / "src/sparse_vggt/models/attention.py").read_text(encoding="utf-8")
        self.assertNotIn("SPARSE_VGGT_DEBT_SCHEDULER_ABLATION", scheduler)

    def test_no_binary_or_private_paths(self):
        for folder in (ROOT / "src", ROOT / "evaluation", ROOT / "third_party"):
            for path in folder.rglob("*"):
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                self.assertNotIn(path.suffix, {".so", ".pyd", ".pt", ".safetensors"})
                if path.suffix in {".py", ".cpp", ".cu", ".cuh", ".h"}:
                    text = path.read_text(encoding="utf-8")
                    self.assertNotIn("/home/" + "fanqingkong", text)
                    self.assertNotIn("/data/user/" + "fanqingkong", text)


if __name__ == "__main__":
    unittest.main()
