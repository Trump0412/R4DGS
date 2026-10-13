"""Exercise setup scripts in isolated local repositories, without network/GPU."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "posix" and shutil.which("bash") and shutil.which("git"), "Linux shell integration")
class SetupReproductionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()

    def git(self, target, *args):
        return subprocess.check_output(["git", "-C", str(target), *args], stderr=subprocess.DEVNULL, text=True).strip()

    def setup_dependencies(self):
        shutil.copy2(ROOT / "scripts/bootstrap_external.sh", self.root / "scripts/bootstrap_external.sh")
        env = os.environ.copy()
        for name, marker, variable in [("4DGaussians", "train.py", "REFERGAUSSIAN_4DGS_REF"),
                                       ("Grounded-SAM-2", "sam2/__init__.py", "REFERGAUSSIAN_GSAM2_REF")]:
            target = self.root / "external" / name
            (target / marker).parent.mkdir(parents=True)
            (target / marker).write_text("# test fixture\n")
            self.git(target, "init")
            self.git(target, "add", ".")
            self.git(target, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
            env[variable] = self.git(target, "rev-parse", "HEAD")
        return env

    def test_existing_clean_pins_are_accepted(self):
        result = subprocess.run(["bash", str(self.root / "scripts/bootstrap_external.sh")],
                                env=self.setup_dependencies(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_modified_existing_dependency_is_rejected_without_overwrite(self):
        env = self.setup_dependencies()
        target = self.root / "external/4DGaussians/train.py"
        target.write_text("# locally modified\n")
        result = subprocess.run(["bash", str(self.root / "scripts/bootstrap_external.sh")], env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be clean", result.stderr)
        self.assertEqual(target.read_text(), "# locally modified\n")

    def test_clean_wrong_revision_is_rejected(self):
        env = self.setup_dependencies()
        target = self.root / "external/4DGaussians"
        self.git(target, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "commit", "--allow-empty", "-m", "different revision")
        result = subprocess.run(["bash", str(self.root / "scripts/bootstrap_external.sh")], env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be clean", result.stderr)

    def test_hypernerf_preprocessing_uses_upstream_directory_and_python(self):
        for name in ("common.sh", "prepare_hypernerf.sh"):
            shutil.copy2(ROOT / "scripts" / name, self.root / "scripts" / name)
        upstream = self.root / "external/4DGaussians"
        (upstream / "scripts").mkdir(parents=True)
        env_root = self.root / "env"
        (env_root / "bin").mkdir(parents=True)
        (env_root / "bin/python").symlink_to(sys.executable)
        colmap = env_root / "bin/colmap"
        colmap.write_text("#!/bin/sh\nexit 0\n")
        colmap.chmod(0o755)
        (upstream / "colmap.sh").write_text('python scripts/hypernerf2colmap.py "$1"\n')
        (upstream / "scripts/hypernerf2colmap.py").write_text(
            'from pathlib import Path\nimport sys\np=Path(sys.argv[1])/"colmap/dense/workspace/fused.ply"\n'
            'p.parent.mkdir(parents=True)\np.write_text(str(Path.cwd()))\n')
        (upstream / "scripts/downsample_point.py").write_text('import sys,shutil\nshutil.copyfile(sys.argv[1],sys.argv[2])\n')
        scene = self.root / "data/hypernerf/misc/espresso"
        scene.mkdir(parents=True)
        (scene / "dataset.json").write_text("{}")
        env = {**os.environ, "GS_DATA_ROOT": str(self.root / "data"), "GS4D_ENV_PATH": str(env_root),
               "CONDA_PREFIX": str(env_root), "GS4D_CACHE_ROOT": str(self.root / "cache"),
               "PATH": str(env_root / "bin") + os.pathsep + os.environ["PATH"]}
        result = subprocess.run(["bash", str(self.root / "scripts/prepare_hypernerf.sh"), "misc", "espresso"],
                                cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((scene / "points3D_downsample2.ply").read_text(), str(upstream))


if __name__ == "__main__":
    unittest.main()
