"""Regression cases for a fresh upstream run without legacy sidecars."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from refergaussian.checkpoint import read_cfg_args, saved_model_args, checkpoint_identity
from refergaussian.run_identity import validate_query_ready_4dgs_run
from refergaussian.entitybank.tube_bank import (
    GaussianState, bank_from_samples, save_tube_bank, load_gaussian_state,
    load_scene_bank, prepare_scene_bank,
)


class StandardCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.saved = dict(source_path="scene", sh_degree=3,
                          kplanes_config={"output_coordinate_dim": 16, "resolution": [64, 64, 64, 150]},
                          multires=[1, 2, 4])
        (self.run / "cfg_args").write_text(str(argparse.Namespace(**self.saved)))
        self.iteration = self.run / "point_cloud" / "iteration_14000"
        self.iteration.mkdir(parents=True)
        (self.iteration / "point_cloud.ply").write_bytes(b"fixture-ply")
        (self.iteration / "deformation.pth").write_bytes(b"fixture-deformation")
        renders = self.run / "test" / "ours_14000" / "renders"
        renders.mkdir(parents=True)
        (renders / "00000.png").write_bytes(b"fixture-render")

    def sample_bank(self):
        times = np.linspace(0, 1, 9, dtype=np.float32)
        xyz = np.array([[0, 0, 1], [1, 1, 2]], dtype=np.float32)
        state = GaussianState(xyz, np.zeros((2, 3), np.float32), np.ones((2, 3), np.float32),
                              np.zeros((2, 1)), np.zeros((2, 1)), np.zeros((2, 3)),
                              np.zeros((2, 3)), np.zeros(2, np.float32))
        # A quadratic path and a static point; a zero or linear fallback cannot pass.
        trajectories = np.repeat(xyz[:, None], len(times), axis=1)
        trajectories[0, :, 0] = times ** 2
        bank = bank_from_samples(state, times, trajectories, np.zeros((2, len(times))))
        return state, bank

    def test_standard_run_needs_no_temporal_file(self):
        self.assertEqual(validate_query_ready_4dgs_run(self.run), [])
        self.assertFalse((self.iteration / "temporal_params.pth").exists())

    def test_missing_deformation_fails_preflight(self):
        (self.iteration / "deformation.pth").unlink()
        self.assertIn("deformation.pth", " ".join(validate_query_ready_4dgs_run(self.run)))

    def test_latest_incomplete_iteration_does_not_reuse_old_renders(self):
        later = self.run / "point_cloud" / "iteration_20000"
        later.mkdir()
        (later / "point_cloud.ply").write_bytes(b"later")
        errors = " ".join(validate_query_ready_4dgs_run(self.run))
        self.assertIn("iteration_20000", errors)
        self.assertIn("ours_20000", errors)

    def test_saved_network_structure_overrides_defaults(self):
        merged = saved_model_args(self.run, {"multires": [1, 2, 4, 8], "kplanes_config": {"output_coordinate_dim": 32}})
        self.assertEqual(merged["multires"], self.saved["multires"])
        self.assertEqual(merged["kplanes_config"], self.saved["kplanes_config"])
        self.assertEqual(merged["model_path"], str(self.run.resolve()))

    def test_cfg_is_not_executable(self):
        (self.run / "cfg_args").write_text("Namespace(source_path=__import__('os').getcwd())")
        with self.assertRaises(ValueError):
            read_cfg_args(self.run)

    def test_custom_scene_model_is_rejected(self):
        (self.run / "cfg_args").write_text("Namespace(source_path='scene', temporal_extent_enabled=True)")
        self.assertIn("Unsupported custom", " ".join(validate_query_ready_4dgs_run(self.run)))

    def test_exact_samples_and_derivatives_survive_round_trip(self):
        state, bank = self.sample_bank()
        np.testing.assert_allclose(state.acceleration[0], [2, 0, 0], atol=1e-5)
        np.testing.assert_allclose(state.velocity[1], 0, atol=1e-5)
        np.testing.assert_allclose(bank["gate"], 1)
        save_tube_bank(self.run, state, bank, 14000)
        loaded, config, iteration = load_gaussian_state(self.run)
        np.testing.assert_allclose(loaded.acceleration, state.acceleration)
        np.testing.assert_array_equal(load_scene_bank(self.run, 9)["trajectories"], bank["trajectories"])
        self.assertEqual(iteration, 14000)
        self.assertEqual(config["source_path"], "scene")

    def test_changed_weights_invalidate_bank_even_with_same_point_count(self):
        state, bank = self.sample_bank()
        save_tube_bank(self.run, state, bank, 14000)
        (self.iteration / "deformation.pth").write_bytes(b"different-deformation")
        with self.assertRaisesRegex(ValueError, "Stale"):
            load_scene_bank(self.run, 9)

    def test_legacy_cache_cannot_silently_supply_static_motion(self):
        root = self.run / "entitybank"
        root.mkdir()
        (root / "tube_bank.json").write_text(json.dumps({"schema_version": 1}))
        with self.assertRaisesRegex(ValueError, "Stale"):
            load_gaussian_state(self.run)

    def test_sample_count_is_part_of_contract(self):
        state, bank = self.sample_bank()
        save_tube_bank(self.run, state, bank, 14000)
        with self.assertRaisesRegex(ValueError, "--num-frames"):
            load_scene_bank(self.run, 64)

    def test_valid_cache_does_not_initialize_cuda(self):
        state, bank = self.sample_bank()
        save_tube_bank(self.run, state, bank, 14000)
        with patch("refergaussian.semantics.renderer_geometry._load_renderer_runtime", side_effect=AssertionError("CUDA must not be loaded")):
            self.assertEqual(prepare_scene_bank(self.run, 9), self.run / "entitybank")

    def test_first_preparation_samples_network_in_original_point_order(self):
        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value, dtype=np.float32)
            def __getitem__(self, index):
                return Tensor(self.value[index])
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.value
        state, _ = self.sample_bank()
        model = argparse.Namespace(get_xyz=Tensor(state.xyz), get_features=Tensor(state.rgb[:, None, :]),
                                   get_scaling=Tensor(state.spatial_scale), _opacity=Tensor(state.opacity))
        def evaluate(model, time, ids):
            centers = state.xyz[ids].copy()
            centers[:, 0] += np.sin(time * np.pi) * (ids + 1)
            return centers, np.zeros((len(ids), 6)), np.zeros(len(ids))
        with patch("refergaussian.semantics.renderer_geometry._load_renderer_runtime",
                   return_value=(None, model, None, None, 14000)), patch(
                       "refergaussian.semantics.renderer_geometry._renderer_state_at_time", side_effect=evaluate) as sampler:
            prepare_scene_bank(self.run, num_frames=5, chunk_size=1)
        self.assertEqual(sampler.call_count, 10)
        bank = load_scene_bank(self.run, 5)
        np.testing.assert_allclose(bank["trajectories"][:, 2, 0], [1, 3], atol=1e-6)
        np.testing.assert_allclose(bank["trajectories"][:, 0], state.xyz)
        self.assertFalse((self.iteration / "temporal_params.pth").exists())

    def test_nonfinite_geometry_is_rejected(self):
        state, bank = self.sample_bank()
        bank["trajectories"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            bank_from_samples(state, bank["time_values"], bank["trajectories"], np.zeros((2, 9)))


if __name__ == "__main__":
    unittest.main()
