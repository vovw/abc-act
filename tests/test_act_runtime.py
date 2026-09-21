import unittest
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np

from abc_act.policies.act import CursorBatchSampler, TemporalEnsembler, RenderedACTDataset


class CursorTests(unittest.TestCase):
    def test_recycled_and_resumed_loaders_preserve_complete_passes(self):
        # Include a partial final batch and a dataset smaller than one batch.
        for size, batch_size in ((23, 8), (24, 8), (3, 8)):
            expected = []
            for epoch in range(3):
                batches = list(CursorBatchSampler(size, batch_size, 42, epoch * size))
                self.assertEqual(sorted(sum(batches, [])), list(range(size)))
                expected.extend(batches)
            seen, actual = 0, []
            while seen < size * 3:
                batches = list(CursorBatchSampler(size, batch_size, 42, seen, max_batches=2))
                actual.extend(batches)
                seen += sum(map(len, batches))
            self.assertEqual(actual, expected)

    def test_prefetch_does_not_change_committed_cursor(self):
        full = list(CursorBatchSampler(31, 4, 7))
        prefetched = iter(CursorBatchSampler(31, 4, 7))
        for _ in range(5):
            next(prefetched)
        # Only two batches were optimized; discard the unconsumed prefetch.
        self.assertEqual(list(CursorBatchSampler(31, 4, 7, 8)), full[2:])


class EnsembleTests(unittest.TestCase):
    def test_alignment_includes_zero_predictions(self):
        ensemble = TemporalEnsembler(0)
        np.testing.assert_allclose(ensemble.update([[0, 0], [2, 4], [4, 8]]), [0, 0])
        np.testing.assert_allclose(ensemble.update([[6, 8], [8, 10], [10, 12]]), [4, 6])
        np.testing.assert_allclose(ensemble.update([[12, 14], [14, 16], [16, 18]]), [8, 32 / 3])

    def test_oldest_prediction_gets_reference_act_weight(self):
        ensemble = TemporalEnsembler(np.log(2))
        ensemble.update([[0], [3]])
        # Weights are 1 for the old forecast, 1/2 for the new forecast.
        np.testing.assert_allclose(ensemble.update([[9], [0]]), [5])

    def test_history_is_bounded_and_reset_between_episodes(self):
        ensemble = TemporalEnsembler()
        for _ in range(100):
            ensemble.update(np.zeros((3, 14)))
        self.assertEqual(len(ensemble.pending), 2)
        ensemble.reset()
        self.assertFalse(ensemble.pending)


class RenderedDataTests(unittest.TestCase):
    def test_normalization_padding_and_worker_pickle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root / "episode"
            episode.mkdir()
            (episode / "episode_metadata.json").write_text(json.dumps({"task_name": "test"}))
            manifest = dict(chunk_size=20, camera_keys=["top", "left", "right"],
                            camera_backend="mujoco", splits={})
            for split in ("train", "val"):
                directory = root / split
                directory.mkdir()
                np.save(directory / "images.npy", np.full((2, 3, 3, 168, 224), 100, np.uint8))
                np.save(directory / "states.npy", np.full((2, 14), 5, np.float32))
                np.save(directory / "actions.npy", np.full((2, 20, 14), 7, np.float32))
                mask = np.zeros((2, 20), bool)
                mask[1, 1:] = True
                np.save(directory / "is_pad.npy", mask)
                manifest["splits"][split] = dict(samples=2, episodes=[str(episode)])
            (root / "manifest.json").write_text(json.dumps(manifest))
            stats = dict(state_mean=np.ones(14), state_std=np.full(14, 2),
                         action_mean=np.ones(14), action_std=np.full(14, 3))
            dataset = RenderedACTDataset(root, "train", stats)
            sample = dataset[1]
            np.testing.assert_allclose(sample["state"], 2)
            np.testing.assert_allclose(sample["actions"][0], 2)
            np.testing.assert_array_equal(sample["actions"][1:], 0)
            self.assertEqual(tuple(sample["images"].shape), (3, 3, 224, 224))
            packed = pickle.dumps(dataset)
            self.assertLess(len(packed), 10000)
            restored = pickle.loads(packed)
            self.assertIsNone(restored._arrays)
            np.testing.assert_array_equal(restored[1]["actions"], sample["actions"])


if __name__ == "__main__":
    unittest.main()
