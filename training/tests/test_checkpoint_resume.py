import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from train_utils.checkpoint import (  # noqa: E402
    load_checkpoint_with_backup,
    restore_optimizer_states,
    resume_epoch_from_checkpoint,
    robust_torch_save,
)


class CheckpointResumeTest(unittest.TestCase):
    def test_resume_starts_from_next_epoch_and_supports_old_fields(self):
        self.assertEqual(resume_epoch_from_checkpoint({"next_epoch": 8}), 8)
        self.assertEqual(resume_epoch_from_checkpoint({"prev_epoch": 7}), 8)
        self.assertEqual(resume_epoch_from_checkpoint({"epoch": 7}), 8)
        self.assertIsNone(resume_epoch_from_checkpoint({}))

    def test_restore_single_optimizer_state_dict(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        source = torch.optim.AdamW([parameter], lr=3e-4)
        parameter.grad = torch.tensor([2.0])
        source.step()

        target_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        target = torch.optim.AdamW([target_parameter], lr=1e-5)
        wrappers = [SimpleNamespace(optimizer=target)]
        restore_optimizer_states(wrappers, source.state_dict())

        self.assertEqual(target.param_groups[0]["lr"], 3e-4)
        self.assertTrue(target.state_dict()["state"])

    def test_corrupt_primary_falls_back_to_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            backup_path = Path(f"{checkpoint_path}.bak")
            checkpoint_path.write_bytes(b"interrupted checkpoint")
            torch.save({"next_epoch": 12}, backup_path)

            checkpoint, loaded_path = load_checkpoint_with_backup(str(checkpoint_path))

            self.assertEqual(checkpoint["next_epoch"], 12)
            self.assertEqual(loaded_path, str(backup_path))

    def test_robust_save_replaces_checkpoint_and_cleans_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            robust_torch_save({"next_epoch": 1}, str(checkpoint_path))
            robust_torch_save({"next_epoch": 2}, str(checkpoint_path))

            checkpoint, loaded_path = load_checkpoint_with_backup(str(checkpoint_path))
            self.assertEqual(checkpoint["next_epoch"], 2)
            self.assertEqual(loaded_path, str(checkpoint_path))
            self.assertFalse(Path(f"{checkpoint_path}.bak").exists())


if __name__ == "__main__":
    unittest.main()
