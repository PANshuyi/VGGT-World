# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import torch
import torch.nn as nn
import os
from iopath.common.file_io import g_pathmgr
from wcmatch import fnmatch





# ------------------------------------------------------------
# Glob‑matching flags (behave like the Unix shell) 
# ------------------------------------------------------------
GLOB_FLAGS = (
    fnmatch.CASE       # case‑sensitive
    | fnmatch.DOTMATCH # '*' also matches '.'
    | fnmatch.EXTMATCH # extended patterns like *(foo|bar)
    | fnmatch.SPLIT    # "pat1|pat2" works out‑of‑the‑box
)




class DDPCheckpointSaver:
    def __init__(
        self,
        checkpoint_folder: str,
        checkpoint_names: List[str],
        rank: int,
        epoch: int,
    ):
        super().__init__()
        self.checkpoint_folder = checkpoint_folder
        self.checkpoint_names = checkpoint_names
        self.worker_id = rank
        self.epoch = epoch

    def save_checkpoint(
        self,
        model: nn.Module,
        **kwargs: Any,
    ) -> None:
        checkpoint = dict(**kwargs)
        checkpoint["model"] = model.state_dict()

        if self.worker_id == 0:
            for ckpt_name in self.checkpoint_names:
                checkpoint_path = os.path.join(
                    self.checkpoint_folder, f"{ckpt_name}.pt"
                )
                logging.info(
                    f"Saving checkpoint at epoch {self.epoch} to {checkpoint_path}"
                )
                robust_torch_save(checkpoint, checkpoint_path)


def load_checkpoint_with_backup(
    checkpoint_path: str,
    map_location: Any = "cpu",
) -> Tuple[Dict[str, Any], str]:
    """Load a checkpoint, falling back to ``.bak`` after an interrupted save.

    Returns both the checkpoint and the path that was actually loaded so the
    caller can make recovery visible in the training log.
    """
    candidates = (checkpoint_path, checkpoint_path + ".bak")
    errors = []
    for candidate in candidates:
        if not g_pathmgr.isfile(candidate):
            continue
        try:
            with g_pathmgr.open(candidate, "rb") as f:
                return torch.load(f, map_location=map_location), candidate
        except Exception as error:
            errors.append(f"{candidate}: {error!r}")
            logging.exception("Failed to load checkpoint candidate %s", candidate)

    details = "; ".join(errors) if errors else "neither primary nor backup exists"
    raise RuntimeError(f"Unable to load checkpoint {checkpoint_path}: {details}")


def resume_epoch_from_checkpoint(checkpoint: Mapping[str, Any]) -> Optional[int]:
    """Return the next epoch to train, including compatibility with old files."""
    if "next_epoch" in checkpoint:
        return int(checkpoint["next_epoch"])
    if "prev_epoch" in checkpoint:
        return int(checkpoint["prev_epoch"]) + 1
    if "epoch" in checkpoint:
        return int(checkpoint["epoch"]) + 1
    return None


def restore_optimizer_states(optimizers: Sequence[Any], optimizer_states: Any) -> None:
    """Restore one or multiple ``OptimizerWrapper`` instances safely."""
    if isinstance(optimizer_states, Mapping):
        optimizer_states = [optimizer_states]
    elif isinstance(optimizer_states, (list, tuple)):
        optimizer_states = list(optimizer_states)
    else:
        raise TypeError(
            "checkpoint['optimizer'] must be a state dict or a list of state dicts"
        )

    if len(optimizer_states) != len(optimizers):
        raise ValueError(
            "Checkpoint optimizer count does not match current training: "
            f"{len(optimizer_states)} vs {len(optimizers)}"
        )
    for optimizer, optimizer_state in zip(optimizers, optimizer_states):
        optimizer.optimizer.load_state_dict(optimizer_state)



def robust_torch_save(checkpoint: Dict[str, Any], checkpoint_path: str) -> None:
    """
    A more robust version of torch.save that works better with preemptions
    and corruptions if a job is preempted during save.
    """
    # Move the existing checkpoint to a backup location
    backup_checkpoint_path = checkpoint_path + ".bak"
    backup_checkpoint_path_saved = g_pathmgr.isfile(backup_checkpoint_path)
    if g_pathmgr.exists(checkpoint_path) and not backup_checkpoint_path_saved:
        g_pathmgr.mv(checkpoint_path, backup_checkpoint_path)
        backup_checkpoint_path_saved = True
    # Save the checkpoint
    with g_pathmgr.open(checkpoint_path, "wb") as f:
        torch.save(checkpoint, f)
    # Remove the backup checkpoint
    if backup_checkpoint_path_saved:
        g_pathmgr.rm(backup_checkpoint_path)
