# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os


# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


import contextlib
import gc
import json
import logging
import math
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from train_utils.checkpoint import (
    DDPCheckpointSaver,
    load_checkpoint_with_backup,
    restore_optimizer_states,
    resume_epoch_from_checkpoint,
)
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from train_utils.optimizer import construct_optimizers
from train_utils.geometry_visualizer import (
    FixedProbeRegistry,
    GeometryProbeVisualizer,
    batch_string_value,
    fallback_ratio_from_batch,
)
from train_utils.geometry_metrics import (
    GeometryMetricEvaluator,
    build_eval_id,
    deduplicate_metric_rows,
    gather_metric_rows,
    summarize_metric_rows,
    write_metric_reports,
)


class Trainer:
    """
    A generic trainer for DDP training. This should naturally support multi-node training.

    This class orchestrates the entire training and validation process, including:
    - Setting up the distributed environment (DDP).
    - Initializing the model, optimizers, loss functions, and data loaders.
    - Handling checkpointing for resuming training.
    - Executing the main training and validation loops.
    - Logging metrics and visualizations to TensorBoard.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        **kwargs,
    ):
        """
        Initializes the Trainer.

        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging: Hydra config for logging (TensorBoard, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: "train" for training and validation, "val" for validation only.
            device: "cuda" or "cpu".
            seed_value: A random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run validation.
            distributed: Hydra config for DDP settings.
            cuda: Hydra config for CUDA-specific settings (e.g., cuDNN).
            limit_train_batches: Limit the number of training batches per epoch (for debugging).
            limit_val_batches: Limit the number of validation batches per epoch (for debugging).
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            env_variables: Dictionary of environment variables to set.
            accum_steps: Number of steps to accumulate gradients before an optimizer step.
        """
        self._setup_env_variables(env_variables)
        self._setup_timers()

        # Store Hydra configurations
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim

        # Store hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value
        
        # 'where' tracks training progress from 0.0 to 1.0 for schedulers
        self.where = 0.0

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        # Setup logging directory and configure logger
        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        assert is_dist_avail_and_initialized(), "Torch distributed needs to be initialized before calling the trainer."

        # Instantiate components (model, loss, etc.)
        self._setup_components()
        self._setup_dataloaders()

        # Move model to the correct device
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # Construct optimizers (after moving model to device)
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        # 优先恢复当前实验目录中的最新训练 checkpoint。否则配置中
        # 始终存在的初始 VGGT 权重会遮住自动续训 checkpoint。
        auto_resume = bool(getattr(self.checkpoint_conf, "auto_resume", True))
        auto_ckpt_path = (
            get_resume_checkpoint(self.checkpoint_conf.save_dir)
            if auto_resume
            else None
        )
        configured_ckpt_path = self.checkpoint_conf.resume_checkpoint_path
        ckpt_path = auto_ckpt_path or configured_ckpt_path
        if ckpt_path is not None:
            if auto_ckpt_path is not None:
                logging.info(f"Auto-resume checkpoint found: {auto_ckpt_path}")
            self._load_resuming_checkpoint(ckpt_path)

        # Wrap the model with DDP
        self._setup_ddp_distributed_training(distributed, device)
        
        # Barrier to ensure all processes are synchronized before starting
        dist.barrier()

    def _setup_timers(self):
        """Initializes timers for tracking total elapsed time."""
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """Sets environment variables from the configuration."""
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        """Initializes the distributed process group and configures PyTorch backends."""
        if torch.cuda.is_available():
            # Configure CUDA backend settings for performance
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        # Initialize the DDP process group
        dist.init_process_group(
            backend=distributed_conf.backend,
            timeout=timedelta(minutes=distributed_conf.timeout_mins)
        )
        self.rank = dist.get_rank()

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Loads model state and, for training checkpoints, all progress state."""
        logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")

        # 若上次任务恰好在写 checkpoint 时中断，主文件可能不完整。
        # loader 会在主文件无法解析时自动回退到上一个 .bak。
        checkpoint, loaded_path = load_checkpoint_with_backup(
            ckpt_path, map_location="cpu"
        )
        if loaded_path != ckpt_path:
            logging.warning(f"Recovered checkpoint from backup: {loaded_path}")
        
        # Load model state
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        missing, unexpected = self.model.load_state_dict(
            model_state_dict, strict=self.checkpoint_conf.strict
        )
        if self.rank == 0:
            logging.info(f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")

        # 恢复全部 optimizer。既兼容历史单 optimizer dict，也支持列表格式。
        if self.mode != "val" and "optimizer" in checkpoint:
            logging.info(f"Loading optimizer state dict (rank {self.rank})")
            restore_optimizer_states(self.optims, checkpoint["optimizer"])

        # checkpoint 在一个 epoch 完整训练后保存，因此续训必须从
        # next_epoch 开始，不能重复上一个 epoch。同时兼容旧字段。
        resumed_epoch = resume_epoch_from_checkpoint(checkpoint)
        if resumed_epoch is not None:
            self.epoch = resumed_epoch
        self.steps = checkpoint["steps"] if "steps" in checkpoint else {"train": 0, "val": 0}
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

        # Load AMP scaler state if available
        if self.mode != "val" and self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        logging.info(
            "Checkpoint restored: next_epoch=%d, train_step=%d, val_step=%d",
            self.epoch,
            int(self.steps.get("train", 0)),
            int(self.steps.get("val", 0)),
        )

    def _setup_device(self, device: str):
        """Sets up the device for training (CPU or CUDA)."""
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _setup_components(self):
        """Initializes all core training components using Hydra configs."""
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.steps = {'train': 0, 'val': 0}

        # Instantiate components from configs
        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)

        # 固定验证探针独立于旧的通用 log_visuals 开关。首轮选中的
        # seq_name 会写入 manifest，续训和后续验证均复用同一批样本。
        self.geometry_visualizer = None
        self.visual_probe_registry = None
        geometry_visual_conf = self.logging_conf.get("geometry_visuals", None)
        if (
            self.rank == 0
            and geometry_visual_conf is not None
            and bool(geometry_visual_conf.get("enabled", False))
        ):
            self.geometry_visualizer = GeometryProbeVisualizer(geometry_visual_conf)
            self.visual_probe_registry = FixedProbeRegistry(
                Path(self.logging_conf.log_dir) / "visual_probe_manifest.json",
                geometry_visual_conf.get("probe_plan", {}),
            )

        # 正式几何指标在所有 rank 的验证 no_grad 路径上计算，
        # 最后只把脱离计算图的 Python 标量汇总到 rank 0。
        # 这一对象不持有模型参数，也不参与 forward/loss/backward。
        self.geometry_metric_evaluator = None
        geometry_metric_conf = self.logging_conf.get("geometry_metrics", None)
        if (
            geometry_metric_conf is not None
            and bool(geometry_metric_conf.get("enabled", False))
        ):
            self.geometry_metric_evaluator = GeometryMetricEvaluator(
                geometry_metric_conf
            )

        # Freeze specified model parameters if any
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(
                f"[Start] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(
                f"[Done] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )

        # Log model summary on rank 0
        if self.rank == 0:
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary saved to {model_summary_path}")

        logging.info("Successfully initialized training components.")

    def _setup_dataloaders(self):
        """Initializes train and validation datasets and dataloaders."""
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(
                self.data_conf.get('val', None), _recursive_=False
            )
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value

        if self.mode in ["train"]:
            self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        """Wraps the model with DistributedDataParallel (DDP)."""
        assert isinstance(self.model, torch.nn.Module)

        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )

    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None):
        """
        Saves a training checkpoint.

        Args:
            epoch: The current epoch number.
            checkpoint_names: A list of names for the checkpoint file (e.g., "checkpoint_latest").
                              If None, saves "checkpoint" and "checkpoint_{epoch}" on frequency.
        """
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and int(epoch) % self.checkpoint_conf.save_freq == 0
                and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_content = {
            "checkpoint_version": 2,
            "completed_epoch": int(epoch),
            "next_epoch": int(epoch) + 1,
            # 保留旧字段，使旧版读取器仍可识别该 checkpoint。
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
        }
        
        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        # Save the checkpoint for DDP only
        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        model = (
            self.model.module
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else self.model
        )

        saver.save_checkpoint(
            model=model,
            ema_models = None,
            skip_saving_parameters=[],
            **checkpoint_content,
        )




    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        """Retrieves keys for scalar values to be logged for a given phase."""
        if self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        return []

    def run(self):
        """Main entry point to start the training or validation process."""
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        if self.mode == "train":
            self.run_train()
            # Optionally run a final validation after all training is done
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def run_train(self):
        """Runs the main training loop over all epochs."""
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)
            
            # 所有 DDP rank 必须使用相同 epoch seed，确保每一步选择相同的
            # 数据集及动态输入形状，避免分布式 collective 发生形状不一致。
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch))
            self.train_epoch(dataloader)
            
            # Save checkpoint after each training epoch
            self.save_checkpoint(self.epoch)

            # Clean up memory
            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Run validation at the specified frequency
            # Skips validation after the last training epoch, as it can be run separately.
            if self.epoch % self.val_epoch_freq == 0 and self.epoch < self.max_epochs - 1:
                self.run_val()
            
            self.epoch += 1
        
        self.epoch -= 1

    def run_val(self):
        """Runs a full validation epoch if a validation dataset is available."""
        if not self.val_dataset:
            logging.info("No validation dataset configured. Skipping validation.")
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch))
        self.val_epoch(dataloader)
        
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


    @torch.no_grad()
    def val_epoch(self, val_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'val'
        if self.visual_probe_registry is not None:
            self.visual_probe_registry.start_validation()
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        # 每个 rank 只暂存自己分片上的逐 clip CPU 标量。验证
        # 结束后再一次性 gather，避免逐 batch collective 拖慢训练。
        # loss 也携带唯一 eval_id，因此 DistributedSampler 为补齐
        # rank 而重复的末尾 clip 不会被二次计权。
        local_validation_loss_rows: list[dict[str, Any]] = []
        local_geometry_metric_rows: list[dict[str, Any]] = []
        
        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        end = time.time()

        iters_per_epoch = len(val_loader)
        limit_val_batches = (
            iters_per_epoch
            if self.limit_val_batches is None
            else self.limit_val_batches
        )

        for data_iter, batch in enumerate(val_loader):
            if data_iter >= limit_val_batches:
                break
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)
            
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch, phase=phase)
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            amp_type = self.optim_conf.amp.amp_dtype
            assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
            if amp_type == "bfloat16":
                amp_type = torch.bfloat16
            else:
                amp_type = torch.float16
            
            # compute output
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    val_loss_dict, val_predictions = self._step(
                        batch,
                        self.model,
                        phase,
                        loss_meters,
                        return_predictions=True,
                    )

            if self.geometry_metric_evaluator is not None:
                try:
                    local_geometry_metric_rows.extend(
                        self.geometry_metric_evaluator.compute_batch(
                            batch, val_predictions
                        )
                    )
                except Exception:
                    # 指标是旁路评估；单个异常 clip 只记录错误，
                    # 不中断长时间多机训练或影响 loss。
                    logging.exception(
                        "Failed to compute validation geometry metrics at batch %d",
                        data_iter,
                    )

            # 分数据集的 camera/depth/point 记录“对 objective 的实际贡献”，
            # 因此包含 MultitaskLoss 配置的 task weight。
            weighted_totals = self._get_weighted_validation_loss_totals(val_loss_dict)
            local_validation_loss_rows.append(
                self._build_validation_loss_row(
                    batch,
                    val_loss_dict,
                    weighted_totals,
                )
            )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        # 整轮验证结束后才做一次跨 rank 归并，并按
        # dataset+scene+camera+clip_start 去重。TensorBoard 横坐标
        # 使用当前训练 global step，便于直接比较 checkpoint。
        self._log_exact_validation_loss_summaries(local_validation_loss_rows)
        self._save_validation_geometry_metrics(local_geometry_metric_rows)

        if self.rank == 0 and self.visual_probe_registry is not None:
            missing_probes = self.visual_probe_registry.missing_slots()
            if missing_probes:
                logging.warning(
                    "Fixed visual probes not found in this validation run: %s",
                    ", ".join(missing_probes),
                )

        return True

    def _save_validation_geometry_metrics(
        self,
        local_rows: list[dict[str, Any]],
    ) -> None:
        """汇总、精确去重并保存七数据集的逐 clip 几何指标。"""
        if self.geometry_metric_evaluator is None:
            return

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        gathered_rows = gather_metric_rows(local_rows, self.rank, world_size)
        if self.rank != 0:
            return
        assert gathered_rows is not None

        rows = deduplicate_metric_rows(gathered_rows)
        dataset_names = self._get_validation_dataset_names()
        summary = summarize_metric_rows(rows, dataset_names)
        train_step = int(self.steps.get("train", 0))
        report_dir = (
            Path(self.logging_conf.log_dir)
            / "validation_metrics"
            / f"epoch_{int(self.epoch):04d}_train_step_{train_step:08d}"
        )
        try:
            write_metric_reports(rows, summary, report_dir)
            # 与 loss summary 使用同一个 train global step，便于
            # 不同 checkpoint 在 TensorBoard 上直接对比。
            for dataset_name, values in summary.items():
                for metric_name, value in values.items():
                    if metric_name == "num_clips" or not math.isfinite(value):
                        continue
                    self.tb_writer.log(
                        f"Metrics/val_by_dataset/{dataset_name}/{metric_name}",
                        value,
                        train_step,
                    )
            self.tb_writer.flush()
            logging.info(
                "Validation metric rows: gathered=%d, unique_clips=%d",
                len(gathered_rows),
                len(rows),
            )
        except Exception:
            # JuiceFS 偶发写失败不能反过来改变训练结果。
            logging.exception(
                "Failed to save validation geometry metric reports to %s",
                report_dir,
            )

    @staticmethod
    def _mean_finite_rows(rows: Sequence[Mapping[str, Any]], key: str) -> float:
        """计算逐 clip 标量的有限值平均；空集返回 NaN。"""
        values = [
            float(row[key])
            for row in rows
            if key in row and math.isfinite(float(row[key]))
        ]
        return sum(values) / len(values) if values else math.nan

    def _build_validation_loss_row(
        self,
        batch: Mapping[str, Any],
        loss_dict: Mapping[str, Any],
        weighted_totals: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """把 batch_size=1 的验证 loss 转成可精确去重的逐 clip 记录。

        验证配置故意固定 batch_size=1，因为不同 clip 可以具有
        不同帧数，且一个 batch 的平均 loss 无法无损还原到单 clip。
        """
        batch_size = int(batch["extrinsics"].shape[0])
        if batch_size != 1:
            raise ValueError(
                "Exact validation loss summaries require validation batch_size=1, "
                f"got {batch_size}"
            )

        raw_dataset_name = batch.get("dataset_name")
        if isinstance(raw_dataset_name, str):
            dataset_name = raw_dataset_name
        elif isinstance(raw_dataset_name, (list, tuple)):
            dataset_name = str(raw_dataset_name[0])
        else:
            raise TypeError(
                "Validation batch dataset_name must be str/list/tuple, got "
                f"{type(raw_dataset_name).__name__}"
            )

        row: Dict[str, Any] = {
            "eval_id": build_eval_id(batch, 0),
            "dataset_name": dataset_name,
        }
        for scalar_key in self._get_scalar_log_keys("val"):
            data_key = "objective" if scalar_key == "loss_objective" else scalar_key
            if data_key not in loss_dict:
                continue
            raw_value = loss_dict[data_key]
            row[scalar_key] = (
                float(raw_value.detach().item())
                if torch.is_tensor(raw_value)
                else float(raw_value)
            )
        for total_name, raw_value in weighted_totals.items():
            row[f"weighted_{total_name}"] = (
                float(raw_value.detach().item())
                if torch.is_tensor(raw_value)
                else float(raw_value)
            )
        return row

    def _log_exact_validation_loss_summaries(
        self,
        local_rows: list[dict[str, Any]],
    ) -> None:
        """跨 rank 汇总、去除 DDP 补齐 clip，记录全局与分数据集 loss。"""
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        gathered_rows = gather_metric_rows(local_rows, self.rank, world_size)
        if self.rank != 0:
            return
        assert gathered_rows is not None
        rows = deduplicate_metric_rows(gathered_rows)
        if not rows:
            logging.warning("No validation loss rows were collected")
            return

        train_step = int(self.steps.get("train", 0))
        global_payload = {}
        for scalar_key in self._get_scalar_log_keys("val"):
            value = self._mean_finite_rows(rows, scalar_key)
            if not math.isfinite(value):
                continue
            tag_name = "objective" if scalar_key == "loss_objective" else scalar_key
            self.tb_writer.log(f"Summary/val/{tag_name}", value, train_step)
            global_payload[tag_name] = value

        dataset_payload = {}
        for dataset_name in self._get_validation_dataset_names():
            dataset_rows = [
                row for row in rows if row.get("dataset_name") == dataset_name
            ]
            dataset_payload[dataset_name] = {}
            for total_name in ("objective", "camera", "depth_total", "point_total"):
                value = self._mean_finite_rows(
                    dataset_rows, f"weighted_{total_name}"
                )
                if not math.isfinite(value):
                    continue
                self.tb_writer.log(
                    f"Summary/val_by_dataset/{dataset_name}/{total_name}",
                    value,
                    train_step,
                )
                dataset_payload[dataset_name][total_name] = value

        self.tb_writer.flush()
        logging.info(
            "Exact validation loss summary at train_step=%d: gathered=%d, "
            "unique_clips=%d, global=%s",
            train_step,
            len(gathered_rows),
            len(rows),
            ", ".join(
                f"{key}={value:.6f}" for key, value in global_payload.items()
            ),
        )
        logging.debug("Per-dataset validation loss summary: %s", dataset_payload)

    def _get_validation_dataset_names(self) -> List[str]:
        """返回已配置的验证数据集名，供所有 rank 构造同序 meter。"""
        composed_dataset = getattr(self.val_dataset, "dataset", None)
        concatenated_dataset = getattr(composed_dataset, "base_dataset", None)
        base_datasets = getattr(concatenated_dataset, "datasets", ())
        dataset_names = []
        for dataset in base_datasets:
            dataset_name = getattr(dataset, "dataset_name", None)
            if dataset_name is None:
                raise ValueError(
                    f"Validation dataset {type(dataset).__name__} misses dataset_name"
                )
            dataset_name = str(dataset_name)
            if dataset_name in dataset_names:
                raise ValueError(f"Duplicate validation dataset_name: {dataset_name}")
            dataset_names.append(dataset_name)
        return dataset_names

    def _get_loss_task_weight(self, task_name: str) -> float:
        """读取 MultitaskLoss 中 camera/depth/point 对 objective 的权重。"""
        task_config = getattr(self.loss, task_name, None)
        if task_config is None:
            return 1.0
        if hasattr(task_config, "get"):
            return float(task_config.get("weight", 1.0))
        return float(task_config["weight"])

    def _get_weighted_validation_loss_totals(
        self,
        loss_dict: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """构造可与 objective 直接对账的四个验证量。"""
        totals = {}
        if "objective" in loss_dict:
            totals["objective"] = loss_dict["objective"]
        if "loss_camera" in loss_dict:
            totals["camera"] = (
                loss_dict["loss_camera"] * self._get_loss_task_weight("camera")
            )

        depth_keys = ("loss_conf_depth", "loss_reg_depth", "loss_grad_depth")
        if all(key in loss_dict for key in depth_keys):
            totals["depth_total"] = (
                sum(loss_dict[key] for key in depth_keys)
                * self._get_loss_task_weight("depth")
            )

        point_keys = ("loss_conf_point", "loss_reg_point", "loss_grad_point")
        if all(key in loss_dict for key in point_keys):
            totals["point_total"] = (
                sum(loss_dict[key] for key in point_keys)
                * self._get_loss_task_weight("point")
            )
        return totals

    def train_epoch(self, train_loader):        
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'train'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        for config in self.gradient_clipper.configs: 
            param_names = ",".join(config['module_names'])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")


        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )
        
        if self.gradient_clipper is not None:
            # setup gradient clipping at the beginning of training
            self.gradient_clipper.setup_clipping(self.model)

        for data_iter, batch in enumerate(train_loader):
            if data_iter >= limit_train_batches:
                break
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch, phase=phase)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            accum_steps = self.accum_steps

            if accum_steps==1:
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            self._run_steps_on_batch_chunks(
                chunked_batches, phase, loss_meters
            )

            # compute gradient and do SGD step
            assert data_iter < limit_train_batches
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.where = float(exact_epoch) / self.max_epochs
            
            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(
                    f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                )
                    
            # Log schedulers
            if self.steps[phase] % self.logging_conf.log_freq == 0:
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (
                                f"{i}_"
                                if len(self.optims) > 1
                                else (
                                    "" + f"{j}_"
                                    if len(optim.optimizer.param_groups) > 1
                                    else ""
                                )
                            )
                            self.tb_writer.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )
                self.tb_writer.log(
                    os.path.join("Optim", "where"),
                    self.where,
                    self.steps[phase],
                )

            # Clipping gradients and detecting diverging gradients
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)

                grad_norm_dict = self.gradient_clipper(model=self.model)

                for key, grad_norm in grad_norm_dict.items():
                    loss_meters[f"Grad/{key}"].update(grad_norm)

            # Optimizer step
            for optim in self.optims:   
                self.scaler.step(optim.optimizer)
            self.scaler.update()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        return True

    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],
        phase: str,
        loss_meters: Dict[str, AverageMeter],
    ):
        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward
        """        
        
        for optim in self.optims:   
            optim.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)

        amp_type = self.optim_conf.amp.amp_dtype
        assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
        if amp_type == "bfloat16":
            amp_type = torch.bfloat16
        else:
            amp_type = torch.float16
        
        for i, chunked_batch in enumerate(chunked_batches):
            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1
                else contextlib.nullcontext()
            )

            with ddp_context:
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    loss_dict = self._step(
                        chunked_batch, self.model, phase, loss_meters
                    )


                loss = loss_dict["objective"]
                if not math.isfinite(loss.item()):
                    error_msg = f"Loss is {loss.item()}; aborting before optimizer.step()"
                    logging.error(error_msg)
                    # 原逻辑只 return，外层仍可能执行 optimizer.step，
                    # 从而把非有限梯度写入权重。显式抛错可保住最后一份
                    # 健康 checkpoint，也不改变任何正常 batch 的训练结果。
                    raise FloatingPointError(error_msg)

                loss /= accum_steps
                self.scaler.scale(loss).backward()


    def _apply_batch_repetition(self, batch: Mapping) -> Mapping:
        """
        Applies a data augmentation by concatenating the original batch with a
        flipped version of itself.
        """
        # 这些张量含有时序维，第二份样本同时反转时间顺序。
        tensor_keys = [
            "images", "depths", "extrinsics", "intrinsics", 
            "cam_points", "world_points", "point_masks", "ids",
            "use_lidar_proj_depth",
        ]
        # 字符串元数据只有 batch 维，直接复制即可。它们只用于
        # 日志、可视化与验证唯一 ID，不会传入模型或 loss。
        string_keys = ["seq_name", "dataset_name", "scene_name", "camera_name"]
        
        for key in tensor_keys:
            if key in batch:
                original_tensor = batch[key]
                batch[key] = torch.concatenate([original_tensor, 
                                                torch.flip(original_tensor, dims=[1])], 
                                                dim=0)
        
        for key in string_keys:
            if key in batch:
                batch[key] = batch[key] * 2

        # 以下是 (B,) 样本属性，没有时序维，因此只复制
        # batch 维而不能像图像那样沿 dim=1 翻转。
        for scalar_key in ("is_metric", "clip_start"):
            if scalar_key in batch:
                batch[scalar_key] = torch.concatenate(
                    [batch[scalar_key], batch[scalar_key]], dim=0
                )
        
        return batch

    def _process_batch(self, batch: Mapping, phase: str = "train"):
        # 训练和验证显式读取各自 common_config。当前两者
        # 的尺度策略一致，但这样可防止未来修改 val 配置时
        # 被静默地用 train 配置覆盖。
        phase_conf = self.data_conf.val if phase == "val" else self.data_conf.train
        common_config = phase_conf.common_config
        if common_config.repeat_batch:
            batch = self._apply_batch_repetition(batch)
        
        # Normalize camera extrinsics and points. The function returns new tensors.
        normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = \
            normalize_camera_extrinsics_and_points_batch(
                extrinsics=batch["extrinsics"],
                cam_points=batch["cam_points"],
                world_points=batch["world_points"],
                depths=batch["depths"],
                point_masks=batch["point_masks"],
                # metric 版不再对可信米制数据做逐场景归一化。
                # 具体策略放在 dataset common_config 中，便于和归一化版对照。
                scale_mode=getattr(
                    common_config,
                    "geometry_scale_mode",
                    "vggt",
                ),
                is_metric=batch.get("is_metric"),
                metric_scale_factor=float(
                    getattr(
                        common_config,
                        "metric_scale_factor",
                        0.1,
                    )
                ),
            )

        # Replace the original values in the batch with the normalized ones.
        batch["extrinsics"] = normalized_extrinsics
        batch["cam_points"] = normalized_cam_points
        batch["world_points"] = normalized_world_points
        batch["depths"] = normalized_depths

        return batch

    def _step(
        self,
        batch,
        model: nn.Module,
        phase: str,
        loss_meters: dict,
        return_predictions: bool = False,
    ):
        """
        Performs a single forward pass, computes loss, and logs results.
        
        Returns:
            A dictionary containing the computed losses.
        """
        # Forward pass
        y_hat = model(images=batch["images"])
        
        # Loss computation
        loss_dict = self.loss(y_hat, batch)
        
        # Combine all data for logging
        log_data = {**y_hat, **loss_dict, **batch}

        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        self._log_tb_visuals(log_data, phase, self.steps[phase])
        self._log_geometry_probes(batch, y_hat, phase)

        self.steps[phase] += 1
        if return_predictions:
            return loss_dict, y_hat
        return loss_dict

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict):
        """Updates average meters and logs scalar values to TensorBoard."""
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = data['extrinsics'].shape[0]
        
        for key in keys_to_log:
            # MultitaskLoss 的总损失键是 ``objective``，而日志对外统一
            # 使用 ``loss_objective``。在唯一的逐 batch 日志入口做别名
            # 映射，使训练和验证都不会漏记或重复计数。
            data_key = "objective" if key == "loss_objective" else key
            if data_key in data:
                value = (
                    data[data_key].item()
                    if torch.is_tensor(data[data_key])
                    else data[data_key]
                )
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                    self.tb_writer.log(f"Values/{phase}/{key}", value, step)

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        """Logs image or video visualizations to TensorBoard."""
        if not (
            self.logging_conf.log_visuals
            and (phase in self.logging_conf.log_visual_frequency)
            and self.logging_conf.log_visual_frequency[phase] > 0
            and (step % self.logging_conf.log_visual_frequency[phase] == 0)
            and (self.logging_conf.visuals_keys_to_log is not None)
        ):
            return

        if phase in self.logging_conf.visuals_keys_to_log:
            keys_to_log = self.logging_conf.visuals_keys_to_log[phase][
                "keys_to_log"
            ]
            assert (
                len(keys_to_log) > 0
            ), "Need to include some visual keys to log"
            modality = self.logging_conf.visuals_keys_to_log[phase][
                "modality"
            ]
            assert modality in [
                "image",
                "video",
            ], "Currently only support video or image logging"

            name = f"Visuals/{phase}"

            visuals_to_log = torchvision.utils.make_grid(
                [
                    torchvision.utils.make_grid(
                        batch[key][0],  # Ensure batch[key][0] is tensor and has at least 3 dimensions
                        nrow=self.logging_conf.visuals_per_batch_to_log,
                    )
                    for key in keys_to_log if key in batch and batch[key][0].dim() >= 3
                ],
                nrow=1,
            ).clamp(-1, 1)

            visuals_to_log = visuals_to_log.cpu()
            if visuals_to_log.dtype == torch.bfloat16:
                visuals_to_log = visuals_to_log.to(torch.float16)
            visuals_to_log = visuals_to_log.numpy()

            self.tb_writer.log_visuals(
                name, visuals_to_log, step, self.logging_conf.video_logging_fps
            )

    def _log_geometry_probes(
        self,
        batch: Mapping[str, Any],
        predictions: Mapping[str, torch.Tensor],
        phase: str,
    ) -> None:
        """仅在 rank 0 为固定验证 clip 写入米制几何对比图。

        这里显式分开 ``batch`` 和 ``predictions``，避免旧日志代码用
        ``{**prediction, **batch}`` 合并时 GT world_points 覆盖预测值。
        """
        if (
            phase != "val"
            or self.rank != 0
            or self.geometry_visualizer is None
            or self.visual_probe_registry is None
        ):
            return

        batch_size = int(batch["images"].shape[0])
        train_step = int(self.steps.get("train", 0))
        for sample_index in range(batch_size):
            dataset_name = batch_string_value(batch, "dataset_name", sample_index)
            seq_name = batch_string_value(batch, "seq_name", sample_index)
            eval_id = build_eval_id(batch, sample_index)
            fallback_ratio = fallback_ratio_from_batch(batch, sample_index)
            selected = self.visual_probe_registry.consider(
                dataset_name,
                eval_id,
                seq_name,
                fallback_ratio,
            )
            if selected is None:
                continue

            slot_index, role = selected
            try:
                rendered = self.geometry_visualizer.render(
                    batch,
                    predictions,
                    sample_index,
                )
                frame_ids = batch.get("ids")
                if torch.is_tensor(frame_ids):
                    frame_ids_text = ", ".join(
                        str(int(value))
                        for value in frame_ids[sample_index].detach().cpu().reshape(-1)
                    )
                else:
                    frame_ids_text = "unavailable"
                tag_root = (
                    f"Probes/{dataset_name}/"
                    f"probe_{slot_index:02d}_{role}"
                )
                for visual_name, image in rendered.items():
                    self.tb_writer.log_visuals(
                        f"{tag_root}/{visual_name}",
                        image,
                        train_step,
                    )
                self.tb_writer.log_text(
                    f"{tag_root}/00_metadata",
                    "\n".join(
                        [
                            f"seq_name: `{seq_name}`",
                            f"eval_id: `{eval_id}`",
                            f"role: `{role}`",
                            f"frame_ids: `{frame_ids_text}`",
                            f"LiDAR fallback frame ratio: `{fallback_ratio:.3f}`",
                            "metric visualization: GT and prediction restored from x0.1 training scale to meters",
                            "BEV coordinates: x=right, z=forward in the first camera C0 frame",
                        ]
                    ),
                    train_step,
                )
                self.tb_writer.flush()
            except Exception:
                # 可视化失败不能让长时间多机训练或全量验证中断。
                logging.exception(
                    "Failed to render geometry probe %s/%s", dataset_name, seq_name
                )




def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    """Splits a batch into smaller chunks for gradient accumulation."""
    if accum_steps == 1:
        return [batch]
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]

def is_sequence_of_primitives(data: Any) -> bool:
    """Checks if data is a sequence of primitive types (str, int, float, bool)."""
    return (
        isinstance(data, Sequence)
        and not isinstance(data, str)
        and len(data) > 0
        and isinstance(data[0], (str, int, float, bool))
    )

def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    """
    Recursively splits tensors and sequences within a data structure into chunks.

    Args:
        data: The data structure to split (e.g., a dictionary of tensors).
        chunk_id: The index of the chunk to retrieve.
        num_chunks: The total number of chunks to split the data into.

    Returns:
        A chunk of the original data structure.
    """
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        # either a tensor or a list of primitive objects
        # assert len(data) % num_chunks == 0
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    elif isinstance(data, str):
        # NOTE: this is a hack to support string keys in the batch
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data
