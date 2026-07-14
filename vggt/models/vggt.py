# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub
import torch.nn.functional as F

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.models.fm import Flowmatching, FMConfig

class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=True,
        fm_train_mode: str = "stage_1",
        fm_pred_steps: int = 50,
        fm_pred_weight: float = 0.1,
        # 本次修改：新增联合几何微调开关，并显式配置历史/未来图像槽位数和每时间步相机数。
        # Keep the historical ``*_frames`` names for checkpoint/config
        # compatibility; in multi-view training each count means image slots.
        enable_fm_geometry_supervision: bool = False,
        fm_history_frames: int = 2,
        fm_future_frames: int = 2,
        fm_views_per_timestep: int = 1,
        fm_dynamic_views: bool = False,
        fm_history_timesteps: int = 2,
        fm_future_timesteps: int = 2,
    ):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

        # ---- FM init: 8-layer condition, each layer token dim = embed_dim (1024) ----
        fm_cfg = FMConfig(
            in_dim=embed_dim,
            model_dim=1024,
            depth=8,
            n_heads=16,
            mlp_ratio=4.0,
            attn_drop=0.0,
            proj_drop=0.0,
            n_max=20000,
            # 本次修改：将 FM 的序列元数据与未来图像槽位数对齐（实际 shape 仍由运行时张量决定）。
            t_frames=fm_future_frames,
            # 本次修改：将每时间步视角数传入 FM RoPE，使双目槽位使用稳定的分数时间位置。
            views_per_timestep=fm_views_per_timestep,
            # use_patch_pos=True,
        )
        self.fm = Flowmatching(fm_cfg)
        self.fm = self.fm.to(dtype=torch.bfloat16)

        self.fm_train_mode = fm_train_mode
        self.fm_pred_steps = fm_pred_steps
        self.fm_pred_weight = fm_pred_weight
        self.fm_mix_progress = 0.0
        # 本次修改：保存联合微调配置；关闭开关时保持原 FM-only 接口。
        # Optional joint fine-tuning path.  The legacy FM-only behavior remains
        # the default so existing checkpoints/scripts keep the same interface.
        self.enable_fm_geometry_supervision = enable_fm_geometry_supervision
        self.fm_history_frames = int(fm_history_frames)
        self.fm_future_frames = int(fm_future_frames)
        self.fm_views_per_timestep = int(fm_views_per_timestep)
        # 本次修改：动态多数据集模式固定物理历史/未来步数，由每个 batch 的 V 推导图像槽位数。
        # These are plain Python metadata, not Parameters/Buffers, so old
        # checkpoints keep exactly the same state_dict keys and tensor shapes.
        self.fm_dynamic_views = bool(fm_dynamic_views)
        self.fm_history_timesteps = int(fm_history_timesteps)
        self.fm_future_timesteps = int(fm_future_timesteps)

    # 本次修改：统一解析固定布局与 source bucket 提供的运行时多相机布局。
    @staticmethod
    def _layout_int(value, name: str):
        """Convert a scalar layout value to int and reject mixed/non-scalar input."""
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(
                    f"{name} must be one scalar for the whole batch, got "
                    f"shape={tuple(value.shape)}."
                )
            value = value.item()
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        converted = int(value)
        if float(value) != converted or converted <= 0:
            raise ValueError(
                f"{name} must be a positive integer, got {value!r}."
            )
        return converted

    # 本次修改：统一校验“历史图像槽位 -> 未来图像槽位”布局，并支持相邻 batch 使用不同相机数。
    def _validate_fm_sequence_layout(
        self,
        images: torch.Tensor,
        views_per_timestep=None,
        history_timesteps=None,
        future_timesteps=None,
    ):
        """Validate FM layout and return slots, views, and physical time counts.

        Tensor shapes cannot reveal whether a slot is left or right camera, so
        the dataset remains responsible for the documented time-major order.
        """
        runtime_layout = self.fm_dynamic_views or any(
            value is not None
            for value in (
                views_per_timestep,
                history_timesteps,
                future_timesteps,
            )
        )

        if runtime_layout:
            history_timesteps = self._layout_int(
                self.fm_history_timesteps
                if history_timesteps is None
                else history_timesteps,
                "history_timesteps",
            )
            future_timesteps = self._layout_int(
                self.fm_future_timesteps
                if future_timesteps is None
                else future_timesteps,
                "future_timesteps",
            )
            views_per_timestep = self._layout_int(
                views_per_timestep, "views_per_timestep"
            )
            if views_per_timestep is None:
                physical_steps = history_timesteps + future_timesteps
                if images.shape[1] % physical_steps:
                    raise ValueError(
                        "Cannot infer views_per_timestep from the batch: "
                        f"S={images.shape[1]} is not divisible by "
                        f"history+future timesteps={physical_steps}."
                    )
                views_per_timestep = images.shape[1] // physical_steps
            history_slots = history_timesteps * views_per_timestep
            future_slots = future_timesteps * views_per_timestep
        else:
            history_slots = self.fm_history_frames
            future_slots = self.fm_future_frames
            views_per_timestep = self._layout_int(
                self.fm_views_per_timestep, "fm_views_per_timestep"
            )
            history_timesteps = history_slots // views_per_timestep
            future_timesteps = future_slots // views_per_timestep

        if history_slots <= 0 or future_slots <= 0:
            raise ValueError(
                "FM history/future image-slot counts must be positive, got "
                f"history={history_slots}, future={future_slots}."
            )
        if views_per_timestep <= 0:
            raise ValueError(
                "fm_views_per_timestep must be positive, got "
                f"{views_per_timestep}."
            )
        if history_slots % views_per_timestep or future_slots % views_per_timestep:
            raise ValueError(
                "FM image-slot counts must be divisible by the number of views "
                f"per timestep: history={history_slots}, future={future_slots}, "
                f"views={views_per_timestep}."
            )
        if history_slots != future_slots:
            raise ValueError(
                "Current FM implementation requires equal history/future image-slot "
                f"counts, got history={history_slots}, future={future_slots}."
            )

        total_slots = history_slots + future_slots
        if images.shape[1] < total_slots:
            raise ValueError(
                f"Need at least {total_slots} image slots for FM training, "
                f"got {images.shape[1]}."
            )
        if self.enable_fm_geometry_supervision and images.shape[1] != total_slots:
            raise ValueError(
                "Joint history/future geometry supervision requires exactly "
                f"{total_slots} image slots, got {images.shape[1]}. Set dataset "
                "common_config.img_nums accordingly."
            )
        return (
            history_slots,
            future_slots,
            total_slots,
            views_per_timestep,
            history_timesteps,
            future_timesteps,
        )

    # 本次修改：统一计算矩形图像的 patch 网格，供 FM RoPE 和 part2 共用。
    def _get_patch_hw(self, images: torch.Tensor):
        """Return the spatial patch grid shared by FM RoPE and VGGT part2."""
        H_img, W_img = images.shape[-2:]
        patch_size = self.aggregator.patch_size
        if isinstance(patch_size, (tuple, list)):
            patch_h, patch_w = patch_size
        else:
            patch_h = patch_w = patch_size
        return H_img // patch_h, W_img // patch_w

    # 本次修改：封装“latent → 原有 part2 → 原有几何 heads”的可微解码路径。
    def _decode_latent_sequence(
        self,
        latent_sequence: torch.Tensor,
        images: torch.Tensor,
        patch_start_idx: int,
        patch_hw,
        history_slots: int,
        views_per_timestep: int,
    ):
        """Decode an early VGGT latent with the existing part2 and task heads.

        This is the same path already used by ``demo/kitti_demo.py`` and the
        evaluation scripts.  During joint training, detaches in part2 are
        disabled so geometry losses remain differentiable with respect to both
        the observed latent and the FM-predicted future latent.
        """
        # 本次修改：联合训练时关闭 part2 内部 detach，使几何 loss 能回传到 FM/part1。
        aggregator_dtype = next(self.aggregator.parameters()).dtype
        aggregated_tokens_list, _ = self.aggregator.part2(
            [latent_sequence.to(aggregator_dtype)],
            patch_hw=patch_hw,
            detach_intermediates=False,
        )

        # 本次修改：记录历史/未来图像槽位分界和相机数，并用原有 heads 解码所有启用的几何任务。
        predictions = {
            "_history_frames": history_slots,
            "_views_per_timestep": views_per_timestep,
        }

        # Each pretrained head may have a different parameter dtype.  Casting
        # the shared DPT features preserves gradients while avoiding dtype
        # mismatches when AMP is disabled or a checkpoint is stored in fp32.
        # 本次修改：复用原 camera head 输出多阶段位姿，供历史/未来 camera loss 监督。
        if self.camera_head is not None:
            camera_dtype = next(self.camera_head.parameters()).dtype
            camera_features = [x.to(camera_dtype) for x in aggregated_tokens_list]
            pose_enc_list = self.camera_head(camera_features)
            predictions["pose_enc"] = pose_enc_list[-1]
            predictions["pose_enc_list"] = pose_enc_list

        # 本次修改：复用原 depth head 输出深度与置信度，供历史/未来 depth loss 监督。
        if self.depth_head is not None:
            depth_dtype = next(self.depth_head.parameters()).dtype
            depth_features = [x.to(depth_dtype) for x in aggregated_tokens_list]
            depth, depth_conf = self.depth_head(
                depth_features,
                images=images.to(depth_dtype),
                patch_start_idx=patch_start_idx,
            )
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf

        # 本次修改：数据集含 world_points GT 时，可复用原 point head 输出三维点与置信度。
        if self.point_head is not None:
            point_dtype = next(self.point_head.parameters()).dtype
            point_features = [x.to(point_dtype) for x in aggregated_tokens_list]
            world_points, world_points_conf = self.point_head(
                point_features,
                images=images.to(point_dtype),
                patch_start_idx=patch_start_idx,
            )
            predictions["world_points"] = world_points
            predictions["world_points_conf"] = world_points_conf

        return predictions

    def _forward_stage_2(self, images: torch.Tensor, query_points: torch.Tensor = None):
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)
        if images.shape[1] < 5:
            raise ValueError(f"Need at least 5 frames, got {images.shape[1]}")

        # 1) condition tokens from 12 (no grad, used only for pred sampling)
        
        cond_12, _ = self.aggregator.part1(images[:, 0:2])
        cond_12 = [x.detach() for x in cond_12]

        cond_23, _ = self.aggregator.part1(images[:, 1:3])
        cond_23 = [x.detach() for x in cond_23]

        tgt_1234_stage_list, _ = self.aggregator.part1(images[:, 0:4])
        tgt34_layers = [x[:, 2:4, :, :] for x in tgt_1234_stage_list]

        # 2) gt tokens for 34 and 56 (need gradients, avoid seeing future frames)
        tgt12345_stage_list, _ = self.aggregator.part1(images)
        pseudo_gt34_layers = [x[:, 2:4, :, :] for x in tgt12345_stage_list]
        gt45_layers = [x[:, 3:5, :, :] for x in tgt12345_stage_list]

        # 3) infer pred(34) using 12 as condition (detach and store)
        _, _, _, H_img, W_img = images.shape
        patch_size = self.aggregator.patch_size
        if isinstance(patch_size, (tuple, list)):
            patch_h, patch_w = patch_size
        else:
            patch_h = patch_w = patch_size
        patch_hw = (H_img // patch_h, W_img // patch_w)

        dtype_fm = next(self.fm.parameters()).dtype if self.fm is not None else pseudo_gt34_layers[0].dtype
        shape_like = torch.zeros_like(pseudo_gt34_layers[0], dtype=dtype_fm)

        with torch.no_grad():
            cond_stage_list_fm = [x.to(dtype_fm) for x in cond_12]
            pred_layers = self.fm.sample_euler(
                cond_layers=cond_stage_list_fm,
                shape_like=shape_like,
                # steps=self.fm_pred_steps,
                steps=25,
                patch_hw=patch_hw,
            )
        pred34 = torch.cat(pred_layers, dim=1).to(pseudo_gt34_layers[0].dtype).detach()

        # 4) fuse condition for stage-2 training:
        #    use gt for frame-2, only mix frame-3 (gt from cond_23, pred from pred34)
        mix_weight = getattr(self, "fm_mix_progress", None)
        if mix_weight is None:
            mix_weight = float(self.fm_pred_weight)
        mix_weight = max(0.0, min(1.0, float(mix_weight)))
        gt2 = cond_12[0][:, 0:1, :, :]
        gt3 = cond_23[0][:, 1:2, :, :]
        pred3 = pred34[:, 0:1, :, :]

        mix3 = mix_weight * pred3 + (1.0 - mix_weight) * gt3

        # mix3 = pred3
        cond_mix = torch.cat([gt2, mix3], dim=1)
        cond_layers = [cond_mix]

        # 5) train to predict 56 using fused condition
        fm_loss = self.fm.loss_rectified_multilayer(
            x1_layers=gt45_layers,
            cond_layers=cond_layers,
            patch_hw=patch_hw,
        )

        loss_dict = {
            "train/loss_total": fm_loss,
            "train/loss_fm": fm_loss,
        }
        return fm_loss, loss_dict


    # 本次修改：主 forward 接收运行时 V/Ht/Ft，并把动态布局贯穿 part1、FM、part2 与几何 heads。
    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
        *,
        views_per_timestep=None,
        history_timesteps=None,
        future_timesteps=None,
    ):
        if self.fm_train_mode == "stage_2":
            # 本次修改：旧 stage-2 含固定帧切片，动态 source layout 尚未接入时显式拒绝。
            if any(
                value is not None
                for value in (
                    views_per_timestep,
                    history_timesteps,
                    future_timesteps,
                )
            ):
                raise ValueError(
                    "Runtime multi-view layouts are supported by stage_1 joint "
                    "training only; legacy stage_2 still uses fixed frame slices."
                )
            return self._forward_stage_2(images, query_points=query_points)

        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        # 本次修改：从当前 source bucket 解析 V，并动态得到 2V 历史/2V 未来槽位。
        (
            history_slots,
            future_slots,
            total_slots,
            runtime_views,
            _,
            _,
        ) = self._validate_fm_sequence_layout(
            images,
            views_per_timestep=views_per_timestep,
            history_timesteps=history_timesteps,
            future_timesteps=future_timesteps,
        )

        # 本次修改：将输入显式分为历史观测和未来监督槽位。
        # Keep the decoded sequence aligned with the history/future supervision.
        images = images[:, :total_slots]
        history_images = images[:, :history_slots]

        # 本次修改：历史 part1 分支不 detach，FM loss 和几何 loss 均可更新该路径。
        # Current-observation branch: intentionally NOT detached (case 1).
        # Consequently both FM loss and downstream geometry losses can update
        # the current-observation part1 path.
        cond_stage_list, patch_start_idx = self.aggregator.part1(history_images)
        cond_stage_list = [x for x in cond_stage_list]

        # 本次修改：只截断未来 target 分支，它提供数值监督但不接收任何反向梯度。
        # Joint case-1 future-target branch: this is the only branch detached.
        # The sliced future target values supervise FM, but neither FM loss nor
        # decoded geometry losses can backpropagate through this full-sequence
        # teacher part1 call.
        # no_grad is autograd-equivalent to detaching the future target, but it
        # avoids retaining a second, unused full-sequence part1 graph in memory.
        if self.enable_fm_geometry_supervision:
            with torch.no_grad():
                target_stage_list, _ = self.aggregator.part1(images)
        else:
            # Preserve the original FM-only gradient behavior for old configs;
            # case 1 is activated explicitly by the joint-training switch.
            target_stage_list, _ = self.aggregator.part1(images)
        target_z_list = [
            x[:, history_slots:total_slots]
            for x in target_stage_list
        ]
        if any(x.shape[1] != future_slots for x in target_z_list):
            raise ValueError(
                "Future target latent does not match the configured image-slot "
                f"count {future_slots}."
            )

        # 本次修改：统一 FM 输入 dtype；类型转换保留历史分支的 autograd 连接。
        # FM is stored in bf16. ``to`` preserves the current-branch autograd
        # path and, in joint mode, does not reattach the future target.
        fm_dtype = next(self.fm.parameters()).dtype
        cond_z_list = [x.to(fm_dtype) for x in cond_stage_list]
        target_z_list = [x.to(fm_dtype) for x in target_z_list]

        # 本次修改：联合模式同时取得 FM latent loss 和可微的 clean future latent。
        # ---- FM loss in z-space ----
        patch_hw = self._get_patch_hw(images)

        fm_result = self.fm.loss_rectified_multilayer(
            x1_layers=target_z_list,
            cond_layers=cond_z_list,
            patch_hw=patch_hw,
            return_x1_pred=self.enable_fm_geometry_supervision,
            views_per_timestep=runtime_views,
        )

        # 本次修改：拼接真实历史 latent 和 FM 预测 latent，再复用 part2/heads 生成几何输出。
        if self.enable_fm_geometry_supervision:
            fm_loss, pred_future_z = fm_result

            # Reuse the established inference path: real history latent plus
            # differentiable FM future latent -> VGGT part2 -> geometry heads.
            # The history latent is deliberately not detached here, so global
            # attention in part2 can also send geometry gradients to part1.
            pred_future_z = pred_future_z.to(cond_stage_list[0].dtype)
            combined_latent = torch.cat(
                [cond_stage_list[0], pred_future_z],
                dim=1,
            )
            geometry_predictions = self._decode_latent_sequence(
                combined_latent,
                images=images,
                patch_start_idx=patch_start_idx,
                patch_hw=patch_hw,
                history_slots=history_slots,
                views_per_timestep=runtime_views,
            )
        else:
            fm_loss = fm_result

        loss = fm_loss
        loss_dict = {
            "train/loss_total": loss,
            "train/loss_fm": fm_loss,
        }

        # 本次修改：联合模式额外返回几何预测，交由 Trainer 计算 MultitaskLoss。
        if self.enable_fm_geometry_supervision:
            # Trainer applies the existing MultitaskLoss to these predictions
            # and replaces loss_total with FM + history/future geometry losses.
            return loss, loss_dict, geometry_predictions
        return loss, loss_dict
            
    def forward_vggt(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions
