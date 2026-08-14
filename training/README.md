# Training

This is a re-implementation of our framework for training VGGT. This document shows how to set up the environment and run VGGT training. I have aimed to faithfully reproduce the original training framework, but please open an issue if anything looks off.

## 1. Prerequisites

Before you begin, ensure you have completed the following steps:

1. **Install VGGT as a package:**
   ```bash
   pip install -e .
   ```

2. **Prepare the seven driving datasets:**
   - OpenScene, Waymo, DDAD, nuScenes and KITTI use the filtered parquet,
     image, aligned-depth and projected-depth outputs produced by DVGT.
   - MVS-Synth uses the raw `GTAV_1080_new` directory.
   - VKITTI uses the raw `SceneXX/condition/...` directory.

## 2. Configuration

After downloading the dataset and annotations, configure the paths in `training/config/default.yaml`.

### Required Path Configuration

The default configuration already contains seven datasets. Set these environment
variables before launching (or replace their defaults in `default.yaml`):

- `DVGT_META_ROOT`, `DVGT_IMAGE_ROOT`, `DVGT_ALIGN_DEPTH_ROOT`,
  `DVGT_PROJ_DEPTH_ROOT`
- `MVS_SYNTH_ROOT`, `VKITTI_ROOT`
- `VGGT_CHECKPOINT`

### Configuration Example

```bash
export MVS_SYNTH_ROOT=/path/to/MVS-Synth/GTAV_1080_new
export VKITTI_ROOT=/path/to/vkitti
export VGGT_CHECKPOINT=/path/to/model.pt
```

## 3. Seven-dataset metric fine-tuning

Run from the `training/` directory. This example uses 4 GPUs with PyTorch
Distributed Data Parallel (DDP):

```bash
torchrun --nproc_per_node=4 launch.py
```

The configuration loads the official VGGT checkpoint and trains the complete
Aggregator and Camera/Depth/Point heads. DINOv2's unused `mask_token` remains
frozen by the model implementation; Track head/loss is disabled.

### Automatic resume after interruption

`checkpoint.auto_resume=True` is enabled by default. On restart, the trainer
first looks for `logs/${exp_name}/ckpts/checkpoint.pt` and restores model,
optimizer, AMP scaler, global train/validation steps, elapsed time and the next
epoch. Only when no training checkpoint exists does it load `VGGT_CHECKPOINT`
as the initialization weight. If saving was interrupted and the primary file
is corrupt, the loader automatically falls back to `checkpoint.pt.bak`.

Resume is epoch-level: a job interrupted in the middle of an epoch restarts
from the latest fully saved epoch. To intentionally start a new experiment,
use a new `exp_name`/`checkpoint.save_dir`, or launch with
`checkpoint.auto_resume=False` after confirming that overwriting the existing
experiment is intended.

Metric datasets are converted to metres, multiplied by the fixed DVGT factor
`0.1`, and supervised directly. MVS-Synth is marked non-metric and uses one
shared scale-invariant loss space for point/depth/camera translation. Inference
outputs from the metric branch are converted back to metres by multiplying by
`10`.

## 4. Multi-dataset sampling

The ratio of datasets is controlled by each loader's virtual `len_train`.

当前默认训练加载器使用 DVGT 风格的分组采样。它先按照各数据集虚拟
`len_train` 的比例选择一个数据集，再完全从该数据集中构造当前 batch，
因此同一个 batch 不会混入不同数据集。子数据集内部仍然启用
`inside_random`，所以增大虚拟长度不需要在内存或磁盘中复制场景元数据。

如果需要退回原始 VGGT 的拼接索引采样器，请删除
`data.train.sampler_config`，并设置
`data.train.common_config.grouped_sampling: False`。

## 5. Common Questions

### Memory Management

If you encounter out-of-memory (OOM) errors on your GPU, consider adjusting the following parameters in `training/config/default.yaml`:

- `max_img_per_gpu`: Reduce this value to decrease the batch size per GPU
- `accum_steps`: Sets the number of gradient accumulation steps (this configuration uses 1). Increasing it splits batches into smaller chunks, but `max_epochs` must be increased if you want to keep the same number of optimizer updates.

### Learning Rate Tuning

The main hyperparameter to be careful about is learning rate. Note that learning rate depends on the effective batch size, which is `batch_size_per_gpu × num_gpus`. Therefore, I highly recommend trying several learning rates based on your training setup. Generally, trying values like `5e-6`, `1e-5`, `5e-5`, `1e-4`, `5e-4` should be sufficient.

### Tracking Head

The first version intentionally disables Track Head because these seven loaders
do not provide track annotations. Camera, Depth and Point heads are enabled.

### Dataloader Validation

To check if your dataloader is working correctly, the best approach is to visualize its output. You can save the 3D world points as follows and then visually inspect the PLY files:

```python
def save_ply(points, colors, filename):
    import open3d as o3d                
    if torch.is_tensor(points):
        points_visual = points.reshape(-1, 3).cpu().numpy()
    else:
        points_visual = points.reshape(-1, 3)
    if torch.is_tensor(colors):
        points_visual_rgb = colors.reshape(-1, 3).cpu().numpy()
    else:
        points_visual_rgb = colors.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_visual.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(points_visual_rgb.astype(np.float64))
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)

# Usage example
save_ply(
    batch["world_points"][0].reshape(-1, 3), 
    batch["images"][0].permute(0, 2, 3, 1).reshape(-1, 3), 
    "debug.ply"
)
```

### Handling Unordered Sequences

For unordered sequences, you can check how we compute the ranking (similarity) between one frame and all other frames, as discussed in [Issue #82](https://github.com/facebookresearch/vggt/issues/82).

### Expected Coordinate System

Camera poses are expected to follow the OpenCV `camera-from-world` convention. Depth maps should be aligned with their corresponding camera poses.
