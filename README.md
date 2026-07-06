<div align="center">

# VGGT-World: Transforming VGGT into an Autoregressive Geometry World Model

### Xiangyu Sun, Shijie Wang, Fengyi Zhang, Lin Liu, Caiyan Jia, Ziying Song, Zi Huang, Yadan Luo

[![arXiv](https://img.shields.io/badge/arXiv-2603.12655-b31b1b)](https://arxiv.org/abs/2603.12655)

<img src="assets/pipeline.png" width="95%" alt="VGGT-World main figure">

</div>

This repository contains the training, evaluation, and demo code for **VGGT-World**, built on top of [VGGT](https://github.com/facebookresearch/vggt). Only the flow-matching (`fm`) module is trained; the VGGT aggregator and geometry heads are frozen by default.

## Checkpoints

Download our pretrained checkpoints from OneDrive:

**[VGGT-World Checkpoints (OneDrive)](https://1drv.ms/f/c/991ebd14386d50f5/IgAJa17DGykcT6LEAhV89viUAYbdSNHZm-woBHrWxHWZYbk?e=XLHhcc)**

After downloading, place the `.pt` file locally, e.g. `checkpoints/kitti_checkpoint.pt`. Checkpoints are PyTorch dicts with a `model` key (online weights).

| Config | Use case |
|--------|----------|
| `default_kitti.yaml` | KITTI short-sequence (stage 1) |
| `default_cityscapes.yaml` | Cityscapes short-sequence (stage 1) |
| `default_cityscapes_stage2.yaml` | Cityscapes mid-sequence (stage 2, autoregressive roll-out) |

## Installation

**Requirements:** Linux, Python 3.10+, CUDA GPU (tested with CUDA 12.8).

```bash
git clone https://github.com/SimonSun0810/VGGT-World.git
cd VGGT-World

conda create -n vggt_world python=3.10 -y
conda activate vggt_world

pip install -r requirements.txt
```

For other CUDA versions, install `torch` / `torchvision` from [pytorch.org](https://pytorch.org) first, then install the remaining packages.

Training is launched from `training_fm/` (Hydra resolves `data.*` imports relative to that directory):

```bash
cd training_fm
```

## Data Preparation

### KITTI

Expected layout (root = `kitti_DIR` in config):

```text
kitti_DIR/
├── train/
│   └── 2011_09_26/2011_09_26_drive_XXXX_sync/image_02/data/*.png
├── val/
│   └── 2011_09_26/2011_09_26_drive_XXXX_sync/image_02/data/*.png
└── val_depth/          # required for eval only
    └── 2011_09_26/2011_09_26_drive_XXXX_sync/proj_depth/groundtruth/image_02/*.png
```

Set the path in config or via CLI override, e.g. `data.train.dataset.dataset_configs.0.kitti_DIR=/path/to/kitti`.

### Cityscapes

Use **leftImg8bit_sequence** (video sequences), not single-frame `leftImg8bit`:

```text
Cityscapes_DIR/
├── train/
│   └── <city>/<city>_<seq>_*_leftImg8bit.png
└── val/
    └── <city>/<city>_<seq>_*_leftImg8bit.png
```

Set `Cityscapes_DIR` in `default_cityscapes.yaml` or override at launch.

## Training

From `training_fm/`:

**KITTI (stage 1, 4-frame clips):**

```bash
python launch.py --config default_kitti \
  data.train.dataset.dataset_configs.0.kitti_DIR=/path/to/kitti \
  data.val.dataset.dataset_configs.0.kitti_DIR=/path/to/kitti \
  checkpoint.resume_checkpoint_path=/path/to/vggt_or_fm_init.pt
```

**Cityscapes (stage 1):**

```bash
python launch.py --config default_cityscapes \
  data.train.dataset.dataset_configs.0.Cityscapes_DIR=/path/to/cityscapes \
  data.val.dataset.dataset_configs.0.Cityscapes_DIR=/path/to/cityscapes \
  checkpoint.resume_checkpoint_path=/path/to/init.pt
```

**Cityscapes (stage 2, 5-frame autoregressive):**

```bash
python launch.py --config default_cityscapes_stage2 \
  data.train.dataset.dataset_configs.0.Cityscapes_DIR=/path/to/cityscapes \
  checkpoint.resume_checkpoint_path=/path/to/stage1.pt
```

Logs and checkpoints are written under `training_fm/logs/<exp_name>/`. TensorBoard and W&B can be toggled in the yaml (`logging.wandb.enabled`).

Multi-GPU (example, 4 GPUs):

```bash
torchrun --nproc_per_node=4 launch.py --config default_kitti ...
```

## Evaluation

Run from the **repository root**. Eval scripts load Hydra config name `default`; create a symlink to match your checkpoint:

```bash
# KITTI
ln -sf default_kitti.yaml training_fm/config/default.yaml

python eval/kitti_val_short.py \
  --kitti_root /path/to/kitti \
  --ckpt /path/to/checkpoint.pt
```

```bash
# Cityscapes short
ln -sf default_cityscapes.yaml training_fm/config/default.yaml

python eval/cityscapes_val_short.py \
  --cityscapes_dir /path/to/leftImg8bit_sequence \
  --ckpt /path/to/checkpoint.pt
```

```bash
# Cityscapes mid (stage 2)
ln -sf default_cityscapes_stage2.yaml training_fm/config/default.yaml

python eval/cityscapes_val_mid.py \
  --cityscapes_dir /path/to/leftImg8bit_sequence \
  --ckpt /path/to/checkpoint.pt
```

| Script | Setting |
|--------|---------|
| `eval/kitti_val_short.py` | KITTI, 4 frames, stage 1 |
| `eval/kitti_val_mid.py` | KITTI, 5 frames, stage 2 roll-out |
| `eval/cityscapes_val_short.py` | Cityscapes, 4 frames |
| `eval/cityscapes_val_mid.py` | Cityscapes, 6 frames, stage 2 |

Metrics printed: `abs_rel`, `delta1` (KITTI also reports pred-vs-VGGT and VGGT-vs-GT).

## Demo

A minimal KITTI demo is included with preprocessed sample frames under `demo/frames/` (224×448, same crop as eval).

```bash
# from repo root
python demo/kitti_demo.py \
  --ckpt /path/to/kitti_checkpoint.pt \
  --steps 50
```

**Inputs:** `frame1`, `frame2` (conditioning); `frame3`, `frame4` (context for depth decode). Defaults use bundled frames from the first KITTI val sequence.

**Outputs** (color depth maps in `demo/outputs/`):

- `pred_depth_frame3.png`, `pred_depth_frame4.png` — FM prediction
- `vggt_depth_frame3.png`, `vggt_depth_frame4.png` — VGGT baseline on the same frames

Bundled `demo/frames/gt_depth_*.png` are KITTI projected GT depth visualizations for reference.

## Project Layout

```text
training_fm/     # Training (launch.py, configs, datasets, trainer)
eval/            # KITTI & Cityscapes evaluation scripts
demo/            # KITTI inference demo + sample frames
vggt/            # VGGT backbone + FM module (flux blocks)
assets/          # Figures for README
```

## License

This project is based on [VGGT](https://github.com/facebookresearch/vggt). Please retain `LICENSE.txt` from the upstream VGGT repository and comply with its terms for the VGGT-derived portions. New code added in this repository is subject to the same license unless otherwise noted.

## Citation

If you find this work useful, please cite:

```bibtex
@article{sun2026vggtworld,
  title={VGGT-World: Transforming VGGT into an Autoregressive Geometry World Model},
  author={Sun, Xiangyu and Wang, Shijie and Zhang, Fengyi and Liu, Lin and Jia, Caiyan and Song, Ziying and Huang, Zi and Luo, Yadan},
  journal={arXiv preprint arXiv:2603.12655},
  year={2026}
}
```
