import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from diffusers.models.activations import get_activation, FP32SiLU
from vggt.models.flux_modules.modeling_flux_block import FluxTransformerBlock, FluxSingleTransformerBlock
from vggt.models.flux_modules.modeling_normalization import AdaLayerNormContinuous

def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    dim = int(dim) 
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)

    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(2)

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb

class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample

class TimestepEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep):
        timesteps_proj = self.time_proj(timestep)
        timesteps_proj = timesteps_proj.to(self.timestep_embedder.linear_1.weight.dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)

        return timesteps_emb

# 本次修改：多目槽位共享物理时间轴，并用分数偏移区分同一时刻的左/右目。
def build_ids(B, T, H, W, device, views_per_timestep: int = 1):
    if views_per_timestep <= 0:
        raise ValueError(
            f"views_per_timestep must be positive, got {views_per_timestep}."
        )
    # V=2 maps time-major slots to 0,0.5,1,1.5,... .  V=1 exactly
    # preserves the pretrained monocular behavior 0,1,2,3,... .
    slot_positions = torch.arange(T, device=device, dtype=torch.float32)
    slot_positions = slot_positions / float(views_per_timestep)
    t = slot_positions[:, None, None].expand(T, H, W)  # [T,H,W]
    y = torch.arange(H, device=device, dtype=slot_positions.dtype)
    y = y[None, :, None].expand(T, H, W)  # [T,H,W]
    x = torch.arange(W, device=device, dtype=slot_positions.dtype)
    x = x[None, None, :].expand(T, H, W)  # [T,H,W]

    ids = torch.stack([t, y, x], dim=-1)   # [T,H,W,3]
    ids = ids.reshape(1, T*H*W, 3).repeat(B, 1, 1)  # [B,N,3]
    return ids

# -------------------------
# Config
# -------------------------
@dataclass
class FMConfig:
    in_dim: int = 1024        
    model_dim: int = 512    
    depth: int = 8 
    n_heads: int = 8
    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    # token counts
    n_max: int = 30000
    use_patch_pos: bool = True
    t_frames: int = 2
    # 本次修改：记录每个物理时间步的相机数，仅影响 RoPE 位置 ID，不增加可学参数。
    views_per_timestep: int = 1


# -------------------------
# Flow Matching
# -------------------------
class Flowmatching(nn.Module):
    """
    Flow matching over an equal number of condition and target image slots.

    Inputs:
      x_t_layers:  list len=1, each [B, T, N, 1024] (noisy target state)
      cond_layers: list len=1, each [B, T, N, 1024] (observed condition)
      t: [B] or [B,1] in [0,1]

    Output:
      clean target tokens: [B, T*N, 1024]
    """
    def __init__(self, cfg: FMConfig):
        super().__init__()

        assert cfg.model_dim % cfg.n_heads == 0, "model_dim must be divisible by n_heads."
        self.cfg = cfg

        self.max_steps = 1000
        timesteps = np.linspace(self.max_steps, 0, self.max_steps + 1,) # [1000,999,...,1,0]
        self.timesteps_per_stage = torch.from_numpy(timesteps[:-1]).to("cuda").float() # [1000,999,...,1]
        self.stage_sigmas = np.linspace(1, 0, self.max_steps + 1,) # [1,0.999,...,0.001,0]
        self.sigmas_per_stage = torch.from_numpy(self.stage_sigmas[:-1]).to("cuda").float() # [1,0.999,...,0.001]

        self.head_dim = self.cfg.model_dim // self.cfg.n_heads
        self.pos_embed = EmbedND(dim=self.cfg.model_dim, theta=10000, axes_dim = [self.head_dim//4, self.head_dim*3//8, self.head_dim*3//8])
        self.time_embed = TimestepEmbeddings(embedding_dim=cfg.model_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.cfg.model_dim,
                    num_attention_heads=self.cfg.n_heads,
                    attention_head_dim=self.cfg.model_dim // self.cfg.n_heads,
                    use_flash_attn=False,
                )
                for i in range(self.cfg.depth)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.cfg.model_dim,
                    num_attention_heads=self.cfg.n_heads,
                    attention_head_dim=self.cfg.model_dim // self.cfg.n_heads,
                    use_flash_attn=False,
                )
                for i in range(self.cfg.depth)
            ]
        )    
        self.norm_out = AdaLayerNormContinuous(self.cfg.model_dim, self.cfg.model_dim, elementwise_affine=False, eps=1e-6)    
        self.proj_out = nn.Linear(self.cfg.model_dim, self.cfg.model_dim, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize all the conditioning to normal init
        nn.init.normal_(self.time_embed.timestep_embedder.linear_1.weight, std=0.02)
        nn.init.normal_(self.time_embed.timestep_embedder.linear_2.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.norm1.linear.weight, 0)
            nn.init.constant_(block.norm1.linear.bias, 0)
            nn.init.constant_(block.norm1_context.linear.weight, 0)
            nn.init.constant_(block.norm1_context.linear.bias, 0)

        for block in self.single_transformer_blocks:
            nn.init.constant_(block.norm.linear.weight, 0)
            nn.init.constant_(block.norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)        

    # 本次修改：修正返回类型标注，FM forward 返回单个展平的 clean-latent 张量而非列表。
    def forward(
        self,
        x_t_layers: List[torch.Tensor],
        cond_layers: List[torch.Tensor],
        t: torch.Tensor,
        patch_hw: Optional[tuple] = None,
        views_per_timestep: Optional[int] = None,
    ) -> torch.Tensor:
        assert len(x_t_layers) == 1 and len(cond_layers) == 1, "Need 1 x layers and 1 cond layers."

        x = torch.cat(x_t_layers, dim=2)
        B, T, Ntot, C = x.shape
        assert C == self.cfg.model_dim
        cond = torch.cat(cond_layers, dim=2)

        # 本次修改：对 4 图历史 -> 4 图未来的等长约束给出显式报错，避免在 view 时才出现难定位的 shape 异常。
        if cond.shape != x.shape:
            raise ValueError(
                "FM condition and target must have identical [B,T,N,C] shapes, "
                f"got condition={tuple(cond.shape)} and target={tuple(x.shape)}."
            )

        hidden_states = x.view(B, T*Ntot, C)
        hidden_length = [T*Ntot] 
        encoder_hidden_states = cond.view(B, T*Ntot, C)

        num_special = 5
        patch_per_frame = Ntot - num_special
        if patch_hw is None:
            H = math.isqrt(patch_per_frame)
            W = H
            if H * W != patch_per_frame:
                raise ValueError(
                    f"patch_per_frame={patch_per_frame} is not square; pass patch_hw=(H,W)"
                )
        else:
            H, W = patch_hw
            if H * W != patch_per_frame:
                raise ValueError(
                    f"patch_hw=({H},{W}) does not match patch_per_frame={patch_per_frame}"
                )
        
        # 本次修改：允许每个 source bucket 在运行时提供相机路数；仅改变无参数 RoPE ID，不改变 checkpoint。
        raw_views = (
            self.cfg.views_per_timestep
            if views_per_timestep is None
            else views_per_timestep
        )
        runtime_views = int(raw_views)
        if float(raw_views) != runtime_views or runtime_views <= 0:
            raise ValueError(
                "views_per_timestep must be a positive integer, got "
                f"{raw_views!r}."
            )

        # 本次修改：patch 和 special token 使用同一套多目时间位置，避免将多路相机误当成完整时间步。
        # patch ids: [B, T*H*W, 3] -> [B,T,H*W,3]
        patch_ids = build_ids(
            B,
            2*T,
            H,
            W,
            device=x.device,
            views_per_timestep=runtime_views,
        )
        patch_ids = patch_ids.view(B, 2*T, H * W, 3)

        # special ids: [B,T,5,3]
        special_ids = torch.zeros((B, 2*T, num_special, 3), device=x.device, dtype=patch_ids.dtype)
        t_ids = torch.arange(2*T, device=x.device, dtype=patch_ids.dtype)
        t_ids = t_ids / float(runtime_views)
        t_ids = t_ids.view(1, 2*T, 1)  # [1,2T,1]
        special_ids[..., 0] = t_ids.repeat(B, 1, 1)

        # [special + patch]
        frame_ids = torch.cat([special_ids, patch_ids], dim=2)  # [B,T,5+H*W,3]
        image_ids = frame_ids.view(B, 2*T * (num_special + H * W), 3)

        encoder_len = encoder_hidden_states.shape[1]
        input_ids = image_ids
        frame_patch_rope_embed = self.pos_embed(input_ids)

        t = t.to(dtype=hidden_states.dtype)
        temb = self.time_embed(t)

        L = T * Ntot
        total_len = L + encoder_len
        attention_mask = [torch.ones((B, 1, total_len, total_len), device=x.device, dtype=torch.bool)]

        for _, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,          # [B,L,C]
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=None,
                temb=temb,                      
                attention_mask=attention_mask,        # None
                hidden_length=hidden_length,
                image_rotary_emb=frame_patch_rope_embed,    # [B,L,1,hd//2,2,2]
            )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        concat_hidden_length = [hidden_states.shape[1]]

        concat_len = hidden_states.shape[1]
        single_attention_mask = [torch.ones((B, 1, concat_len, concat_len), device=hidden_states.device, dtype=torch.bool)]

        for _, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_attention_mask=None,
                temb=temb,
                attention_mask=single_attention_mask,
                hidden_length=concat_hidden_length,
                image_rotary_emb=frame_patch_rope_embed,
            )

        hidden_states = hidden_states[:, hidden_length[0]:, :]

        hidden_states = self.norm_out(hidden_states, temb, hidden_length=[hidden_states.shape[1]])
        hidden_states = self.proj_out(hidden_states)

        return hidden_states

    # 本次修改：FM loss 可选返回可微 clean future latent，供 part2 和几何 loss 继续监督。
    def loss_rectified_multilayer(
        self,
        x1_layers: List[torch.Tensor],
        cond_layers: List[torch.Tensor],
        patch_hw: Optional[tuple] = None,
        return_x1_pred: bool = False,
        views_per_timestep: Optional[int] = None,
    ):
        """
        Rectified flow (Flow Matching) loss in original token space (1024):
          x0 ~ N(0,I)
          t ~ U(0,1)
          x_t = (1-t)x0 + t x1
          v*  = x1 - x0
          minimize ||v_hat(x_t,t,cond) - v*||^2
        """
        assert len(x1_layers) == 1 and len(cond_layers) == 1, "Need 1 target layers and 1 cond layers."
        B,T,N,C = x1_layers[0].shape
        device = x1_layers[0].device
        dtype = x1_layers[0].dtype

        # 本次修改：用 reshape 展平全部未来帧，同时兼容非连续的 target 时间切片。
        # ``reshape`` also supports the non-contiguous temporal slice used for
        # detached future targets.
        x1_big = x1_layers[0].reshape(B, T*N, C)
        cond_big = cond_layers[0].reshape(B, T*N, C)

        x0_big = torch.randn_like(x1_big)

        t = torch.rand(B, device=device, dtype=dtype)
        t = (t * self.max_steps).long()
        t = t.clamp(0, self.max_steps-1)
        timestep = self.timesteps_per_stage[t]
        ratios = self.sigmas_per_stage[t]
        ratios = ratios.view(B, 1, 1)

        x_t_big = ratios * x0_big + (1 - ratios) * x1_big
        v_star_big = (x1_big - x_t_big) / ratios

        x_t_layers = [x_t_big.reshape(B, T, N, C)]
        cond_layers = [cond_big.reshape(B, T, N, C)]

        # 本次修改：保留 FM forward 的 clean latent 预测张量，不调用 no-grad Euler sampler。
        # ``forward`` predicts the clean future latent x_1.  Keeping this tensor
        # differentiable lets the existing VGGT part2/depth heads supervise the
        # FM prediction directly, without running the no-grad Euler sampler.
        # 本次修改：把本 batch 的相机路数继续传给 FM RoPE，支持相邻 batch 使用不同 V。
        x1_pred_big = self.forward(
            x_t_layers,
            cond_layers,
            timestep,
            patch_hw=patch_hw,
            views_per_timestep=views_per_timestep,
        )
        v_hat_big = (x1_pred_big - x_t_big) / ratios

        # 本次修改：直接对 [B,T*N,C] 计算 MSE，修复原实现只监督第一张未来帧的问题。
        # Compute the loss over *all* future frames/tokens.  The previous split
        # loop only consumed the first temporal chunk because this implementation
        # has one latent level containing T frames.
        loss = F.mse_loss(v_hat_big, v_star_big)

        # 本次修改：仅联合模式返回 clean latent，旧 FM-only 调用仍只返回标量 loss。
        if return_x1_pred:
            x1_pred = x1_pred_big.reshape(B, T, N, C)
            return loss, x1_pred
        return loss

    # 本次修改：Euler 推理接收运行时相机数，使验证采样与训练使用完全相同的多目 RoPE 布局。
    @torch.no_grad()
    def sample_euler(
        self,
        cond_layers: List[torch.Tensor],
        shape_like: torch.Tensor,
        steps: int = 20,
        patch_hw: Optional[tuple] = None,
        views_per_timestep: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        Euler sampling (whole-x):
        dx/dt = v_theta(x,t,cond),  t: 0 -> 1

        cond_layers: len=1, each [B,T,Ntot,1024]
        shape_like: [B,T,Ntot,1024]  (overall latent shape)
        returns: legacy per-slot list at t=1, len=T, each [B,1,Ntot,1024]
        """
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert len(cond_layers) == 1
            if steps < 1:
                raise ValueError(f"steps must be >= 1, got {steps}")

            B, T, Ntot, C = shape_like.shape
            device = shape_like.device
            dtype = shape_like.dtype

            Ns = [c.shape[2] for c in cond_layers]

            x_big = torch.randn(B, T, Ntot, C, device=device, dtype=dtype)
            idxs = torch.linspace(
                0,
                self.max_steps - 1,
                steps + 1,
                device=device,
                dtype=torch.float32,
            ).long()

            for k in range(steps):
                t_cur = torch.full((B,), idxs[k].item(), device=device, dtype=torch.long)
                t_next = torch.full((B,), idxs[k + 1].item(), device=device, dtype=torch.long)
                timestep = self.timesteps_per_stage[t_cur]
                sigma_cur = self.sigmas_per_stage[t_cur].view(B, 1, 1, 1).clamp_min(1e-4)
                sigma_next = self.sigmas_per_stage[t_next].view(B, 1, 1, 1)
                # 本次修改：验证/推理采样与训练使用同一个运行时多相机布局。
                x_hat = self.forward(
                    [x_big],
                    cond_layers,
                    timestep,
                    patch_hw=patch_hw,
                    views_per_timestep=views_per_timestep,
                )
                x_hat = x_hat.view(B, T, Ntot, C)
                step_ratio = ((sigma_cur - sigma_next) / (1-sigma_cur)).clamp(0.0, 1.0)
                x_big = x_big + step_ratio * (x_hat - x_big)

            x_layers_out = list(x_big.split(1, dim=1))
            return x_layers_out
