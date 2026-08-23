# -*- coding: utf-8 -*-
"""
Full liquid-neural U-Net for dense prediction.

The U-shaped topology, down/up-sampling operations, skip connections, and
prediction head remain conventional. Every feature-processing stage in the
encoder, bottleneck, and decoder is made from ParallelLiquidBlock2d blocks.

The liquid update is a stable, parallel, closed-form continuous-depth update:

    z       = spatial_mix(norm(x))
    rate    = softplus(f(z))
    gate    = sigmoid(-rate * dt + bias(z))
    target  = gate * tanh(g(z)) + (1 - gate) * tanh(h(z))
    alpha   = exp(-dt / tau)
    state   = alpha * z + (1 - alpha) * target
    output  = x + gamma * state

Unlike a recurrent LTC implementation that scans H*W pixels sequentially,
this formulation evaluates all pixels in parallel and is practical for
256x1024 images. It is a research implementation inspired by LTC/CfC visual
dynamics, not a drop-in reproduction of the sequence-oriented ncps.CfC cell.

The model returns raw outputs:
  * regression: use MSE/L1 directly;
  * binary segmentation: use BCEWithLogitsLoss;
  * multi-class segmentation: use CrossEntropyLoss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_group_count(channels: int, preferred: int = 8) -> int:
    """Largest useful GroupNorm group count that divides channels."""
    for groups in range(min(int(preferred), int(channels)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, probability: float = 0.0):
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("drop-path probability must be in [0, 1).")
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep_probability = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_probability)
        return x * mask / keep_probability


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ):
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                int(in_channels),
                int(out_channels),
                kernel_size=int(kernel_size),
                stride=int(stride),
                padding=int(padding),
                groups=int(groups),
                bias=False,
            ),
            nn.GroupNorm(
                _valid_group_count(int(out_channels)),
                int(out_channels),
            ),
        ]
        if activation:
            layers.append(nn.GELU())
        super().__init__(*layers)


class ParallelLiquidBlock2d(nn.Module):
    """
    Spatially parallel closed-form liquid feature block.

    ``dt`` and ``tau`` are positive learned quantities. The input-dependent
    ``rate`` controls how quickly each pixel/channel transitions between two
    candidate states. All operations preserve NCHW shape.
    """

    def __init__(
        self,
        channels: int,
        expansion: float = 2.0,
        spatial_kernel: int = 5,
        drop_path: float = 0.0,
        layer_scale_init: float = 0.1,
        minimum_tau: float = 1e-3,
    ):
        super().__init__()
        channels = int(channels)
        hidden_channels = max(channels, int(round(channels * expansion)))
        if spatial_kernel % 2 == 0:
            raise ValueError("spatial_kernel must be odd.")

        self.channels = channels
        self.minimum_tau = float(minimum_tau)

        self.pre_norm = nn.GroupNorm(
            _valid_group_count(channels),
            channels,
        )
        self.spatial_mix = nn.Conv2d(
            channels,
            channels,
            kernel_size=int(spatial_kernel),
            padding=int(spatial_kernel) // 2,
            groups=channels,
            bias=False,
        )
        self.backbone = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GELU(),
        )
        # One projection is more efficient than three independent convolutions.
        self.liquid_projection = nn.Conv2d(
            hidden_channels,
            4 * channels,
            kernel_size=1,
            bias=True,
        )

        # softplus(0) ~= 0.693: a sensible initial unit step and time constant.
        self.raw_dt = nn.Parameter(torch.zeros(1))
        self.raw_tau = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.liquid_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), float(layer_scale_init))
        )
        self.drop_path = DropPath(drop_path)

        self.ffn_norm = nn.GroupNorm(
            _valid_group_count(channels),
            channels,
        )
        self.ffn_expand = nn.Conv2d(
            channels,
            hidden_channels,
            kernel_size=1,
            bias=False,
        )
        self.ffn_depthwise = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            bias=False,
        )
        self.ffn_project = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.ffn_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), float(layer_scale_init))
        )

    @property
    def dt(self) -> torch.Tensor:
        return F.softplus(self.raw_dt) + 1e-4

    @property
    def tau(self) -> torch.Tensor:
        return F.softplus(self.raw_tau) + self.minimum_tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.spatial_mix(self.pre_norm(x))
        liquid_parameters = self.liquid_projection(self.backbone(z))
        raw_rate, raw_bias, candidate_a, candidate_b = torch.chunk(
            liquid_parameters,
            chunks=4,
            dim=1,
        )

        rate = F.softplus(raw_rate) + 1e-4
        gate = torch.sigmoid(-rate * self.dt + raw_bias)
        target = (
            gate * torch.tanh(candidate_a)
            + (1.0 - gate) * torch.tanh(candidate_b)
        )

        alpha = torch.exp(-self.dt / self.tau)
        liquid_state = alpha * z + (1.0 - alpha) * target
        x = x + self.drop_path(self.liquid_scale * liquid_state)

        ffn = self.ffn_expand(self.ffn_norm(x))
        ffn = F.gelu(ffn)
        ffn = self.ffn_depthwise(ffn)
        ffn = F.gelu(ffn)
        ffn = self.ffn_project(ffn)
        return x + self.drop_path(self.ffn_scale * ffn)


class LiquidStage2d(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int,
        expansion: float,
        spatial_kernel: int,
        drop_paths: Sequence[float],
    ):
        super().__init__()
        if len(drop_paths) != int(depth):
            raise ValueError("drop_paths length must equal stage depth.")
        self.blocks = nn.Sequential(
            *[
                ParallelLiquidBlock2d(
                    channels=channels,
                    expansion=expansion,
                    spatial_kernel=spatial_kernel,
                    drop_path=drop_paths[index],
                )
                for index in range(int(depth))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class Downsample2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = ConvNormAct(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class LiquidUpBlock2d(nn.Module):
    """Upsample, fuse a skip connection, then process with liquid blocks."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        depth: int,
        expansion: float,
        spatial_kernel: int,
        drop_paths: Sequence[float],
    ):
        super().__init__()
        self.up_projection = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=1,
            bias=False,
        )
        self.skip_gate = nn.Sequential(
            nn.Conv2d(
                2 * int(out_channels),
                int(out_channels),
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )
        self.merge = ConvNormAct(
            int(out_channels) + int(skip_channels),
            int(out_channels),
            kernel_size=1,
        )
        self.liquid_stage = LiquidStage2d(
            channels=out_channels,
            depth=depth,
            expansion=expansion,
            spatial_kernel=spatial_kernel,
            drop_paths=drop_paths,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = self.up_projection(x)

        # The gate is derived from decoder and skip content. If skip channels
        # differ, the merge convolution still handles concatenation below.
        if skip.shape[1] == x.shape[1]:
            gate = self.skip_gate(torch.cat([x, skip], dim=1))
            skip = skip * gate

        x = self.merge(torch.cat([x, skip], dim=1))
        return self.liquid_stage(x)


@dataclass(frozen=True)
class LNNUNetConfig:
    input_channels: int = 3
    output_channels: int = 1
    channels: tuple[int, ...] = (24, 48, 96, 192, 384)
    encoder_depths: tuple[int, ...] = (1, 1, 1, 2)
    bottleneck_depth: int = 2
    decoder_depths: tuple[int, ...] = (1, 1, 1, 1)
    expansion: float = 2.0
    spatial_kernel: int = 5
    drop_path_rate: float = 0.05


class LNNUNet(nn.Module):
    """
    U-Net whose encoder, bottleneck, and decoder feature blocks are liquid.

    Input and output spatial sizes are identical. Internally, inputs are padded
    to a multiple of 16 and cropped back, so non-divisible shapes are supported.
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 1,
        channels: Sequence[int] = (24, 48, 96, 192, 384),
        encoder_depths: Sequence[int] = (1, 1, 1, 2),
        bottleneck_depth: int = 2,
        decoder_depths: Sequence[int] = (1, 1, 1, 1),
        expansion: float = 2.0,
        spatial_kernel: int = 5,
        drop_path_rate: float = 0.05,
    ):
        super().__init__()
        channels = tuple(int(value) for value in channels)
        encoder_depths = tuple(int(value) for value in encoder_depths)
        decoder_depths = tuple(int(value) for value in decoder_depths)

        if len(channels) != 5:
            raise ValueError("channels must contain five pyramid widths.")
        if len(encoder_depths) != 4:
            raise ValueError("encoder_depths must contain four values.")
        if len(decoder_depths) != 4:
            raise ValueError("decoder_depths must contain four values.")
        if any(value <= 0 for value in channels):
            raise ValueError("all channel widths must be positive.")
        if any(value <= 0 for value in encoder_depths + decoder_depths):
            raise ValueError("all stage depths must be positive.")
        if int(bottleneck_depth) <= 0:
            raise ValueError("bottleneck_depth must be positive.")

        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.channels = channels
        self.required_stride = 2 ** 4

        total_blocks = (
            sum(encoder_depths)
            + int(bottleneck_depth)
            + sum(decoder_depths)
        )
        drop_path_values = torch.linspace(
            0.0,
            float(drop_path_rate),
            total_blocks,
        ).tolist()
        cursor = 0

        self.stem = ConvNormAct(
            self.input_channels,
            channels[0],
            kernel_size=3,
        )

        encoder_stages: list[nn.Module] = []
        downsamples: list[nn.Module] = []
        for index, depth in enumerate(encoder_depths):
            stage_drop_paths = drop_path_values[cursor : cursor + depth]
            cursor += depth
            encoder_stages.append(
                LiquidStage2d(
                    channels=channels[index],
                    depth=depth,
                    expansion=expansion,
                    spatial_kernel=spatial_kernel,
                    drop_paths=stage_drop_paths,
                )
            )
            downsamples.append(
                Downsample2d(channels[index], channels[index + 1])
            )
        self.encoder_stages = nn.ModuleList(encoder_stages)
        self.downsamples = nn.ModuleList(downsamples)

        bottleneck_drop_paths = drop_path_values[
            cursor : cursor + int(bottleneck_depth)
        ]
        cursor += int(bottleneck_depth)
        self.bottleneck = LiquidStage2d(
            channels=channels[-1],
            depth=int(bottleneck_depth),
            expansion=expansion,
            spatial_kernel=spatial_kernel,
            drop_paths=bottleneck_drop_paths,
        )

        decoder_blocks: list[nn.Module] = []
        for decoder_index, depth in enumerate(decoder_depths):
            in_index = 4 - decoder_index
            skip_index = in_index - 1
            stage_drop_paths = drop_path_values[cursor : cursor + depth]
            cursor += depth
            decoder_blocks.append(
                LiquidUpBlock2d(
                    in_channels=channels[in_index],
                    skip_channels=channels[skip_index],
                    out_channels=channels[skip_index],
                    depth=depth,
                    expansion=expansion,
                    spatial_kernel=spatial_kernel,
                    drop_paths=stage_drop_paths,
                )
            )
        self.decoder_blocks = nn.ModuleList(decoder_blocks)

        self.output_head = nn.Sequential(
            nn.GroupNorm(
                _valid_group_count(channels[0]),
                channels[0],
            ),
            nn.GELU(),
            nn.Conv2d(
                channels[0],
                self.output_channels,
                kernel_size=1,
                bias=True,
            ),
        )

        self.apply(self._initialize_weights)
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            fan_out = (
                module.kernel_size[0]
                * module.kernel_size[1]
                * module.out_channels
            )
            if module.groups > 1:
                fan_out //= module.groups
            nn.init.normal_(module.weight, 0.0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def initialize_output_bias(self, value: float) -> None:
        """Useful for absolute-valued regression targets."""
        with torch.no_grad():
            self.output_head[-1].bias.fill_(float(value))

    def _pad_input(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        height, width = x.shape[-2:]
        pad_height = (-height) % self.required_stride
        pad_width = (-width) % self.required_stride
        if pad_height == 0 and pad_width == 0:
            return x, (height, width)
        x = F.pad(x, (0, pad_width, 0, pad_height), mode="replicate")
        return x, (height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected NCHW input, received shape {tuple(x.shape)}."
            )
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, "
                f"received {x.shape[1]}."
            )

        x, original_size = self._pad_input(x)
        x = self.stem(x)

        skips: list[torch.Tensor] = []
        for stage, downsample in zip(
            self.encoder_stages,
            self.downsamples,
        ):
            x = stage(x)
            skips.append(x)
            x = downsample(x)

        x = self.bottleneck(x)
        for decoder, skip in zip(
            self.decoder_blocks,
            reversed(skips),
        ):
            x = decoder(x, skip)

        output = self.output_head(x)
        return output[..., : original_size[0], : original_size[1]]

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method for binary or multi-class segmentation."""
        logits = self.forward(x)
        if self.output_channels == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)


def lnn_unet_tiny(
    input_channels: int = 3,
    output_channels: int = 1,
) -> LNNUNet:
    return LNNUNet(
        input_channels=input_channels,
        output_channels=output_channels,
        channels=(16, 32, 64, 128, 256),
        encoder_depths=(1, 1, 1, 1),
        bottleneck_depth=1,
        decoder_depths=(1, 1, 1, 1),
        expansion=1.5,
        drop_path_rate=0.03,
    )


def lnn_unet_base(
    input_channels: int = 3,
    output_channels: int = 1,
) -> LNNUNet:
    return LNNUNet(
        input_channels=input_channels,
        output_channels=output_channels,
        channels=(24, 48, 96, 192, 384),
        encoder_depths=(1, 1, 1, 2),
        bottleneck_depth=2,
        decoder_depths=(1, 1, 1, 1),
        expansion=2.0,
        drop_path_rate=0.05,
    )


def lnn_unet_large(
    input_channels: int = 3,
    output_channels: int = 1,
) -> LNNUNet:
    return LNNUNet(
        input_channels=input_channels,
        output_channels=output_channels,
        channels=(32, 64, 128, 256, 512),
        encoder_depths=(2, 2, 2, 3),
        bottleneck_depth=3,
        decoder_depths=(2, 2, 2, 2),
        expansion=2.0,
        drop_path_rate=0.10,
    )


def build_lnn_unet(
    variant: str = "base",
    input_channels: int = 3,
    output_channels: int = 1,
) -> LNNUNet:
    builders = {
        "tiny": lnn_unet_tiny,
        "base": lnn_unet_base,
        "large": lnn_unet_large,
    }
    try:
        builder = builders[str(variant).lower()]
    except KeyError as error:
        raise ValueError(
            f"Unknown variant {variant!r}; choose from {tuple(builders)}."
        ) from error
    return builder(
        input_channels=input_channels,
        output_channels=output_channels,
    )


if __name__ == "__main__":
    model = lnn_unet_tiny(input_channels=3, output_channels=1)
    sample = torch.randn(2, 3, 65, 129)
    prediction = model(sample)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(model.__class__.__name__)
    print(f"parameters: {parameter_count:,}")
    print(f"input:  {tuple(sample.shape)}")
    print(f"output: {tuple(prediction.shape)}")
