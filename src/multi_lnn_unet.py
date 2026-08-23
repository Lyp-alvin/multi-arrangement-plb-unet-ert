from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from lnn_unet import (
    ConvNormAct,
    Downsample2d,
    LiquidStage2d,
    LiquidUpBlock2d,
    _valid_group_count,
)


ARRANGEMENTS = ("wa", "wb", "slm")


def _initialize_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        if module.groups > 1:
            fan_out //= module.groups
        nn.init.normal_(module.weight, 0.0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class LiquidEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: Sequence[int],
        encoder_depths: Sequence[int],
        bottleneck_depth: int,
        expansion: float,
        spatial_kernel: int,
        drop_path_rate: float,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        self.channels = tuple(int(value) for value in channels)
        self.required_stride = 16
        total_depth = sum(encoder_depths) + int(bottleneck_depth)
        drop_paths = torch.linspace(0.0, float(drop_path_rate), total_depth).tolist()
        cursor = 0
        self.stem = ConvNormAct(self.input_channels, self.channels[0], kernel_size=3)
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index, depth in enumerate(encoder_depths):
            self.stages.append(
                LiquidStage2d(
                    channels=self.channels[index],
                    depth=int(depth),
                    expansion=expansion,
                    spatial_kernel=spatial_kernel,
                    drop_paths=drop_paths[cursor : cursor + int(depth)],
                )
            )
            cursor += int(depth)
            self.downsamples.append(Downsample2d(self.channels[index], self.channels[index + 1]))
        self.bottleneck = LiquidStage2d(
            channels=self.channels[-1],
            depth=int(bottleneck_depth),
            expansion=expansion,
            spatial_kernel=spatial_kernel,
            drop_paths=drop_paths[cursor : cursor + int(bottleneck_depth)],
        )

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        height, width = x.shape[-2:]
        pad_height = (-height) % self.required_stride
        pad_width = (-width) % self.required_stride
        if pad_height or pad_width:
            x = F.pad(x, (0, pad_width, 0, pad_height), mode="replicate")
        return x, (height, width)

    def forward(self, x: torch.Tensor):
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(f"Expected NCHW with {self.input_channels} channels, got {tuple(x.shape)}")
        x, original_size = self._pad(x)
        x = self.stem(x)
        skips: list[torch.Tensor] = []
        for stage, downsample in zip(self.stages, self.downsamples):
            x = stage(x)
            skips.append(x)
            x = downsample(x)
        return self.bottleneck(x), skips, original_size


class LiquidDecoder(nn.Module):
    def __init__(
        self,
        output_channels: int,
        channels: Sequence[int],
        decoder_depths: Sequence[int],
        expansion: float,
        spatial_kernel: int,
        drop_path_rate: float,
    ):
        super().__init__()
        channels = tuple(int(value) for value in channels)
        total_depth = sum(decoder_depths)
        drop_paths = torch.linspace(0.0, float(drop_path_rate), total_depth).tolist()
        cursor = 0
        blocks = []
        for decoder_index, depth in enumerate(decoder_depths):
            in_index = 4 - decoder_index
            skip_index = in_index - 1
            blocks.append(
                LiquidUpBlock2d(
                    in_channels=channels[in_index],
                    skip_channels=channels[skip_index],
                    out_channels=channels[skip_index],
                    depth=int(depth),
                    expansion=expansion,
                    spatial_kernel=spatial_kernel,
                    drop_paths=drop_paths[cursor : cursor + int(depth)],
                )
            )
            cursor += int(depth)
        self.blocks = nn.ModuleList(blocks)
        self.output_head = nn.Sequential(
            nn.GroupNorm(_valid_group_count(channels[0]), channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], int(output_channels), kernel_size=1, bias=True),
        )

    def initialize_output_bias(self, value: float) -> None:
        with torch.no_grad():
            self.output_head[-1].bias.fill_(float(value))

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: Sequence[torch.Tensor],
        original_size: tuple[int, int],
    ) -> torch.Tensor:
        x = bottleneck
        for block, skip in zip(self.blocks, reversed(skips)):
            x = block(x, skip)
        output = self.output_head(x)
        return output[..., : original_size[0], : original_size[1]]


class MultiForwardLNNUNet(nn.Module):
    """Shared liquid encoder with one decoder per electrode arrangement."""

    def __init__(
        self,
        arrangements: Sequence[str] = ARRANGEMENTS,
        channels: Sequence[int] = (24, 48, 96, 192, 384),
        encoder_depths: Sequence[int] = (1, 1, 1, 2),
        bottleneck_depth: int = 2,
        decoder_depths: Sequence[int] = (1, 1, 1, 1),
        expansion: float = 2.0,
        spatial_kernel: int = 5,
        drop_path_rate: float = 0.05,
    ):
        super().__init__()
        self.arrangements = tuple(arrangements)
        self.encoder = LiquidEncoder(
            1, channels, encoder_depths, bottleneck_depth,
            expansion, spatial_kernel, drop_path_rate,
        )
        self.decoders = nn.ModuleDict(
            {
                name: LiquidDecoder(
                    1, channels, decoder_depths, expansion, spatial_kernel, drop_path_rate
                )
                for name in self.arrangements
            }
        )
        self.apply(_initialize_weights)
        for decoder in self.decoders.values():
            nn.init.zeros_(decoder.output_head[-1].weight)
            nn.init.zeros_(decoder.output_head[-1].bias)

    def initialize_output_bias(self, value: float) -> None:
        for decoder in self.decoders.values():
            decoder.initialize_output_bias(value)

    def forward(self, rho: torch.Tensor) -> dict[str, torch.Tensor]:
        bottleneck, skips, original_size = self.encoder(rho)
        return {
            name: decoder(bottleneck, skips, original_size)
            for name, decoder in self.decoders.items()
        }


class MultiInverseLNNUNet(nn.Module):
    """Arrangement-specific liquid encoders with fused skips and one decoder."""

    def __init__(
        self,
        arrangements: Sequence[str] = ARRANGEMENTS,
        channels: Sequence[int] = (24, 48, 96, 192, 384),
        encoder_depths: Sequence[int] = (1, 1, 1, 2),
        bottleneck_depth: int = 2,
        decoder_depths: Sequence[int] = (1, 1, 1, 1),
        expansion: float = 2.0,
        spatial_kernel: int = 5,
        drop_path_rate: float = 0.05,
    ):
        super().__init__()
        self.arrangements = tuple(arrangements)
        channels = tuple(int(value) for value in channels)
        self.encoders = nn.ModuleDict(
            {
                name: LiquidEncoder(
                    1, channels, encoder_depths, bottleneck_depth,
                    expansion, spatial_kernel, drop_path_rate,
                )
                for name in self.arrangements
            }
        )
        arrangement_count = len(self.arrangements)
        self.skip_fusions = nn.ModuleList(
            [ConvNormAct(arrangement_count * width, width, kernel_size=1) for width in channels[:-1]]
        )
        self.bottleneck_fusion = ConvNormAct(
            arrangement_count * channels[-1], channels[-1], kernel_size=1
        )
        self.decoder = LiquidDecoder(
            1, channels, decoder_depths, expansion, spatial_kernel, drop_path_rate
        )
        self.apply(_initialize_weights)
        nn.init.zeros_(self.decoder.output_head[-1].weight)
        nn.init.zeros_(self.decoder.output_head[-1].bias)

    def initialize_output_bias(self, value: float) -> None:
        self.decoder.initialize_output_bias(value)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        missing = set(self.arrangements) - set(inputs)
        if missing:
            raise KeyError(f"Missing arrangement inputs: {sorted(missing)}")
        encoded = [self.encoders[name](inputs[name]) for name in self.arrangements]
        original_size = encoded[0][2]
        if any(item[2] != original_size for item in encoded[1:]):
            raise ValueError("All arrangement inputs must have the same spatial size")
        bottleneck = self.bottleneck_fusion(torch.cat([item[0] for item in encoded], dim=1))
        fused_skips = [
            fusion(torch.cat([item[1][level] for item in encoded], dim=1))
            for level, fusion in enumerate(self.skip_fusions)
        ]
        return self.decoder(bottleneck, fused_skips, original_size)


def build_multi_forward_lnn_unet() -> MultiForwardLNNUNet:
    return MultiForwardLNNUNet()


def build_multi_inverse_lnn_unet() -> MultiInverseLNNUNet:
    return MultiInverseLNNUNet()

