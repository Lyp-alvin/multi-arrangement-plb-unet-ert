from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = _group_count(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.fusion = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.projection(x)
        return self.fusion(torch.cat([x, skip], dim=1))


class PlainUNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        channels: Sequence[int] = (24, 48, 96, 192, 384),
    ):
        super().__init__()
        channels = tuple(int(value) for value in channels)
        if len(channels) != 5:
            raise ValueError("channels must contain five pyramid widths")
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.required_stride = 16

        self.stem = ConvBlock(self.input_channels, channels[0])
        self.downsamples = nn.ModuleList(
            [DownBlock(channels[index], channels[index + 1]) for index in range(4)]
        )
        self.encoder_blocks = nn.ModuleList(
            [ConvBlock(channels[index], channels[index]) for index in range(1, 4)]
        )
        self.bottleneck = ConvBlock(channels[4], channels[4])
        self.decoder_blocks = nn.ModuleList(
            [
                UpBlock(channels[4], channels[3], channels[3]),
                UpBlock(channels[3], channels[2], channels[2]),
                UpBlock(channels[2], channels[1], channels[1]),
                UpBlock(channels[1], channels[0], channels[0]),
            ]
        )
        self.output_head = nn.Conv2d(channels[0], self.output_channels, 1)
        self.apply(self._initialize_weights)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def initialize_output_bias(self, value: float) -> None:
        with torch.no_grad():
            self.output_head.bias.fill_(float(value))

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        height, width = x.shape[-2:]
        pad_height = (-height) % self.required_stride
        pad_width = (-width) % self.required_stride
        if pad_height or pad_width:
            x = F.pad(x, (0, pad_width, 0, pad_height), mode="replicate")
        return x, (height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(f"Expected NCHW with {self.input_channels} channels, got {tuple(x.shape)}")
        x, original_size = self._pad(x)
        skips = [self.stem(x)]
        x = skips[0]
        for index, downsample in enumerate(self.downsamples):
            x = downsample(x)
            if index < len(self.encoder_blocks):
                x = self.encoder_blocks[index](x)
                skips.append(x)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoder_blocks, reversed(skips)):
            x = decoder(x, skip)
        output = self.output_head(x)
        return output[..., : original_size[0], : original_size[1]]


def build_plain_unet(input_channels: int = 1, output_channels: int = 1) -> PlainUNet:
    return PlainUNet(input_channels=input_channels, output_channels=output_channels)

