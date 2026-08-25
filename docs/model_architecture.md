# Model architecture

## LIB

The Liquid-Inspired Block is implemented by `ParallelLiquidBlock2d` in
`src/lnn_unet.py`.

The block uses GroupNorm, depthwise spatial mixing, a parameter generator, a
closed-form liquid update, and a residual FFN.

## Single-arrangement LIB-U-Net

The single-arrangement LIB-U-Net is implemented in `src/lnn_unet.py`.

## Multiarray inverse network

The proposed inverse model is implemented by `MultiInverseLNNUNet` in
`src/multi_lnn_unet.py`.

It uses:

- one encoder per arrangement,
- scale-wise feature fusion by concatenation and 1 x 1 convolution,
- one shared decoder.

## Multiarray forward network

The forward surrogate is implemented by `MultiForwardLNNUNet` in
`src/multi_lnn_unet.py`.

It uses:

- a shared encoder,
- one decoder head per arrangement.
