# Data format

All arrays are stored as MATLAB `.mat` files.

## Resistivity model

```text
rho/rho_<id>.mat
key: rho
shape: 256 x 1024
value: log10 resistivity
```

## Traditional inversion inputs

```text
inv_input_wa/wainv_<id>.mat
key: WA

inv_input_wb/wbinv_<id>.mat
key: WB

inv_input_slm/slminv_<id>.mat
key: SLM
```

Each array has shape `256 x 1024` and stores log10 resistivity values.

## Forward apparent-resistivity responses

```text
wa_256_layered/rhoa_2d_<id>.mat
wb_256_layered/rhoa_2d_<id>.mat
slm_256_layered/rhoa_2d_<id>.mat
```

Required keys:

```text
img   Apparent-resistivity response image.
mask  Valid-region mask.
```

The valid response region is trapezoidal or triangular depending on the
arrangement and acquisition protocol.

