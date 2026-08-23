# Multi-arrangement PLB-U-Net for ERT inversion

This repository contains the reference implementation for a multi-arrangement
closed-loop deep-learning framework for two-dimensional electrical resistivity
tomography (ERT) inversion.

The proposed model is referred to as **MA-PLB-U-Net-CL**:

- **MA**: multi-arrangement inputs from Wenner-alpha (WA), Wenner-beta (WB),
  and Schlumberger (SLM) arrays.
- **PLB**: Parallel Liquid Block, a liquid-neural-network-inspired feature
  block for dense 2-D feature maps.
- **CL**: closed-loop training using a frozen forward surrogate network.

The code is intended to support the manuscript submitted to
*Computers & Geosciences* and follows the journal's software availability
requirements: source code, training workflows, evaluation scripts, dependency
information, and demo data are provided in a public repository.

## Repository contents

```text
src/
  plain_unet.py                         Plain U-Net baseline.
  lnn_unet.py                           PLB and PLB-U-Net implementation.
  multi_lnn_unet.py                     Multi-arrangement forward/inverse models.
  train_wa_unet.py                      SA-U-Net-OL and SA-U-Net-CL training.
  train_wa_closed_loop_lnn.py           SA-PLB-U-Net-CL training.
  train_multi_closed_loop_lnn.py        MA-PLB-U-Net-CL training.
  evaluate_new_experiment.py            Evaluation and prediction export.
  field_pygimli_forward.py              pyGIMLi-based field forward checks.

examples/demo_data/
  Six representative synthetic samples used for quick tests and figure-style
  visualization (IDs 4, 52, 126, 347, 434, and 487).

scripts/
  MATLAB plotting scripts used to generate representative manuscript figures.

results/
  Summary of the validation metrics reported in the manuscript.

checkpoints/
  Expected location of trained best checkpoints downloaded from GitHub Releases.
```

## Main experiments

The manuscript compares four inversion settings:

1. **SA-U-Net-OL**: single-arrangement U-Net open-loop inversion.
2. **SA-U-Net-CL**: single-arrangement U-Net closed-loop inversion.
3. **SA-PLB-U-Net-CL**: single-arrangement PLB-U-Net closed-loop inversion.
4. **MA-PLB-U-Net-CL**: multi-arrangement PLB-U-Net closed-loop inversion
   (proposed method).

For closed-loop inversion, the training objective is

```text
L_total = 0.8 * L_inv + 0.2 * L_fwd
```

where `L_inv` is the model-domain MSE and `L_fwd` is the apparent-resistivity
response consistency loss computed by a frozen forward surrogate network.

## Parallel Liquid Block

The core PLB implementation is in:

```text
src/lnn_unet.py
```

The relevant class is:

```python
ParallelLiquidBlock2d
```

The block performs:

```text
Z = DepthwiseConv(GroupNorm(X))
[r_raw, beta, a, b] = ParameterGenerator(Z)
r = softplus(r_raw)
g = sigmoid(-r * dt + beta)
target = g * tanh(a) + (1 - g) * tanh(b)
alpha = exp(-dt / tau)
h = alpha * Z + (1 - alpha) * target
X' = X + gamma_liq * h
Y = X' + gamma_ffn * FFN(X')
```

All updates are evaluated in parallel over the 2-D feature map; this is not a
sequential recurrent scan over pixels.

## Installation

The recommended environment is Python 3.10 or 3.11 with PyTorch and pyGIMLi.
An Anaconda environment can be created with:

```bash
conda env create -f environment.yml
conda activate ma-plb-unet-ert
```

Alternatively:

```bash
pip install -r requirements.txt
```

MATLAB R2023b or later is recommended for the figure-generation scripts in
`scripts/`.

## Demo data

The repository includes six representative synthetic samples:

```text
4, 52, 126, 347, 434, 487
```

These samples are intended for quick testing, data-format inspection, and
figure-style visualization only. They are not sufficient to reproduce the full
quantitative validation metrics in the manuscript, which were computed on 400
validation samples from a 2000-sample synthetic dataset.

The demo data follow the same folder and key structure as the full dataset:

```text
rho/rho_<id>.mat                         key: rho
inv_input_wa/wainv_<id>.mat              key: WA
inv_input_wb/wbinv_<id>.mat              key: WB
inv_input_slm/slminv_<id>.mat            key: SLM
wa_256_layered/rhoa_2d_<id>.mat          keys: img, mask
wb_256_layered/rhoa_2d_<id>.mat          keys: img, mask
slm_256_layered/rhoa_2d_<id>.mat         keys: img, mask
```

## Quick checks

From the repository root:

```bash
python src/evaluate_new_experiment.py unet_open \
  --data-root examples/demo_data \
  --run-dir checkpoints/sa_unet_ol \
  --max-samples 2
```

This command requires a compatible checkpoint. If checkpoints are not present,
the command documents the expected evaluation interface but will stop with a
missing-checkpoint error.

To inspect the PLB module:

```bash
python -m py_compile src/lnn_unet.py src/multi_lnn_unet.py
```

## Training workflows

The scripts implement the following training stages:

```bash
# SA-U-Net-OL
python src/train_wa_unet.py open_inverse --data-root <data_root> --work-dir <run_dir>

# SA-U-Net-CL, stage 1: forward surrogate
python src/train_wa_unet.py forward --data-root <data_root> --work-dir <run_dir>

# SA-U-Net-CL, stage 2: closed-loop inversion
python src/train_wa_unet.py closed_inverse --data-root <data_root> --work-dir <run_dir> \
  --forward-checkpoint <run_dir>/forward/forward_unet_best.pth

# SA-PLB-U-Net-CL
python src/train_wa_closed_loop_lnn.py forward --data-root <data_root> --work-dir <run_dir>
python src/train_wa_closed_loop_lnn.py inverse --data-root <data_root> --work-dir <run_dir>

# MA-PLB-U-Net-CL
python src/train_multi_closed_loop_lnn.py forward --data-root <data_root> --work-dir <run_dir>
python src/train_multi_closed_loop_lnn.py inverse --data-root <data_root> --work-dir <run_dir>
```

The full manuscript experiments used 1600 training samples and 400 validation
samples. The included six-sample demo set is too small for meaningful training.

## Metrics

The manuscript reports:

- best validation loss,
- RMSE,
- MAE,
- PSNR,
- SSIM,
- R2.

For closed-loop inversion, `best_val_loss` is the weighted total loss, while
the fair model-domain comparison uses the unweighted `L_inv`.

The reported metric summary is available at:

```text
results/experiment_metrics_summary.txt
```

## Data and checkpoint availability

The small demo data are included in this repository. Trained best checkpoints
are distributed as GitHub Release assets. The full synthetic dataset is large
and should be distributed through a persistent archive such as Zenodo or GitHub
Releases. Update this section with the DOI or release URL before journal
submission.

Field data may be subject to project-specific restrictions. If the full field
data cannot be publicly redistributed, provide an anonymized or processed
example and document the restriction in the manuscript's data availability
statement.

## License

This code is released under the MIT License. See `LICENSE`.

## Citation

If this repository is useful for your work, please cite the associated
manuscript. A `CITATION.cff` file is included and should be updated with the
final title, DOI, and author information after publication.
