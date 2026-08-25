# Workflow commands

The following commands document the main training and testing workflows.
Replace `<data_root>` and `<run_dir>` with local paths.

## SA-U-Net-OL

```bash
python src/train_wa_unet.py open_inverse --data-root <data_root> --work-dir <run_dir>
python src/test_new_experiment.py unet_open --data-root <data_root> --run-dir <run_dir>
```

## SA-U-Net-CL

```bash
python src/train_wa_unet.py forward --data-root <data_root> --work-dir <run_dir>
python src/test_new_experiment.py unet_forward --data-root <data_root> --run-dir <run_dir>

python src/train_wa_unet.py closed_inverse --data-root <data_root> --work-dir <run_dir> \
  --forward-checkpoint <run_dir>/forward/forward_unet_best.pth
python src/test_new_experiment.py unet_closed --data-root <data_root> --run-dir <run_dir>
```

## SA-LIB-U-Net-CL

```bash
python src/train_wa_closed_loop_lnn.py forward --data-root <data_root> --work-dir <run_dir>
python src/train_wa_closed_loop_lnn.py inverse --data-root <data_root> --work-dir <run_dir>
python src/test_wa_closed_loop.py --data-root <data_root> --run-dir <run_dir>
```

## MA-LIB-U-Net-CL

```bash
python src/train_multi_closed_loop_lnn.py forward --data-root <data_root> --work-dir <run_dir>
python src/test_new_experiment.py multi_forward --data-root <data_root> --run-dir <run_dir>

python src/train_multi_closed_loop_lnn.py inverse --data-root <data_root> --work-dir <run_dir>
python src/test_new_experiment.py multi_inverse --data-root <data_root> --run-dir <run_dir>
```
