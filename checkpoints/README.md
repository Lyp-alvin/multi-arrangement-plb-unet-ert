# Checkpoints

Trained model files are distributed as GitHub Release assets, not as normal Git
files.

Expected checkpoint layout:

```text
checkpoints/
  sa_unet_ol/open_inverse/inverse_open_loop_best.pth
  sa_unet_cl/forward/forward_unet_best.pth
  sa_unet_cl/closed_inverse/inverse_closed_loop_best.pth
  sa_plb_unet_cl/forward/forward_lnn_unet_best.pth
  sa_plb_unet_cl/inverse/inverse_lnn_unet_best.pth
  ma_plb_unet_cl/forward/forward_multi_lnn_best.pth
  ma_plb_unet_cl/inverse/inverse_multi_closed_loop_best.pth
```

Download the release archive named
[`checkpoints_best.zip`](https://github.com/Lyp-alvin/multi-arrangement-plb-unet-ert/releases/download/v1.0.0/checkpoints_best.zip)
from GitHub Release
[`v1.0.0`](https://github.com/Lyp-alvin/multi-arrangement-plb-unet-ert/releases/tag/v1.0.0)
and extract it into this directory. The extracted folder layout should match
the tree shown above.

For long-term archival, a Zenodo snapshot can also be added before journal
submission.
