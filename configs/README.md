# Config notes

The original training scripts use command-line arguments rather than a YAML
configuration system. This directory documents the parameter settings used in
the manuscript.

Main settings:

```text
optimizer: AdamW
learning_rate: 1e-4
weight_decay: 1e-4
scheduler: ReduceLROnPlateau
scheduler_patience: 2
maximum_epochs: 100
early_stopping_patience: 10
AMP: disabled
normalization: GroupNorm for PLB-based models
closed_loop_inverse_weight: 0.8
closed_loop_forward_weight: 0.2
```

