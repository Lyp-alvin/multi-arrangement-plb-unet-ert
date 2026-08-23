$ErrorActionPreference = "Stop"

python -m py_compile `
  src/plain_unet.py `
  src/lnn_unet.py `
  src/multi_lnn_unet.py `
  src/train_wa_unet.py `
  src/train_wa_closed_loop_lnn.py `
  src/train_multi_closed_loop_lnn.py `
  src/evaluate_new_experiment.py

Write-Host "Python source files compiled successfully."

