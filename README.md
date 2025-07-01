```bash
# registers sweep config  on the W&B backend (one-time)
wandb sweep sweep.yaml

# Terminal prints:  Sweep ID: 3k1xg8wq
# 2 (start an agent) – one per GPU / machine
wandb agent <ENTITY>/<PROJECT>/<SWEEP-ID>
```