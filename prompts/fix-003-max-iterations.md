train.maxIterations or training.maxIterations in config?

This part is really ugly:
```
    if cfg.train.maxIterations != 10000:  # Default from config.yaml
        overrides.append(f"train.maxIterations={cfg.train.maxIterations}")
```

Overrides in task yaml fail.

`train.py` as a lot of ugly code.
