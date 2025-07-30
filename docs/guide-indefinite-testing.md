# Indefinite Policy Testing with Hot-Reload

This guide covers the comprehensive workflow for running indefinite policy testing with hot-reload functionality, enabling continuous policy evaluation with automatic model updates.

## Overview

Indefinite testing mode allows you to run policy evaluation continuously until manual termination. When combined with hot-reload functionality, this enables powerful workflows for:

- **Live Policy Development**: Watch policy behavior update in real-time as training progresses
- **Long-term Evaluation**: Extended testing runs to assess policy stability
- **Interactive Debugging**: Manual observation of policy behavior over extended periods
- **Performance Monitoring**: Continuous assessment of policy improvements

## Core Configuration

### Test Games Parameter

The `testGamesNum` parameter controls test duration:

```yaml
train:
  testGamesNum: 100    # Finite: Run exactly 100 games then terminate
  testGamesNum: 0      # Indefinite: Run until manual termination
```

### Configuration Hierarchy

1. **Default Config**: `dexhand_env/cfg/config.yaml` - `testGamesNum: 100`
2. **Test Configs**: `dexhand_env/cfg/base/test.yaml` - `testGamesNum: 50`
3. **CLI Override**: `python train.py train.testGamesNum=0`

## Usage Patterns

### 1. Basic Indefinite Testing

```bash
# Start indefinite testing
python train.py train.test=true train.testGamesNum=0 train.checkpoint=path/to/checkpoint.pth

# Stop with Ctrl+C when done
```

### 2. Indefinite Testing with Hot-Reload

**Most Powerful Workflow**: Automatically reload updated checkpoints during testing:

```bash
# Start indefinite testing with hot-reload monitoring
python train.py \
  train.test=true \
  train.testGamesNum=0 \
  train.checkpoint=runs/MyExperiment/checkpoint.pth \
  train.reloadInterval=30 \
  env.viewer=true
```

**What happens:**
1. Policy loads from checkpoint and starts running indefinitely
2. Every 30 seconds, system checks if checkpoint was updated
3. If newer checkpoint detected, policy automatically reloads
4. Testing continues with updated policy seamlessly
5. Viewer shows real-time policy behavior changes

### 3. Multi-Environment Indefinite Testing

```bash
# Scale up for better statistics
python train.py \
  train.test=true \
  train.testGamesNum=0 \
  train.checkpoint=runs/MyExperiment/checkpoint.pth \
  env.numEnvs=16 \
  env.viewer=true
```

### 4. Headless Indefinite Testing

```bash
# Run indefinitely without visualization (faster)
python train.py \
  train.test=true \
  train.testGamesNum=0 \
  train.checkpoint=runs/MyExperiment/checkpoint.pth \
  headless=true \
  env.numEnvs=32
```

## Integration with Training Pipeline

### Simultaneous Training and Testing

Run training and testing simultaneously to observe live policy improvements:

**Terminal 1 - Training:**
```bash
# Start training
python train.py task=BlindGrasping env.numEnvs=4096
```

**Terminal 2 - Live Testing:**
```bash
# Monitor training progress with indefinite testing + hot-reload
python train.py \
  train.test=true \
  train.testGamesNum=0 \
  train.reloadInterval=30 \
  train.checkpoint=runs/latest_train/checkpoint.pth \
  env.viewer=true \
  env.numEnvs=4
```

### Hot-Reload Configuration

Hot-reload monitoring works by:

1. **Checkpoint Detection**: Monitors experiment directory for checkpoint updates
2. **Model Reloading**: Automatically loads newer models when detected
3. **Seamless Transition**: Policy updates without interrupting evaluation
4. **Logging**: Clear logs show when reloads occur

**Key Parameters:**
```yaml
train:
  checkpoint: "path/to/experiment/checkpoint.pth"  # Initial checkpoint to load
  reloadInterval: 30                               # Check interval in seconds
```

## Video Recording During Indefinite Testing

### Record Video Segments

```bash
# Record indefinite testing with video capture
python train.py \
  train.test=true \
  train.testGamesNum=0 \
  train.checkpoint=runs/MyExperiment/checkpoint.pth \
  env.viewer=false \
  env.videoRecord=true \
  env.videoMaxDuration=300  # 5-minute segments
```

### Live HTTP Streaming

```bash
# Stream indefinite testing over HTTP
python train.py \
  train.test=true \
  train.testGamesNum=0 \
  train.checkpoint=runs/MyExperiment/checkpoint.pth \
  env.videoStream=true \
  env.videoStreamPort=8080
```

Then access stream at: `http://localhost:8080`

## Advanced Workflows

### 1. Policy Comparison Workflow

Compare different checkpoints by switching between them:

```bash
# Test checkpoint A indefinitely
python train.py train.test=true train.testGamesNum=0 train.checkpoint=checkpointA.pth

# Switch to checkpoint B (new terminal)
python train.py train.test=true train.testGamesNum=0 train.checkpoint=checkpointB.pth
```

### 2. Automated Testing Scripts

Create scripts for systematic policy evaluation:

```python
#!/usr/bin/env python3
import subprocess
import time

checkpoints = ["checkpoint_1000.pth", "checkpoint_2000.pth", "checkpoint_3000.pth"]

for checkpoint in checkpoints:
    print(f"Testing {checkpoint} for 10 minutes...")

    # Start indefinite testing
    process = subprocess.Popen([
        "python", "train.py",
        "train.test=true",
        "train.testGamesNum=0",
        f"train.checkpoint={checkpoint}",
        "headless=true"
    ])

    # Let it run for 10 minutes
    time.sleep(600)

    # Terminate and move to next
    process.terminate()
    process.wait()
```

### 3. Multi-Task Evaluation

Test across different tasks indefinitely:

```bash
# Task 1: BaseTask indefinite testing
python train.py task=BaseTask train.test=true train.testGamesNum=0 train.checkpoint=base_checkpoint.pth &

# Task 2: BlindGrasping indefinite testing
python train.py task=BlindGrasping train.test=true train.testGamesNum=0 train.checkpoint=grasp_checkpoint.pth &
```

## Best Practices

### 1. Resource Management

- **Monitor GPU Memory**: Indefinite testing uses GPU continuously
- **Set Reasonable Environment Counts**: Balance statistics vs. resource usage
- **Use Headless Mode**: For background/overnight testing

### 2. Logging and Monitoring

- **Check Logs Periodically**: Monitor for errors or unusual behavior
- **TensorBoard Integration**: Metrics logged during indefinite testing appear in TensorBoard
- **Manual Termination**: Always use Ctrl+C for clean shutdown

### 3. Hot-Reload Best Practices

- **Checkpoint Paths**: Use relative paths that remain valid as training progresses
- **Reload Intervals**: 30-60 seconds provides good balance of responsiveness vs. overhead
- **Directory Structure**: Ensure experiment directory structure is consistent

### 4. Development Workflow Integration

**Recommended Development Cycle:**

1. **Start Training**: Begin policy training in background
2. **Launch Indefinite Testing**: Start hot-reload testing in viewer
3. **Observe Behavior**: Watch policy improvements in real-time
4. **Identify Issues**: Spot problems as they emerge
5. **Adjust Training**: Modify hyperparameters based on observations
6. **Continue Cycle**: Repeat until satisfied with policy performance

## Troubleshooting

### Common Issues

**Issue**: Indefinite testing terminates unexpectedly
- **Cause**: Environment errors or hardware issues
- **Solution**: Check logs for error messages, verify GPU memory availability

**Issue**: Hot-reload doesn't detect new checkpoints
- **Cause**: Incorrect checkpoint path or file permissions
- **Solution**: Verify checkpoint path exists and is writable, check reload interval

**Issue**: High GPU memory usage during indefinite testing
- **Cause**: Too many parallel environments
- **Solution**: Reduce `env.numEnvs` parameter

**Issue**: Testing runs slowly
- **Cause**: Viewer overhead or too many environments
- **Solution**: Use `headless=true` or reduce environment count

### Performance Optimization

- **Headless Mode**: 2-3x faster than viewer mode
- **Environment Count**: Scale based on GPU memory (4-32 typical range)
- **Hot-Reload Interval**: Longer intervals reduce overhead
- **Video Recording**: Significantly impacts performance if enabled

## Examples and Use Cases

### Research and Development

```bash
# Research workflow: Train new algorithm while monitoring performance
python train.py task=BlindGrasping env.numEnvs=8192 &
python train.py train.test=true train.testGamesNum=0 train.reloadInterval=30 \
  train.checkpoint=runs/latest_train/checkpoint.pth env.viewer=true env.numEnvs=4
```

### Demo and Presentation

```bash
# Demo workflow: Show policy performance with video streaming
python train.py train.test=true train.testGamesNum=0 \
  train.checkpoint=runs/best_policy/checkpoint.pth \
  env.videoStream=true env.viewer=true env.numEnvs=1
```

### Automated Testing Infrastructure

```bash
# CI/CD workflow: Automated policy regression testing
python train.py train.test=true train.testGamesNum=0 \
  train.checkpoint=latest_release.pth headless=true \
  env.numEnvs=64 > test_results.log &

# Let run for predetermined time, then analyze logs
```

This indefinite testing workflow with hot-reload provides powerful capabilities for policy development, evaluation, and deployment scenarios.
