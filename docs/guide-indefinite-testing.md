# Indefinite Policy Testing with Hot-Reload

During training, you want to monitor how your policy evolves without constantly restarting test scripts or losing visual feedback. This guide shows how to set up continuous policy monitoring that automatically loads new checkpoints as training progresses.

## The Problem

Traditional policy evaluation during training is cumbersome:
- **Manual restarts**: Stop test script, find latest checkpoint, restart with new path
- **Static evaluation**: Test a frozen checkpoint while training continues
- **Visual gaps**: No continuous visual feedback on policy improvement

## The Solution: Hot-Reload Testing

Hot-reload testing solves this by running indefinite policy testing with automatic checkpoint discovery and reloading. The system continuously monitors your experiment directory and seamlessly loads newer checkpoints without interrupting the visual feedback loop.

**Key capabilities:**
- **Automatic discovery**: `checkpoint=latest` finds your most recent training experiment using the experiment management system (see [TRAINING.md](../TRAINING.md))
- **Live reloading**: Monitors the experiment directory and loads new checkpoints every 30 seconds (configurable)
- **Indefinite testing**: Runs until manual termination (`testGamesNum=0`)
- **Deployment flexibility**: Works with local Isaac Gym viewer or remote HTTP streaming

**The `checkpoint=latest` magic:**
1. **Directory discovery**: Resolves to latest experiment directory via `runs/latest_train` symlink
2. **Continuous monitoring**: Watches the resolved directory (not a static file) for new checkpoints
3. **Dynamic loading**: Automatically loads the newest `.pth` file found in `nn/` subdirectory

## Deployment Scenarios

### Scenario 1: Local Workstation with Server Training

**When to use**: You can run Isaac Gym viewer locally but training happens on a remote server.

**Advantages**: Full Isaac Gym interactivity, better visual quality, local keyboard controls
**Trade-offs**: Requires checkpoint synchronization, slightly more setup

**Server (training):**
```bash
python train.py config=train_headless task=BlindGrasping
```

**Local (checkpoint sync):**
```bash
# Option A: Simple rsync loop
while true; do
  rsync -av server:/path/to/dexrobot_isaac/runs/ ./runs/
  sleep 30
done &

# Option B: File synchronization tools
unison server_profile -repeat 30
```

**Local (testing):**
```bash
python train.py config=test_viewer testGamesNum=0 checkpoint=latest
# Uses runs/latest_train symlink → experiment directory → newest checkpoint
```

### Scenario 2: Remote Server Monitoring

**When to use**: Training and testing both happen on remote server, monitor via browser.

**Advantages**: No file synchronization needed, accessible from anywhere, simpler setup
**Trade-offs**: HTTP streaming limitations, browser-based viewing only

**Server (training):**
```bash
python train.py config=train_headless task=BlindGrasping
```

**Server (monitoring):**
```bash
python train.py config=test_stream testGamesNum=0 checkpoint=latest streamBindAll=true
# streamBindAll enables access from external IPs (security warning applies)
```

**Access**: Open `http://server-ip:58080` in browser

## Basic Usage

**Indefinite testing with hot-reload:**
```bash
python train.py config=test_viewer testGamesNum=0 checkpoint=latest
```

**Customize reload timing:**
```bash
python train.py config=test_viewer testGamesNum=0 checkpoint=latest reloadInterval=60
```

**Use specific experiment:**
```bash
python train.py config=test_viewer testGamesNum=0 checkpoint=runs/BlindGrasping_train_20250801_095943
```

## Configuration Reference

**Test duration control:**
- `testGamesNum=0`: Run indefinitely until Ctrl+C (most common for monitoring)
- `testGamesNum=25`: Run exactly 25 episodes then terminate

**Hot-reload settings:**
- `reloadInterval=30`: Check for new checkpoints every 30 seconds (default)
- `reloadInterval=0`: Disable hot-reload, use static checkpoint

**Configuration presets:**
- `test_viewer.yaml`: Interactive Isaac Gym viewer (4 environments)
- `test_stream.yaml`: HTTP video streaming (headless)
- `test.yaml`: Base headless testing configuration

**Parameter overrides:**
```bash
# Fewer environments for smoother visualization
python train.py config=test_viewer testGamesNum=0 checkpoint=latest numEnvs=1

# Longer reload interval to reduce overhead
python train.py config=test_viewer testGamesNum=0 checkpoint=latest reloadInterval=120
```

## How Hot-Reload Works

The hot-reload system uses a background thread that:

1. **Resolves experiment directory**: `checkpoint=latest` → `runs/latest_train` symlink → actual experiment directory
2. **Monitors for changes**: Uses `find_latest_checkpoint_file()` to check for new `.pth` files in the experiment's `nn/` directory
3. **Detects updates**: Compares file modification times every `reloadInterval` seconds
4. **Loads seamlessly**: When a newer checkpoint is found, loads the new weights into the running policy without interrupting the episode
5. **Logs events**: Clear console output shows when reloads occur

This design enables true continuous monitoring - you start the test process once and watch your policy improve throughout the entire training session.
