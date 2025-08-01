# Indefinite Policy Testing with Hot-Reload

Monitor your training progress in real-time by running indefinite policy testing with automatic checkpoint reloading.

## Basic Setup

Use `testGamesNum=0` to run indefinitely until Ctrl+C:

```bash
python train.py train.test=true testGamesNum=0 train.checkpoint=runs/MyExperiment
```

## Recommended Workflow

The most effective development pattern uses two terminals:

**Terminal 1 - Training:**
```bash
python train.py config=train_headless task=BlindGrasping
```

**Terminal 2 - Live Monitoring:**
```bash
python train.py config=test_viewer testGamesNum=0 checkpoint=latest
```

The test process automatically reloads newer checkpoints every 30 seconds, letting you watch the policy improve as training progresses.

## Video Monitoring

### On Remote Servers

Use HTTP streaming to monitor training from anywhere:

```bash
# Training
python train.py config=train_headless task=BlindGrasping

# Monitoring with streaming
python train.py config=test_stream testGamesNum=0 checkpoint=latest streamBindAll=true
```

Access the stream at `http://server-ip:58080`

### On Local Workstations

Use rendering on local Isaac Gym viewer with checkpoint synchronization:

**Server:**
```bash
python train.py config=train_headless task=BlindGrasping
```

**Local - Sync checkpoints:**
```bash
# Keep checkpoints synced
while true; do
  rsync -av server:/path/to/runs/ ./runs
  sleep 30
done &
```
or use file synchronization tools like `unison`.

**Local - Viewing:**
```bash
python train.py config=test_viewer testGamesNum=0 checkpoint=latest
```

## Configuration Presets

Use existing test configurations instead of long command lines:

- `test_viewer.yaml` - Interactive viewer with 4 environments
- `test_stream.yaml` - HTTP streaming for remote monitoring
- `test.yaml` - Headless testing base configuration

Override specific parameters as needed:

```bash
python train.py config=test_viewer testGamesNum=0 numEnvs=1
```

## Hot-Reload Details

The system monitors your experiment directory and automatically:
1. Detects when checkpoints are updated
2. Loads the newer model seamlessly
3. Continues testing without interruption
4. Logs reload events clearly

Set `train.reloadInterval` (in seconds) to balance responsiveness vs overhead. 30-60 seconds works well for most cases.
