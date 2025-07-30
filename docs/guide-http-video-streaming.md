# HTTP Video Streaming Guide

This guide explains how to use the HTTP video streaming feature to view Isaac Gym environments in real-time through a web browser.

## Overview

The HTTP video streaming feature allows you to:
- View training/testing in real-time through a web browser
- Monitor multiple environments remotely
- Avoid disk space issues from video files
- Share live views with multiple viewers simultaneously

## Installation

Install the streaming dependencies:

```bash
pip install -e .[streaming]
```

Or install all optional dependencies:

```bash
pip install -e .[all]
```

## Configuration

### Basic Streaming Setup

To enable HTTP streaming, set `videoStream: true` in your configuration:

```yaml
# dexhand_env/cfg/test_stream.yaml
env:
  viewer: false        # Headless mode
  videoRecord: false   # Disable file recording
  videoStream: true    # Enable HTTP streaming

  # HTTP streaming configuration
  videoStreamHost: "127.0.0.1"     # Server host
  videoStreamPort: 8080             # Server port
  videoStreamQuality: 85            # JPEG quality (1-100)
  videoStreamBufferSize: 10         # Frame buffer size

  # Video settings (shared with recording)
  videoFps: 30.0
  videoResolution: [1024, 768]
```

### Available Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `videoStream` | `false` | Enable/disable HTTP streaming |
| `videoStreamHost` | `"127.0.0.1"` | Server host address |
| `videoStreamPort` | `8080` | Server port |
| `videoStreamQuality` | `85` | JPEG quality (1-100, higher = better quality) |
| `videoStreamBufferSize` | `10` | Frame buffer size |

### Recording + Streaming

You can enable both file recording and HTTP streaming simultaneously:

```yaml
env:
  videoRecord: true    # Save to files
  videoStream: true    # Also stream via HTTP
```

## Usage

### Starting Training with Streaming

```bash
# Using the streaming configuration
python train.py --config-name test_stream

# Or enable streaming with any config
python train.py --config-name test_record env.videoStream=true
```

### Viewing the Stream

Once training starts, you'll see a message like:
```
HTTP video streamer started: http://127.0.0.1:8080
```

Open this URL in your web browser to view the live stream.

### Web Interface Features

The web interface provides:
- **Live Video Stream**: Real-time MJPEG stream from Isaac Gym
- **Statistics Panel**: Frame rates, uptime, client count
- **Auto-refresh**: Stats update automatically every 2 seconds

## Advanced Configuration

### Remote Access

To allow remote access, change the host:

```yaml
env:
  videoStreamHost: "0.0.0.0"  # Allow connections from any IP
  videoStreamPort: 8080
```

**Security Note**: Only use `0.0.0.0` in trusted networks.

### Performance Tuning

For better performance:

```yaml
env:
  videoStreamQuality: 70        # Lower quality for faster streaming
  videoStreamBufferSize: 5      # Smaller buffer for lower latency
  videoFps: 15.0               # Lower FPS for slower networks
```

For higher quality:

```yaml
env:
  videoStreamQuality: 95        # Higher quality
  videoStreamBufferSize: 20     # Larger buffer
  videoResolution: [1920, 1080] # Higher resolution
```

## Troubleshooting

### Common Issues

**1. "Flask is required for HTTP video streaming"**
```bash
pip install flask
# or
pip install -e .[streaming]
```

**2. "Failed to start HTTP video streamer"**
- Check if port 8080 is already in use
- Try a different port: `env.videoStreamPort=8081`

**3. "Connection refused" in browser**
- Ensure the training script is running
- Check firewall settings
- Verify the host/port configuration

**4. Slow or choppy streaming**
- Reduce video quality: `env.videoStreamQuality=60`
- Lower FPS: `env.videoFps=15.0`
- Smaller resolution: `env.videoResolution=[800, 600]`

### Debugging

Enable debug logging:

```yaml
env:
  logLevel: DEBUG
```

Check server stats programmatically:
```bash
curl http://127.0.0.1:8080/stats
```

## Examples

### Streaming-Only Mode

```bash
python train.py --config-name test_stream
```

### Recording + Streaming

```bash
python train.py --config-name test_record env.videoStream=true
```

### Custom Port

```bash
python train.py --config-name test_stream env.videoStreamPort=8081
```

### High Quality Stream

```bash
python train.py --config-name test_stream \
  env.videoStreamQuality=95 \
  env.videoResolution=[1920,1080]
```

## API Reference

### Configuration Keys

All streaming configuration goes under `env:` in your YAML config:

- `videoStream`: Enable streaming (boolean)
- `videoStreamHost`: Server host (string)
- `videoStreamPort`: Server port (integer)
- `videoStreamQuality`: JPEG quality 1-100 (integer)
- `videoStreamBufferSize`: Frame buffer size (integer)

### URLs

- **Stream**: `http://host:port/` - Main viewer page
- **Video**: `http://host:port/stream` - Raw MJPEG stream
- **Stats**: `http://host:port/stats` - JSON statistics

### Statistics

The `/stats` endpoint returns:
```json
{
  "frames_received": 1250,
  "frames_served": 1200,
  "frames_dropped": 50,
  "clients_connected": 2,
  "uptime_seconds": 42.5,
  "current_fps": 28.2,
  "server_running": true,
  "host": "127.0.0.1",
  "port": 8080
}
```

## Security Considerations

- The streaming server is intended for development/research use
- Use Flask's development server (not production-ready)
- Only expose to trusted networks
- No authentication is implemented
- Consider using SSH tunneling for remote access:

```bash
ssh -L 8080:localhost:8080 user@remote-machine
```

## Performance Notes

- Streaming adds minimal CPU overhead
- Network bandwidth scales with quality and FPS
- Multiple clients share the same stream efficiently
- Frame dropping prevents server overload
- Works well with headless Isaac Gym environments
