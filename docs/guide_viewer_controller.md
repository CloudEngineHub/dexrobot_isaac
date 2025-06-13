# Viewer Controller Guide

This guide explains how to use the viewer controller for interactive control of the DexHand environment.

## Overview

The ViewerController component provides keyboard shortcuts for controlling the camera view, navigating between robots, and resetting environments during visualization. It's automatically enabled when running with a viewer (non-headless mode).

## Keyboard Shortcuts

### Camera Controls

| Key | Action | Description |
|-----|--------|-------------|
| **Enter** | Toggle View Mode | Cycles through camera views: Free → Rear → Right → Bottom → Free |
| **G** | Toggle Follow Mode | Switches between following a single robot or viewing all robots globally |

### Navigation Controls

| Key | Action | Description |
|-----|--------|-------------|
| **↑** (Up Arrow) | Previous Robot | Navigate to the previous robot (only in single follow mode) |
| **↓** (Down Arrow) | Next Robot | Navigate to the next robot (only in single follow mode) |

### Environment Controls

| Key | Action | Description |
|-----|--------|-------------|
| **P** | Reset Environment | Reset the currently selected robot/environment |

## Camera View Modes

### 1. Free Camera
- Manual camera control using mouse
- No automatic following
- Full freedom to position camera anywhere

### 2. Rear View
- Camera positioned behind the hand
- Follows hand movement automatically
- Good for observing finger movements from behind

### 3. Right View  
- Camera positioned to the right of the hand
- Follows hand movement automatically
- Useful for side perspective of grasping

### 4. Bottom View
- Camera positioned below, looking up at the hand
- Follows hand movement automatically
- Ideal for observing palm and contact points

## Follow Modes

### Single Robot Mode (Default)
- Camera follows one specific robot
- Use arrow keys to switch between robots
- Camera stays focused on the selected robot
- Useful for detailed observation of individual behaviors

### Global View Mode
- Camera shows all robots at once
- Camera position centers on all robots
- Increased camera distance for wider view
- Useful for comparing multiple robots or batch training

## Usage Examples

### Basic Interaction
```bash
# Run with viewer enabled
python examples/dexhand_test.py

# During execution:
# - Press Enter to cycle through camera views
# - Press G to toggle between single/global view
# - Press ↑/↓ to change which robot to follow
# - Press P to reset the current robot
```

### Multi-Environment Setup
```bash
# Run with multiple environments
python examples/dexhand_test.py --num-envs 4

# Use global view (G) to see all robots
# Switch to single mode (G) to focus on one
# Navigate between robots with arrow keys
```

## Console Feedback

The viewer controller provides console output for all actions:
- Camera mode changes: `"Camera: Rear View (following robot 0)"`
- Follow mode changes: `"Camera: Rear View (global view)"`
- Robot selection: `"Following robot 2"`
- Invalid actions: `"Cannot change robot in global view mode. Press Tab to switch to single robot mode."`

## Implementation Details

### Default Settings
- Starts in **Rear View** with **Single Robot** follow mode
- Follows robot 0 by default
- All keyboard events are automatically subscribed when viewer is created

### Camera Positioning
- Each view mode has predefined offset positions
- Global view increases camera distance for better overview
- Camera smoothly updates position each frame when following

### Integration with Environment
- Reset command (`P` key) triggers `reset_idx()` for selected environment
- Camera updates use hand positions from rigid body states
- Works seamlessly with both CPU and GPU pipelines

## Troubleshooting

### Camera Not Following
- Ensure you're not in Free Camera mode (press Enter to cycle)
- Check that follow mode is set to Single (press G if needed)
- Verify hand positions are being updated correctly

### Cannot Change Robot
- You must be in Single Robot mode to navigate between robots
- Press G to switch from Global to Single mode
- Then use arrow keys to select different robots

### Viewer Not Responding
- Ensure viewer was created (not running in headless mode)
- Check that keyboard events are being processed
- Verify no other application has keyboard focus