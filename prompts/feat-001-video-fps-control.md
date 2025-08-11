# feat-001-video-fps-control.md

Implement automatic video FPS calculation based on simulation timing to ensure accurate playback speed.

## Context

The current video recording system uses a fixed `videoFps` configuration that's independent of simulation timing, causing videos to play at incorrect speeds. This creates temporal inaccuracy where recorded videos don't match the actual simulation playback speed.

**Root Cause**: VideoRecorder uses hardcoded FPS while simulation runs at a different frequency determined by `control_dt`.

**Physics Relationship**: Simulation frequency = 1/control_dt, but video encoding uses unrelated config FPS.

## Current State

- VideoRecorder initialized with fixed `env.videoFps` from config (default: 60.0)
- All frames captured during render() are recorded without timing consideration
- Video playback speed incorrect when simulation frequency ≠ configured videoFps
- Example: 50Hz simulation (control_dt=0.02) + 60fps config = 1.2x speed video

## Desired Outcome

- Video FPS automatically calculated as `1.0 / control_dt` for accurate real-time playback
- Remove obsolete `videoFps` configuration option
- Videos play back at correct simulation speed regardless of physics timing
- Maintain all temporal information without frame dropping

## Constraints

- **Two-Stage Initialization**: VideoRecorder created before `control_dt` is measured
- **Component Architecture**: Must follow established `finalize_setup()` pattern like ActionProcessor
- **Fail-Fast Philosophy**: No defensive programming - crash if VideoRecorder used before finalization
- **Single Source of Truth**: FPS comes from physics timing, not config

## Implementation Notes

**Architecture Pattern**: Implement deferred FPS setting following ActionProcessor model:

1. **Phase 1 (Creation)**: VideoRecorder(fps=None) before control_dt available
2. **Phase 2 (Finalization)**: video_recorder.finalize_fps(1.0 / control_dt) after measurement

**Key Changes**:
- Add `finalize_fps()` method to VideoRecorder class
- Modify initialization to accept fps=None initially
- Add finalization call in `_perform_control_cycle_measurement()`
- Remove `videoFps` from base/video.yaml
- Update create_video_recorder_from_config() to handle missing fps

**Testing Approach**:
- Verify different control_dt values produce correct video FPS
- Test initialization order (finalize before recording)
- Validate video playback speed matches simulation timing

## Dependencies

- Requires understanding of two-stage initialization pattern
- Must preserve existing video recording functionality
- Should maintain HTTP streaming compatibility (uses separate FPS logic)
