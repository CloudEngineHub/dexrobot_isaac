"""
Video recording component for Isaac Gym environments.

This module provides video recording functionality that captures frames during
environment execution and saves them as MP4 video files.
"""

import os
import threading
from pathlib import Path
from queue import Queue
from typing import Optional, Tuple, Dict, Any
from functools import wraps

import cv2
import numpy as np
from loguru import logger


def require_finalized_fps(func):
    """Decorator to ensure FPS has been finalized before method execution."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_fps_finalized") or not self._fps_finalized:
            raise RuntimeError(
                f"{func.__name__} can only be called after finalize_fps()"
            )
        return func(self, *args, **kwargs)

    return wrapper


class VideoRecorder:
    """
    Video recorder that captures RGB frames and saves them as MP4 files.

    Features:
    - Asynchronous frame writing for performance
    - Configurable video settings (FPS, resolution, codec)
    - Smart filename generation with episode tracking
    - Automatic cleanup and resource management
    """

    def __init__(
        self,
        output_dir: str,
        fps: Optional[float] = None,
        resolution: Tuple[int, int] = (1024, 768),
        codec: str = "mp4v",
        max_duration: Optional[float] = None,
        max_frames_per_episode: Optional[int] = None,
    ):
        """
        Initialize the video recorder.

        Args:
            output_dir: Directory to save video files
            fps: Frames per second for video output (None = defer until finalize_fps())
            resolution: Video resolution as (width, height)
            codec: Video codec ('mp4v', 'XVID', 'H264')
            max_duration: Maximum duration per video in seconds
            max_frames_per_episode: Maximum frames per episode video
        """
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.resolution = resolution
        self.codec = codec
        self.max_duration = max_duration
        self.max_frames_per_episode = max_frames_per_episode

        # Two-stage initialization state
        self._fps_finalized = fps is not None

        # Calculate max frames from duration if FPS is available
        if max_duration and not max_frames_per_episode and fps is not None:
            self.max_frames_per_episode = int(max_duration * fps)

        # Video writer state
        self._writer: Optional[cv2.VideoWriter] = None
        self._current_filename: Optional[str] = None
        self._frame_count = 0
        self._episode_count = 0
        self._is_recording = False

        # Async writing
        self._frame_queue: Queue = Queue(maxsize=100)
        self._write_thread: Optional[threading.Thread] = None
        self._stop_writing = threading.Event()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if fps is not None:
            logger.info(
                f"VideoRecorder initialized: {output_dir} @ {fps}fps {resolution}"
            )
        else:
            logger.info(
                f"VideoRecorder initialized: {output_dir} @ deferred_fps {resolution}"
            )
            logger.info("  Note: finalize_fps() must be called before recording")

    def finalize_fps(self, fps: float):
        """
        Finalize FPS setting after simulation timing measurement.

        Args:
            fps: Measured frames per second from simulation (typically 1.0 / control_dt)
        """
        if self._fps_finalized:
            raise RuntimeError("VideoRecorder FPS already finalized")

        if fps <= 0:
            raise ValueError(f"FPS must be positive, got {fps}")

        self.fps = fps
        self._fps_finalized = True

        # Calculate max frames from duration if specified and not already set
        if self.max_duration and not self.max_frames_per_episode:
            self.max_frames_per_episode = int(self.max_duration * fps)

        logger.info(f"VideoRecorder FPS finalized: {fps:.3f}fps")
        if self.max_frames_per_episode:
            logger.info(f"  Max frames per episode: {self.max_frames_per_episode}")

    @require_finalized_fps
    def start_episode_recording(self, episode_id: Optional[int] = None) -> bool:
        """
        Start recording a new episode video.

        Args:
            episode_id: Optional episode ID for filename

        Returns:
            True if recording started successfully
        """
        if self._is_recording:
            logger.warning("Already recording, stopping current recording first")
            self.stop_recording()

        # Generate filename
        if episode_id is not None:
            filename = f"episode_{episode_id:04d}.mp4"
        else:
            self._episode_count += 1
            filename = f"episode_{self._episode_count:04d}.mp4"

        self._current_filename = str(self.output_dir / filename)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(
            self._current_filename, fourcc, self.fps, self.resolution
        )

        if not self._writer.isOpened():
            logger.error(f"Failed to create video writer for {self._current_filename}")
            self._writer = None
            return False

        # Start async writing thread
        self._stop_writing.clear()
        self._write_thread = threading.Thread(target=self._async_writer, daemon=True)
        self._write_thread.start()

        self._frame_count = 0
        self._is_recording = True

        logger.info(f"Started recording episode video: {filename}")
        return True

    def add_frame(self, frame: np.ndarray) -> bool:
        """
        Add a frame to the current recording.

        Args:
            frame: RGB frame as numpy array (H, W, 3) with values 0-255

        Returns:
            True if frame was added successfully
        """
        if not self._is_recording or self._writer is None:
            return False

        # Check frame limits
        if (
            self.max_frames_per_episode
            and self._frame_count >= self.max_frames_per_episode
        ):
            logger.info(
                f"Reached max frames ({self.max_frames_per_episode}), stopping recording"
            )
            self.stop_recording()
            return False

        # Validate and convert frame
        if frame.dtype != np.uint8:
            frame = (
                (frame * 255).astype(np.uint8)
                if frame.max() <= 1.0
                else frame.astype(np.uint8)
            )

        # Ensure correct shape and color order (OpenCV uses BGR)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            logger.error(f"Invalid frame shape: {frame.shape}, expected (H, W, 3)")
            return False

        # Resize if needed
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != self.resolution:
            frame_bgr = cv2.resize(frame_bgr, self.resolution)

        # Queue frame for async writing
        try:
            self._frame_queue.put_nowait(frame_bgr.copy())
            self._frame_count += 1
            return True
        except Exception as e:
            logger.warning(f"Frame queue full or error, dropping frame: {e}")
            return False

    def stop_recording(self) -> Optional[str]:
        """
        Stop the current recording and finalize the video file.

        Returns:
            Path to the saved video file, or None if no recording was active
        """
        if not self._is_recording:
            return None

        self._is_recording = False

        # Stop async writer
        if self._write_thread and self._write_thread.is_alive():
            self._stop_writing.set()
            self._write_thread.join(timeout=5.0)

        # Close video writer
        if self._writer:
            self._writer.release()
            self._writer = None

        saved_file = self._current_filename
        if saved_file and os.path.exists(saved_file):
            # Check if video is empty (no frames or very few frames)
            if self._frame_count < 5:
                # Delete empty video file and return None to skip episode numbering
                try:
                    os.remove(saved_file)
                    logger.info(
                        f"Skipped empty episode: {Path(saved_file).name} "
                        f"({self._frame_count} frames - deleted)"
                    )
                    return None
                except Exception as e:
                    logger.warning(
                        f"Failed to delete empty video file {saved_file}: {e}"
                    )
                    return None
            else:
                # Video has content, save it normally
                file_size = os.path.getsize(saved_file) / (1024 * 1024)  # MB
                duration = self._frame_count / self.fps
                logger.info(
                    f"Video saved: {Path(saved_file).name} "
                    f"({self._frame_count} frames, {duration:.1f}s, {file_size:.1f}MB)"
                )
                return saved_file

        return None

    def _async_writer(self):
        """Background thread for writing frames to video file."""
        while not self._stop_writing.is_set():
            try:
                # Wait for frame with timeout
                frame = self._frame_queue.get(timeout=0.1)
                if self._writer and self._writer.isOpened():
                    self._writer.write(frame)
                self._frame_queue.task_done()
            except Exception:
                # Timeout or queue empty, continue
                continue

        # Flush remaining frames
        while not self._frame_queue.empty():
            try:
                frame = self._frame_queue.get_nowait()
                if self._writer and self._writer.isOpened():
                    self._writer.write(frame)
                self._frame_queue.task_done()
            except Exception:
                break

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording

    def get_stats(self) -> Dict[str, Any]:
        """Get recording statistics."""
        return {
            "is_recording": self._is_recording,
            "current_frame_count": self._frame_count,
            "episode_count": self._episode_count,
            "current_filename": self._current_filename,
            "queue_size": self._frame_queue.qsize(),
            "fps": self.fps,
            "resolution": self.resolution,
        }

    def cleanup(self):
        """Clean up resources."""
        if self._is_recording:
            self.stop_recording()

        # Ensure thread cleanup
        if self._write_thread and self._write_thread.is_alive():
            self._stop_writing.set()
            self._write_thread.join(timeout=2.0)

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


def create_video_recorder_from_config(
    output_dir: str, video_config: Dict[str, Any]
) -> VideoRecorder:
    """
    Create a VideoRecorder from configuration dictionary.

    Args:
        output_dir: Base output directory
        video_config: Video configuration dictionary

    Returns:
        Configured VideoRecorder instance
    """
    return VideoRecorder(
        output_dir=output_dir,
        fps=video_config.get(
            "fps"
        ),  # fps now optional - will be set via finalize_fps()
        resolution=tuple(video_config["resolution"]),
        codec=video_config["codec"],
        max_duration=video_config.get("maxDuration"),
        max_frames_per_episode=video_config.get("maxFramesPerEpisode"),
    )
