"""
HTTP video streaming component for Isaac Gym environments.

This module provides HTTP video streaming functionality that serves captured frames
in real-time over HTTP, allowing remote viewing of environment execution.
"""

import threading
import time
from queue import Queue, Empty
from typing import Optional, Dict, Any

import cv2
import numpy as np
from loguru import logger

try:
    from flask import Flask, Response, jsonify, render_template_string

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logger.warning("Flask not available - HTTP streaming disabled")


class HTTPVideoStreamer:
    """
    HTTP video streamer that serves RGB frames over HTTP in real-time.

    Features:
    - Real-time MJPEG streaming over HTTP
    - Configurable quality and frame rate
    - Web interface for viewing streams
    - Statistics and monitoring endpoints
    - Thread-safe frame buffering
    """

    def __init__(
        self,
        host: str,
        port: int,
        quality: int,
        buffer_size: int,
        bind_all: bool = False,
    ):
        """
        Initialize the HTTP video streamer.

        All parameters are required - no defaults provided.
        Use create_http_video_streamer_from_config() for config-based initialization.

        Args:
            host: Host address to bind server to
            port: Port to serve on
            quality: JPEG quality (1-100)
            buffer_size: Maximum frames to buffer (kept at 1 for latest frame only)
            bind_all: If True, bind to all interfaces (0.0.0.0) instead of configured host
        """
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for HTTP video streaming")

        # Store original values for reference
        self.original_host = host
        self.original_port = port

        # Set actual host based on bind_all flag - SINGLE SOURCE OF TRUTH
        self.host = "0.0.0.0" if bind_all else host
        self.port = port
        self.quality = quality
        self.buffer_size = buffer_size
        self.bind_all = bind_all

        # Frame management - use queue for non-blocking streaming
        self._frame_queue: Queue = Queue(maxsize=1)  # Only keep latest frame
        self._current_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        # Create placeholder frame for startup - delayed initialization to avoid OpenCV conflicts
        self._placeholder_frame = None

        # Statistics
        self._stats = {
            "frames_received": 0,
            "frames_served": 0,
            "frames_dropped": 0,
            "clients_connected": 0,
            "start_time": time.time(),
        }
        self._stats_lock = threading.Lock()

        # Flask app
        self._app = Flask(__name__)
        self._setup_routes()

        # Server thread
        self._server_thread: Optional[threading.Thread] = None
        self._server_running = False

        if bind_all:
            logger.info(
                f"HTTPVideoStreamer initialized: {self.host}:{port} (ALL interfaces), quality={quality}"
            )
            logger.warning(
                "Binding to all interfaces (0.0.0.0) - ensure firewall is configured!"
            )
        else:
            logger.info(
                f"HTTPVideoStreamer initialized: {self.host}:{port}, quality={quality}"
            )

    def _create_placeholder_frame(self) -> np.ndarray:
        """Create a placeholder frame for when no real frames are available."""
        # Create a black frame with "Waiting for frames..." text
        height, width = 480, 640  # Default size
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Add text overlay
        text = "Waiting for frames..."
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color = (255, 255, 255)  # White text
        thickness = 2

        # Get text size to center it
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2

        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

        # Convert RGB to BGR for OpenCV
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def _setup_routes(self):
        """Set up Flask routes for the streaming server."""

        @self._app.route("/")
        def index():
            """Serve full-screen video viewer."""
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Isaac Gym Video Stream</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    html, body {
                        height: 100%;
                        background: #000;
                        overflow: hidden;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    }
                    .video-stream {
                        max-width: 100vw;
                        max-height: 100vh;
                        width: auto;
                        height: auto;
                        object-fit: contain;
                    }
                </style>
            </head>
            <body>
                <img src="/stream" class="video-stream" alt="Video Stream">
            </body>
            </html>
            """
            return render_template_string(html_template)

        @self._app.route("/stats-page")
        def stats_page():
            """Serve the statistics monitoring page."""
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Isaac Gym Video Stream - Statistics</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
                    .container { max-width: 800px; margin: 0 auto; }
                    .header { text-align: center; margin-bottom: 30px; }
                    .stats { background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    .stats h3 { margin-top: 0; color: #333; }
                    .stat-item { margin: 8px 0; padding: 5px 0; border-bottom: 1px solid #eee; }
                    .stat-label { font-weight: bold; color: #555; }
                    .stat-value { float: right; color: #007bff; }
                    .refresh-btn {
                        background: #007bff; color: white; border: none; padding: 12px 24px;
                        border-radius: 4px; cursor: pointer; margin: 10px 5px; font-size: 14px;
                    }
                    .refresh-btn:hover { background: #0056b3; }
                    .video-link {
                        background: #28a745; color: white; border: none; padding: 12px 24px;
                        border-radius: 4px; cursor: pointer; margin: 10px 5px; font-size: 14px; text-decoration: none; display: inline-block;
                    }
                    .video-link:hover { background: #1e7e34; color: white; text-decoration: none; }
                    .button-group { text-align: center; margin: 20px 0; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Isaac Gym Video Stream Statistics</h1>
                        <div class="button-group">
                            <a href="/" class="video-link">📹 Back to Video</a>
                            <button class="refresh-btn" onclick="refreshStats()">🔄 Refresh Stats</button>
                        </div>
                    </div>
                    <div class="stats">
                        <h3>📊 Stream Statistics</h3>
                        <div id="stats-content">Loading...</div>
                    </div>
                </div>

                <script>
                    function refreshStats() {
                        fetch('/stats')
                            .then(response => response.json())
                            .then(data => {
                                const statsDiv = document.getElementById('stats-content');
                                statsDiv.innerHTML = `
                                    <div class="stat-item"><span class="stat-label">Frames Received:</span><span class="stat-value">${data.frames_received}</span></div>
                                    <div class="stat-item"><span class="stat-label">Frames Served:</span><span class="stat-value">${data.frames_served}</span></div>
                                    <div class="stat-item"><span class="stat-label">Frames Dropped:</span><span class="stat-value">${data.frames_dropped}</span></div>
                                    <div class="stat-item"><span class="stat-label">Clients Connected:</span><span class="stat-value">${data.clients_connected}</span></div>
                                    <div class="stat-item"><span class="stat-label">Uptime:</span><span class="stat-value">${data.uptime_seconds.toFixed(1)}s</span></div>
                                    <div class="stat-item"><span class="stat-label">Current FPS:</span><span class="stat-value">${data.current_fps.toFixed(1)}</span></div>
                                    <div class="stat-item"><span class="stat-label">Server Host:</span><span class="stat-value">${data.host || 'N/A'}</span></div>
                                    <div class="stat-item"><span class="stat-label">Server Port:</span><span class="stat-value">${data.port || 'N/A'}</span></div>
                                `;
                            })
                            .catch(error => {
                                console.error('Error fetching stats:', error);
                                document.getElementById('stats-content').innerHTML = '<div class="stat-item" style="color: red;">Error loading statistics</div>';
                            });
                    }

                    // Auto-refresh stats every 2 seconds
                    setInterval(refreshStats, 2000);
                    refreshStats(); // Initial load
                </script>
            </body>
            </html>
            """
            return render_template_string(html_template)

        @self._app.route("/stream")
        def stream():
            """Serve the MJPEG video stream."""
            with self._stats_lock:
                self._stats["clients_connected"] += 1

            try:
                return Response(
                    self._generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                )
            finally:
                with self._stats_lock:
                    self._stats["clients_connected"] -= 1

        @self._app.route("/stats")
        def stats():
            """Return streaming statistics as JSON."""
            with self._stats_lock:
                current_time = time.time()
                uptime = current_time - self._stats["start_time"]

                # Calculate current FPS
                frames_per_second = self._stats["frames_served"] / max(uptime, 1.0)

                stats_copy = self._stats.copy()
                stats_copy["uptime_seconds"] = uptime
                stats_copy["current_fps"] = frames_per_second

            return jsonify(stats_copy)

    def _generate_frames(self):
        """Generate MJPEG frames for HTTP streaming using non-blocking queue."""
        while self._server_running:
            try:
                # Try to get a frame from the queue with timeout
                frame = self._frame_queue.get(timeout=0.1)
            except Empty:
                # No frame available, use placeholder (lazy initialization)
                if self._placeholder_frame is None:
                    self._placeholder_frame = self._create_placeholder_frame()
                frame = self._placeholder_frame

            # Encode frame as JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            success, buffer = cv2.imencode(".jpg", frame, encode_params)

            if not success:
                logger.warning("Failed to encode frame as JPEG")
                # Use placeholder frame if encoding fails (lazy initialization)
                if self._placeholder_frame is None:
                    self._placeholder_frame = self._create_placeholder_frame()
                success, buffer = cv2.imencode(
                    ".jpg", self._placeholder_frame, encode_params
                )
                if not success:
                    continue

            # Convert to bytes
            frame_bytes = buffer.tobytes()

            # Update statistics
            with self._stats_lock:
                self._stats["frames_served"] += 1

            # Yield frame in MJPEG format
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

            # No artificial frame rate control - serve frames as fast as they arrive

    def start_server(self) -> bool:
        """
        Start the HTTP streaming server with automatic port conflict resolution.

        Returns:
            True if server started successfully
        """
        if self._server_running:
            logger.warning("Server already running")
            return False

        # Try to find available port starting from configured port
        import socket

        max_attempts = 10
        current_port = self.port

        for attempt in range(max_attempts):
            # First test if port is available by trying to bind to it
            test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                test_sock.bind((self.host, current_port))
                test_sock.close()
                # Port is available, start Flask server
                break
            except OSError as e:
                test_sock.close()
                if attempt < max_attempts - 1:
                    logger.debug(
                        f"Port {current_port} unavailable ({e}), trying {current_port + 1}"
                    )
                    current_port += 1
                    continue
                else:
                    logger.error(
                        f"No available port found after {max_attempts} attempts (tried {self.original_port}-{current_port})"
                    )
                    return False

        # Found available port, now start Flask server
        try:

            def run_server():
                """Run Flask server in thread."""
                try:
                    self._app.run(
                        host=self.host,
                        port=current_port,
                        debug=False,
                        use_reloader=False,
                        threaded=True,
                    )
                except Exception as e:
                    logger.error(f"Server error on {self.host}:{current_port}: {e}")

            self._server_thread = threading.Thread(target=run_server, daemon=True)
            self._server_thread.start()
            self._server_running = True

            # Update port if it changed
            if current_port != self.original_port:
                logger.info(
                    f"Port {self.original_port} was occupied, using port {current_port}"
                )
                self.port = current_port

            # Give server time to start
            time.sleep(1.0)

            logger.info(
                f"HTTP video stream server started at http://{self.host}:{current_port}"
            )
            return True

        except Exception as e:
            logger.error(f"Unexpected error starting server: {e}")
            self._server_running = False
            return False

        return False

    def stop_server(self):
        """Stop the HTTP streaming server."""
        if not self._server_running:
            return

        self._server_running = False

        # Note: Flask development server doesn't have a clean shutdown method
        # In production, you'd use a proper WSGI server like Gunicorn
        logger.info("HTTP video stream server stopped")

    def add_frame(self, frame: np.ndarray) -> bool:
        """
        Add a frame to the stream.

        Args:
            frame: RGB frame as numpy array (H, W, 3) with values 0-255

        Returns:
            True if frame was added successfully
        """
        if not self._server_running:
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

        # Always accept new frames - no rate limiting
        # Try to put frame in queue (non-blocking, keep only latest frame)
        try:
            # Remove old frame if queue is full
            try:
                self._frame_queue.get_nowait()
            except Empty:
                pass
            # Add new frame
            self._frame_queue.put_nowait(frame_bgr)

            # Also update _current_frame for backward compatibility
            with self._frame_lock:
                self._current_frame = frame_bgr

        except Exception:
            # Queue operations failed, count as dropped
            with self._stats_lock:
                self._stats["frames_dropped"] += 1

        # Always count frames as received when they're valid
        with self._stats_lock:
            self._stats["frames_received"] += 1

        return True

    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self._server_running

    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        with self._stats_lock:
            current_time = time.time()
            uptime = current_time - self._stats["start_time"]

            stats_copy = self._stats.copy()
            stats_copy["uptime_seconds"] = uptime
            stats_copy["current_fps"] = self._stats["frames_served"] / max(uptime, 1.0)
            stats_copy["server_running"] = self._server_running
            stats_copy["host"] = self.host
            stats_copy["port"] = self.port
            stats_copy["original_host"] = self.original_host
            stats_copy["original_port"] = self.original_port
            stats_copy["bind_all"] = self.bind_all

        return stats_copy

    def cleanup(self):
        """Clean up resources."""
        self.stop_server()

        # Clear frame queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Empty:
                break

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


def create_http_video_streamer_from_config(
    video_config: Dict[str, Any]
) -> HTTPVideoStreamer:
    """
    Create an HTTPVideoStreamer from configuration dictionary.

    All configuration values must be present - no fallback defaults.
    This ensures config file is the single source of truth.

    Args:
        video_config: Video configuration dictionary with required keys:
                     stream_host, stream_port, stream_quality, stream_buffer_size, stream_bind_all

    Returns:
        Configured HTTPVideoStreamer instance

    Raises:
        KeyError: If required configuration keys are missing
    """
    # Validate all required config keys are present
    required_keys = [
        "stream_host",
        "stream_port",
        "stream_quality",
        "stream_buffer_size",
        "stream_bind_all",
    ]
    missing_keys = [key for key in required_keys if key not in video_config]

    if missing_keys:
        raise KeyError(
            f"Missing required HTTP streaming config keys: {missing_keys}. "
            f"Check base/video.yaml configuration file."
        )

    return HTTPVideoStreamer(
        host=video_config["stream_host"],
        port=video_config["stream_port"],
        quality=video_config["stream_quality"],
        buffer_size=video_config["stream_buffer_size"],
        bind_all=video_config["stream_bind_all"],
    )
