"""
Memory monitoring utilities for debugging performance issues.
"""

import torch
import gc
from loguru import logger


class MemoryMonitor:
    """
    Monitor GPU memory usage and tensor allocation patterns.

    This helps identify memory leaks and performance degradation issues.
    """

    def __init__(self, device="cuda:0", log_interval=1000):
        """
        Initialize memory monitor.

        Args:
            device: Device to monitor
            log_interval: How often to log memory stats (in steps)
        """
        self.device = device
        self.log_interval = log_interval
        self.step_count = 0
        self.initial_memory = None

        # Track memory history
        self.memory_history = []
        self.tensor_count_history = []

    def step(self):
        """Call this each training step to monitor memory."""
        self.step_count += 1

        # Log memory stats at intervals
        if self.step_count % self.log_interval == 0:
            self.log_memory_stats()

    def log_memory_stats(self):
        """Log current memory statistics."""
        if not torch.cuda.is_available():
            return

        # Get current memory usage
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3  # GB

        # Store initial memory if first call
        if self.initial_memory is None:
            self.initial_memory = allocated

        # Calculate memory growth
        memory_growth = allocated - self.initial_memory

        # Count tensors (expensive, so do sparingly)
        tensor_count = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    tensor_count += 1
            except Exception:
                pass

        # Store history
        self.memory_history.append(allocated)
        self.tensor_count_history.append(tensor_count)

        # Log stats
        logger.info(
            f"Memory Stats - Step: {self.step_count}, "
            f"Allocated: {allocated:.2f}GB, "
            f"Reserved: {reserved:.2f}GB, "
            f"Growth: {memory_growth:.2f}GB, "
            f"Tensors: {tensor_count}"
        )

        # Warn if memory is growing significantly
        if memory_growth > 0.5:  # More than 500MB growth
            logger.warning(
                f"Significant memory growth detected: {memory_growth:.2f}GB "
                f"since start. This may indicate a memory leak."
            )

    def get_summary(self):
        """Get summary of memory usage over time."""
        if not self.memory_history:
            return "No memory history available"

        initial = self.memory_history[0]
        final = self.memory_history[-1]
        peak = max(self.memory_history)

        return (
            f"Memory Summary:\n"
            f"  Initial: {initial:.2f}GB\n"
            f"  Final: {final:.2f}GB\n"
            f"  Peak: {peak:.2f}GB\n"
            f"  Total Growth: {final - initial:.2f}GB\n"
            f"  Tensor Count Growth: {self.tensor_count_history[-1] - self.tensor_count_history[0]}"
        )
