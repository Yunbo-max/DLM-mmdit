"""
Efficiency metrics — Pillar 6 of DLM-Eval Suite.

Measures: Wall-clock time, NFE (number of forward evaluations), tokens/sec.
"""

import time

import torch


class EfficiencyTracker:
    """
    Context manager and utility for tracking computational efficiency.

    Usage:
        tracker = EfficiencyTracker()
        with tracker:
            results = sampler.edit(...)
        print(tracker.summary())
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.nfe = 0
        self.total_tokens = 0
        self.peak_gpu_mem = 0.0

    def __enter__(self):
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.end_time = time.perf_counter()
        if torch.cuda.is_available():
            self.peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

    @property
    def wall_clock(self):
        """Wall-clock time in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    @property
    def tokens_per_second(self):
        """Tokens processed per second."""
        if self.wall_clock == 0:
            return 0.0
        return self.total_tokens / self.wall_clock

    def record_step(self, num_tokens=0):
        """Record one forward evaluation."""
        self.nfe += 1
        self.total_tokens += num_tokens

    def summary(self):
        """
        Returns:
            dict with "wall_clock_s", "nfe", "tokens_per_second", "peak_gpu_mem_gb"
        """
        return {
            "wall_clock_s": self.wall_clock,
            "nfe": self.nfe,
            "tokens_per_second": self.tokens_per_second,
            "peak_gpu_mem_gb": self.peak_gpu_mem,
        }


def profile_method(fn, *args, n_runs=3, warmup=1, **kwargs):
    """
    Profile a method over multiple runs and return efficiency stats.

    Args:
        fn: callable to profile
        *args: positional args for fn
        n_runs: int, number of timed runs
        warmup: int, warmup runs (not timed)
        **kwargs: keyword args for fn

    Returns:
        dict with timing stats
    """
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        fn(*args, **kwargs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append(time.perf_counter() - start)

    import numpy as np
    return {
        "mean_time_s": float(np.mean(times)),
        "std_time_s": float(np.std(times)),
        "min_time_s": float(np.min(times)),
        "max_time_s": float(np.max(times)),
    }
