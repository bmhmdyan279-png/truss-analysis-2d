"""Streaming (running) statistics for memory-bounded Monte Carlo.

With limited RAM a large study must never hold all samples' results at
once; batches of samples are folded into Welford accumulators instead, so
memory use stays constant in the number of samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["RunningStat"]


@dataclass
class RunningStat:
    """Welford batch accumulator: count, mean, variance (sample, ddof=1)."""

    count: int = 0
    mean: float = 0.0
    _m2: float = 0.0

    def add(self, x: float) -> None:
        """Fold a single scalar observation into the accumulator."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self._m2 += delta * (x - self.mean)

    def add_batch(self, values: np.ndarray) -> None:
        """Fold a whole batch in one Chan-style update."""
        arr = np.asarray(values, dtype=float).ravel()
        m = arr.size
        if m == 0:
            return
        new_mean = float(arr.mean())
        new_m2 = float(((arr - new_mean) ** 2).sum())
        if self.count == 0:
            self.count = m
            self.mean = new_mean
            self._m2 = new_m2
            return
        total = self.count + m
        delta = new_mean - self.mean
        self._m2 += new_m2 + delta**2 * self.count * m / total
        self.mean += delta * m / total
        self.count = total

    @property
    def variance(self) -> float:
        """Return the sample variance (ddof=1); 0.0 for fewer than 2 points."""
        if self.count < 2:
            return 0.0
        return self._m2 / (self.count - 1)

    @property
    def std(self) -> float:
        """Return the sample standard deviation (ddof=1)."""
        return float(np.sqrt(self.variance))

    def as_dict(self) -> dict[str, float]:
        """Return ``{"count", "mean", "std"}`` as plain floats."""
        return {"count": self.count, "mean": self.mean, "std": self.std}
