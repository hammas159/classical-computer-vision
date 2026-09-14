"""Timing harness.

Why timing is a headline result
-------------------------------
The entire argument for classical computer vision is "no training, no GPU,
milliseconds". A comparison table without a time column throws away the main
finding — a method that wins by 0.3 dB while running 100x slower has not won.

Why a single run is not a measurement
-------------------------------------
The first call to an OpenCV routine pays for lazy library initialisation, thread
pool spin-up and a cold cache. Timing it once and quoting the number routinely
overstates cost by an order of magnitude. This module discards warm-up runs and
reports the **median** of the rest, which is robust to the occasional scheduler
hiccup in a way the mean is not.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass
class Timing:
    """The result of timing one callable."""

    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    runs: int
    warmup: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.median_ms:.2f} ms (median of {self.runs})"


def timeit(
    fn: Callable[[], Any], runs: int = 7, warmup: int = 2
) -> tuple[Any, Timing]:
    """Run ``fn`` ``warmup + runs`` times; return its last result and the timing.

    The result is returned alongside the timing so a project never has to call
    the method twice — once to time it and once to keep the output.
    """
    if runs < 1:
        raise ValueError("runs must be >= 1")

    for _ in range(warmup):
        fn()

    samples: list[float] = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    n = len(samples)
    median = samples[n // 2] if n % 2 else 0.5 * (samples[n // 2 - 1] + samples[n // 2])
    return result, Timing(
        median_ms=float(median),
        mean_ms=float(sum(samples) / n),
        min_ms=float(samples[0]),
        max_ms=float(samples[-1]),
        runs=runs,
        warmup=warmup,
    )


def run_methods(
    methods: dict[str, Callable[[], Any]], runs: int = 7, warmup: int = 2
) -> dict[str, tuple[Any, Timing]]:
    """Time a whole dict of named methods under identical conditions.

    Using one helper for every method in a comparison is deliberate: it
    guarantees the same warm-up count and the same run count, so the time column
    compares like with like.
    """
    return {name: timeit(fn, runs=runs, warmup=warmup) for name, fn in methods.items()}
