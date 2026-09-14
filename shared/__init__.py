"""Shared utilities for every project in classical-computer-vision.

The single rule that governs this package: **images are RGB uint8 everywhere**.
OpenCV loads BGR, matplotlib expects RGB, and mixing them is the most common bug
in computer vision. It is fixed once, here, and never thought about again.
"""

from . import bench, figures, io, metrics, report, synth

__all__ = ["bench", "figures", "io", "metrics", "report", "synth"]
