"""Which numpy SIMD path this machine runs, for byte-exact pins (no tolerance).

numpy's float64 exp/log (and float32 log1p) differ in the last bits between its X86_V4 (AVX512) and X86_V3 (AVX2) code
paths, and GitHub runners are a lottery of both. A byte-exact fingerprint of a synthetic build therefore has one pin per
path (same convention as tests/test_multi_tf.py::_data_golden). Other CPUs (ARM, ...) have no pin: the test is skipped.
"""
from numpy._core._multiarray_umath import __cpu_features__ as _cpu


def simd_path():
    """'x86_v4', 'x86_v3' or None (no recorded pin for this CPU)."""
    if _cpu.get("X86_V4"):
        return "x86_v4"
    if _cpu.get("X86_V3"):
        return "x86_v3"
    return None


def pick(pins):
    """pins: {'x86_v4': ..., 'x86_v3': ...} -> the pin for this machine, or None."""
    return pins.get(simd_path())
