"""
PURPOSE:  The ONE `seed_everything(seed)`: seeds python `random`, numpy and TensorFlow/Keras from a single integer (the run's `TrainSettings.seed` -> `config["run"]["seed"]`); every other place gets its seed from there instead of a literal.
TAGS:     seed, seed_everything, set_random_seed, reproducibility, batch order, run identity, tier 0
PITFALLS: core imports no third-party package (tier 0): the framework seeder is injected by the caller (`framework_seed_fn=tf.keras.utils.set_random_seed`, required keyword, no default). Call it BEFORE building the training dataset: a tf.data shuffle created before the global seed is set is not reproducible (audit 2026-10-31, C3). Calling it again later (build_training_system does, before the model is built) re-seeds the global state, so the model init depends only on the seed, not on how much randomness the data step consumed. The seed must be a non-negative int below 2**32 (numpy's limit); a bool, float or negative raises.
"""
from __future__ import annotations

import random
from typing import Callable

__all__ = ["seed_everything", "SEED_MAX"]

SEED_MAX = 2 ** 32


def seed_everything(seed: int, *, framework_seed_fn: Callable[[int], object]) -> int:
    """Validate ``seed``, seed python ``random`` and call ``framework_seed_fn(seed)``; return the seed.

    ``framework_seed_fn`` is REQUIRED (no silent default): the training code passes ``tf.keras.utils.set_random_seed``, which
    seeds python, numpy and TensorFlow in one call. Injecting it keeps ``core`` stdlib-only (tier 0, tests/test_core.py).
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < SEED_MAX:
        raise ValueError(f"seed must be an int in [0, {SEED_MAX}), got {seed!r}")
    if not callable(framework_seed_fn):
        raise TypeError(f"framework_seed_fn must be callable, got {framework_seed_fn!r}")
    random.seed(seed)
    framework_seed_fn(seed)
    return seed
