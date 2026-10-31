"""
PURPOSE:  The ONE `seed_everything(seed)`: seeds python `random`, numpy and TensorFlow/Keras from a single integer (the run's `TrainSettings.seed` -> `config["run"]["seed"]`); every other place gets its seed from there instead of a literal.
TAGS:     seed, seed_everything, set_random_seed, reproducibility, batch order, run identity, tier 0
PITFALLS: numpy and TensorFlow are imported INSIDE the function (core stays stdlib-only, no import-time work); a package that is not installed is skipped. Call it BEFORE building the training dataset: a tf.data shuffle created before the global seed is set is not reproducible (audit 2026-10-31, C3). Calling it again later (build_training_system does, before the model is built) re-seeds the global state, so the model init depends only on the seed, not on how much randomness the data step consumed. The seed must be a non-negative int below 2**32 (numpy's limit); a bool, float or negative raises.
"""
from __future__ import annotations

import random

__all__ = ["seed_everything", "SEED_MAX"]

SEED_MAX = 2 ** 32


def seed_everything(seed: int) -> int:
    """Seed python, numpy and TensorFlow/Keras with ``seed`` and return it."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < SEED_MAX:
        raise ValueError(f"seed must be an int in [0, {SEED_MAX}), got {seed!r}")
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)
    try:
        import tensorflow as tf
    except ImportError:
        return seed
    tf.keras.utils.set_random_seed(seed)   # python + numpy + tf in one call; the three above stay for the no-TF case
    return seed
