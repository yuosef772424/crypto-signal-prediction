"""
PURPOSE:  The ONE pair of formulas that maps a future price to the standardized log-return target space of price_norm_mode 'logret_*' and back: z = (log(price / anchor) - mean) / std. Used by data.normalizers (pipeline) and evaluation.decode (model evaluation), so training and evaluation can never disagree.
TAGS:     price_norm, logret, logret_forward, logret_inverse, denormalize, normalize targets, log return, anchor, last_close, core
PITFALLS: Pure numpy, float64, no clipping (the input channels are clipped by the pipeline, targets are not). `anchor` is last_close for every head in 'logret_*'. Do not duplicate this formula elsewhere: call it.
"""
import numpy as np


def logret_forward(price, mean: float, std: float, anchor) -> np.ndarray:
    """Future price -> target space: ``(log(price / anchor) - mean) / std``."""
    return (np.log(np.asarray(price, dtype='float64') / np.asarray(anchor, dtype='float64')) - mean) / std


def logret_inverse(pred, mean: float, std: float, anchor, output: str = 'price') -> np.ndarray:
    """Target space -> ``output``: ``'price'`` = ``anchor * exp(pred * std + mean)``; ``'logret'`` = ``pred * std + mean``."""
    if output not in ('price', 'logret'):
        raise ValueError(f"output: 'price' | 'logret' — لا {output!r}")
    lr = np.asarray(pred, dtype='float64') * std + mean
    return lr if output == 'logret' else np.asarray(anchor, dtype='float64') * np.exp(lr)
