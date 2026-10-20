"""
PURPOSE:  The ONE pair of formulas that maps a future price to the standardized log-return target space of price_norm_mode 'logret_*' and back: z = (log(price / anchor) - mean) / std. Used by data.normalizers (pipeline) and evaluation.decode (model evaluation), so training and evaluation can never disagree.
TAGS:     price_norm, logret, logret_forward, logret_inverse, denormalize, normalize targets, log return, anchor, last_close, core
PITFALLS: Stdlib-only (tier 0): array-agnostic arithmetic, the caller passes float64 arrays and the log/exp function (numpy.log / numpy.exp), so results are byte-identical to the old numpy version. No clipping (the input channels are clipped by the pipeline, targets are not). `anchor` is last_close for every head in 'logret_*'. Do not duplicate this formula elsewhere: call it.
"""


def logret_forward(price, mean: float, std: float, anchor, log):
    """Future price -> target space: ``(log(price / anchor) - mean) / std``. ``price`` / ``anchor`` are float64 arrays
    (the caller converts); ``log`` is the array log function the caller owns (e.g. ``numpy.log``)."""
    return (log(price / anchor) - mean) / std


def logret_inverse(pred, mean: float, std: float, anchor, exp, output: str = 'price'):
    """Target space -> ``output``: ``'price'`` = ``anchor * exp(pred * std + mean)``; ``'logret'`` = ``pred * std + mean``.
    ``pred`` / ``anchor`` are float64 arrays (the caller converts); ``exp`` is the array exp function (e.g. ``numpy.exp``)."""
    if output not in ('price', 'logret'):
        raise ValueError(f"output: 'price' | 'logret' — لا {output!r}")
    lr = pred * std + mean
    return lr if output == 'logret' else anchor * exp(lr)
