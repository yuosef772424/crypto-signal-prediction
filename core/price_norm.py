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


# ── target_mode 'volret' (vol-normalised log return) and the input-window decode it needs ─────────────────────────────
def volret_forward(logret, vol):
    """``z = log(future / last) / vol``: the target of ``target_mode='volret'`` before the TargetSettings.clip. ``vol`` is the
    per-row past-only volatility (``stats.volatility.ewma_std`` of the window's close log returns, floored)."""
    return logret / vol


def volret_inverse(z, vol):
    """The ONE decode of a volret prediction: ``z * vol`` is the predicted log return of the row (``vol`` is the per-row
    ``target_vol`` stamped on the split by ``retarget_splits``)."""
    return z * vol


def input_close_logreturns(x_close, mode: str, *, center=None, scale=None, mean=None, std=None, clip_abs=None, log=None):
    """Past close log returns of each window, decoded from the STORED close channel ``x_close`` (n, T) of the base timeframe
    (known at entry, no future). Returns ``(r, ok)``: ``r`` (n, T-1) for 'window_scale' (``close = x * scale + center``, then
    ``diff(log close)``) or (n, T) for 'logret_std' (the channel already is the standardised log return: ``r = x * std + mean``);
    ``ok`` (n,) False for a row whose decode is unreliable (non-finite or non-positive closes, a zero scale, or an input clipped at
    ``clip_abs``). Any other ``mode`` ('logret_sigma': channel / a per-candle sigma; 'pct_change') raises ValueError - the window
    cannot be recovered. ``x_close`` / ``center`` / ``scale`` are float64 numpy arrays (center / scale shaped (n, 1)); ``log`` is
    the array log function the caller owns (numpy.log). Stdlib-only otherwise (tier 0)."""
    if mode == 'window_scale':
        if center is None or scale is None or log is None:
            raise ValueError("input_close_logreturns('window_scale') needs center, scale and log")
        closes = x_close * scale + center
        ok = (scale[:, 0] > 0) & (closes > 0).all(axis=1)
        if clip_abs is not None:
            ok = ok & (abs(x_close) < clip_abs * (1.0 - 1e-6)).all(axis=1)
        safe = closes * 0 + 1.0
        safe = (closes > 0) * closes + (closes <= 0) * safe
        r = log(safe[:, 1:] / safe[:, :-1])
        return r, ok
    if mode == 'logret_std':
        if mean is None or std is None:
            raise ValueError("input_close_logreturns('logret_std') needs mean and std of the close channel")
        return x_close * std + mean, (x_close == x_close).all(axis=1)
    raise ValueError(f"target_mode 'volret' cannot read the past closes of price_norm_mode={mode!r}: only 'window_scale' and "
                     "'logret_std' store them decodably ('logret_sigma' divides each return by a per-candle sigma; 'pct_change' "
                     "drops the level)")
