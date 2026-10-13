"""tests/test_signal_causality.py: look-ahead audit of the signal pipelines (tools/trade_engine.py and the
strategy-discovery grids). For each function group the pipeline is run on the full random-walk frame and on prefixes
(cut points t); every value at bar index <= t must be identical. A second check replaces the bars after t with other
random bars and requires the same values at and before t.
    python -m unittest tests.test_signal_causality -v
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "research", "strategy_discovery"))
import trade_engine as E  # noqa: E402
import grid as G  # noqa: E402
import grid2 as G2  # noqa: E402

N = 800                      # bars in the full frame (>= 600)
CUTS = (300, 400, 500, 650)  # cut points t: the prefix frame is bars 0..t inclusive
GRID_GATES = ["none", "adx25", "ema200_aligned"]


def _ohlcv(n, seed):
    """Reproducible random-walk OHLCV with a UTC hourly index; high/low bracket open and close, volume > 0."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0.0, 0.002, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.006, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.006, n)))
    volume = rng.lognormal(3.0, 0.5, n)
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def _mismatch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype == bool or b.dtype == bool:
        return a.astype(bool) != b.astype(bool)
    a, b = a.astype(float), b.astype(float)
    return ~((a == b) | (np.isnan(a) & np.isnan(b)))


def _as_arrays(d):
    return {c: d[c].to_numpy() for c in d.columns}


class _CausalityMixin:
    def _check(self, name, pipeline, df, min_fired=1):
        """pipeline(frame) -> dict key -> array (one entry per bar). Compares prefixes and the post-cut perturbation."""
        other = _ohlcv(len(df), seed=987654)
        full = pipeline(df)
        fired = sum(int(np.sum(v)) for v in full.values() if np.asarray(v).dtype == bool)
        self.assertGreaterEqual(fired, min_fired, f"{name}: no boolean event fires on the full frame (test is vacuous)")
        failures = []
        for t in CUTS:
            prefix = pipeline(df.iloc[:t + 1].copy())
            alt = df.copy()
            alt.iloc[t + 1:] = other.iloc[t + 1:].to_numpy()
            perturbed = pipeline(alt)
            for label, out in (("truncated frame", prefix), ("bars after t replaced", perturbed)):
                for key, ref in full.items():
                    a = np.asarray(ref)[:t + 1]
                    b = np.asarray(out[key])[:t + 1]
                    bad = _mismatch(a, b)
                    if bad.any():
                        i = int(np.argmax(bad))
                        failures.append(f"{name}[{key}] {label}, cut t={t}: first changed bar index {i} "
                                        f"(full={a[i]!r}, changed={b[i]!r})")
        self.assertFalse(failures, "look-ahead detected:\n" + "\n".join(failures[:20]))


class SignalCausalityTests(_CausalityMixin, unittest.TestCase):
    def setUp(self):
        self.df = _ohlcv(N, seed=20260101)
        self.d_ind = E.indicators(self.df)
        self.gate_names = sorted(E.gates(self.d_ind).keys()) + ["long_only"]
        self.trigger_names = sorted(E.triggers(self.d_ind).keys())

    def test_trade_engine_indicators_are_causal(self):
        self._check("trade_engine.indicators", lambda f: _as_arrays(E.indicators(f)), self.df, min_fired=0)

    def test_trade_engine_triggers_are_causal(self):
        def pipe(f):
            d = E.indicators(f)
            out = {}
            for name, (lg, sh) in E.triggers(d).items():
                out[f"{name}.long"], out[f"{name}.short"] = lg, sh
            return out
        self._check("trade_engine.triggers", pipe, self.df)

    def test_trade_engine_build_signal_is_causal_for_every_trigger_and_gate(self):
        def pipe(f):
            d = E.indicators(f)
            out = {}
            for trig in self.trigger_names:
                for gate in self.gate_names:
                    lg, sh = E.build_signal(d, trig, (gate,))
                    out[f"{trig}|{gate}.long"], out[f"{trig}|{gate}.short"] = lg, sh
                lg, sh = E.build_signal(d, trig, ())
                out[f"{trig}|none.long"], out[f"{trig}|none.short"] = lg, sh
            return out
        self._check("trade_engine.build_signal", pipe, self.df, min_fired=1)

    def test_grid_extra_columns_are_causal(self):
        self._check("grid.extra_columns", lambda f: _as_arrays(G.extra_columns(E.indicators(f))), self.df, min_fired=0)

    def test_grid_signal_is_causal_for_every_family(self):
        def pipe(f):
            d = G.extra_columns(E.indicators(f))
            out = {}
            for fam in G.FAMILIES:
                lg, sh = G.signal(d, fam)
                out[f"{fam}.long"], out[f"{fam}.short"] = lg, sh
            return out
        self._check("grid.signal", pipe, self.df)

    def test_grid_apply_gate_is_causal_for_every_gate(self):
        def pipe(f):
            d = G.extra_columns(E.indicators(f))
            out = {}
            for fam in G.FAMILIES:
                lg, sh = G.signal(d, fam)
                for gate in GRID_GATES:
                    gl, gs = G.apply_gate(d, lg, sh, gate)
                    out[f"{fam}|{gate}.long"], out[f"{fam}|{gate}.short"] = gl, gs
            return out
        self._check("grid.apply_gate", pipe, self.df)

    def test_grid2_extra_is_causal(self):
        self._check("grid2.extra", lambda f: _as_arrays(G2.extra(E.indicators(f))), self.df, min_fired=0)

    def test_grid2_supertrend_is_causal(self):
        def pipe(f):
            d = G2.extra(E.indicators(f))
            trend, up, dn = G2.supertrend(d)
            return {"trend": trend.to_numpy(), "flip_up": up, "flip_dn": dn}
        self._check("grid2.supertrend", pipe, self.df)

    def test_grid2_signal2_is_causal_for_keltner_squeeze_volume_donchian_and_supertrend(self):
        def pipe(f):
            d = G2.extra(E.indicators(f))
            out = {}
            for fam in G2.FAMILIES2:
                lg, sh = G2.signal2(d, fam)
                out[f"{fam}.long"], out[f"{fam}.short"] = lg, sh
            return out
        self._check("grid2.signal2", pipe, self.df)


class HarnessSelfCheckTests(_CausalityMixin, unittest.TestCase):
    """The checker itself must flag a deliberately non-causal pipeline (guards against a vacuous PASS)."""

    def test_checker_flags_an_injected_forward_shift(self):
        df = _ohlcv(N, seed=20260101)

        def leaky(f):
            return {"fwd_close": f.close.shift(-1).to_numpy()}

        with self.assertRaises(AssertionError) as ctx:
            self._check("injected_leak", leaky, df, min_fired=0)
        self.assertIn("first changed bar index", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
