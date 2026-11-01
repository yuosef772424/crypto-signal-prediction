"""
PURPOSE:  The 'level side-channel' (default OFF): per-bar LEVEL features that the per-window normalization of the pipeline destroys (realized volatility level, window return and distances to the window extremes in volatility units, dollar-volume level vs a long trailing mean, vol regime vs the coin's own history), stored WITHOUT per-window scaling and standardized with statistics fitted on TRAIN only. ONE registered component (LEVEL_CHANNELS['level_v1']) with the owner-rule interface: add_features (raw columns) -> fit (train only) -> state -> transform_inputs / stamp / prepare_live; its state is stored on every split as split['level_meta'] and is all the live path needs.
TAGS:     level channel, level side-channel, LVL_, LEVEL_CHANNELS, LevelChannel, level_v1, LEVEL_FEATURES, level_channel_settings, level_channel_columns, add_level_features, finish_level_channel, fit_level_channel, volatility level, realized vol, dollar volume z, distance to high low, train-only fit, per-window normalization, F-0029, data prep audit
PITFALLS: Off unless CONFIG['level_channel'] is a dict (None = nothing changes, byte-identical datasets/splits; the key is not hashed anywhere else). Columns are named LVL_<feature>; their kind LEVEL_RAW in normalize.py is x/10 with NO per-window centering or scaling and is never zeroed when constant; the dataset then holds RAW-scaled values (dataset['level_channel']['state'] == 'unfitted') until split_data / split_holdout / build_dataset_live apply the train-fitted state (clip +-clip after (x - mean) / std per tf and column; mean/std from the train samples' last window step = the entry-time value). Never feed an unfitted dataset to a model. Each feature at bar t uses bars <= t only (rolling windows ending at t, no centering, no shift(-k)); the first long_window bars of a coin are dropped by add_features' dropna. window / long_window count bars of EACH timeframe's own frame (like every other indicator length). infer_feature_columns derives the columns from a 600-row synthetic frame, so long_window <= 500 (checked). A feature name outside LEVEL_FEATURES, an unknown settings key or an unregistered component raises. The model sees these columns through its normal inputs; whether it can USE the level depends on the model keeping it (level_passthrough / stats path) - this module only stops the DATA from erasing it. Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.
"""
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from core.registry import Contract as _Contract, Registry as _Registry

#: prefix of every level column (normalize.py classifies it as LEVEL_RAW by this prefix).
LEVEL_PREFIX = 'LVL_'
#: stored scale of a raw level column: x / LEVEL_RAW_SCALE, a constant (not a learned state) that keeps every raw value inside the global
#: +-clip_abs of process_windows (log vol is about -6, a window-return in vol units a few units); the train-fitted state is fitted on the
#: stored values, so this constant never needs to be undone by a consumer.
LEVEL_RAW_SCALE = 10.0
_SIGMA_FLOOR = 1e-6
_EPS = 1e-12
#: infer_feature_columns builds the column list from a 600-row synthetic frame: the longest trailing window must leave rows.
LEVEL_MAX_LONG_WINDOW = 500

#: the settings a CONFIG['level_channel'] dict may carry (anything else raises) and what an omitted key means (explicit, in one place).
LEVEL_SETTINGS_DEFAULTS: Dict[str, Any] = {
    'component': 'level_v1',
    'features': ['logvol', 'retvol', 'disthi', 'distlo', 'logdvol_z', 'volreg'],
    'window': 32,
    'long_window': 256,
    'clip': 5.0,
}


# ══════════════════════════════════════════════════════════════════════════
# the feature registry (name -> function of the OHLCV frame and the two windows)
# ══════════════════════════════════════════════════════════════════════════
def _log_returns(df: pd.DataFrame) -> pd.Series:
    c = df['close'].astype('float64')
    return np.log(c.clip(lower=_EPS)).diff()


def _sigma(r: pd.Series, n: int) -> pd.Series:
    return r.rolling(n).std(ddof=1).clip(lower=_SIGMA_FLOOR)


def _lvl_logvol(df, window, long_window):
    """log of the realized volatility (std of the last ``window`` bar log-returns)."""
    return np.log(_sigma(_log_returns(df), window))


def _lvl_retvol(df, window, long_window):
    """window log-return in volatility units: log(close_t / close_{t-window}) / (sigma * sqrt(window))."""
    c = np.log(df['close'].astype('float64').clip(lower=_EPS))
    return (c - c.shift(window)) / (_sigma(_log_returns(df), window) * np.sqrt(window))


def _lvl_disthi(df, window, long_window):
    """distance of the last close below the window's highest high, in per-bar volatility units (>= 0)."""
    hi = np.log(df['high'].astype('float64').clip(lower=_EPS)).rolling(window).max()
    c = np.log(df['close'].astype('float64').clip(lower=_EPS))
    return (hi - c).clip(lower=0.0) / _sigma(_log_returns(df), window)


def _lvl_distlo(df, window, long_window):
    """distance of the last close above the window's lowest low, in per-bar volatility units (>= 0)."""
    lo = np.log(df['low'].astype('float64').clip(lower=_EPS)).rolling(window).min()
    c = np.log(df['close'].astype('float64').clip(lower=_EPS))
    return (c - lo).clip(lower=0.0) / _sigma(_log_returns(df), window)


def _lvl_logdvol_z(df, window, long_window):
    """log dollar volume of the window (mean of log1p(close * volume) over ``window`` bars) as a z-score against the trailing
    ``long_window``-bar mean / std of the same quantity: is this coin trading more or less than it has recently."""
    s = np.log1p((df['close'].astype('float64') * df['volume'].astype('float64')).clip(lower=0.0))
    m = s.rolling(window).mean()
    lmean, lstd = s.rolling(long_window).mean(), s.rolling(long_window).std(ddof=1).clip(lower=1e-6)
    return (m - lmean) / lstd


def _lvl_volreg(df, window, long_window):
    """volatility regime: log(sigma_window) - log(sigma_long_window) (> 0: more volatile than its own recent history)."""
    r = _log_returns(df)
    return np.log(_sigma(r, window)) - np.log(_sigma(r, long_window))


def _check_level_feature(name, fn, meta):
    return [] if callable(fn) else ["a level feature must be a callable(df, window, long_window) -> Series"]


#: registry: level feature name -> function. Add a feature = write the function + LEVEL_FEATURES.register(name, fn) (and nothing else).
LEVEL_FEATURES = _Registry("level feature", _Contract("level feature", optional_meta=(), validator=_check_level_feature))
for _n, _f in (('logvol', _lvl_logvol), ('retvol', _lvl_retvol), ('disthi', _lvl_disthi), ('distlo', _lvl_distlo),
               ('logdvol_z', _lvl_logdvol_z), ('volreg', _lvl_volreg)):
    LEVEL_FEATURES.register(_n, _f, version=1)
del _n, _f


def level_channel_settings(config: Optional[dict] = None) -> Optional[Dict[str, Any]]:
    """The validated settings of ``CONFIG['level_channel']`` (omitted keys filled from LEVEL_SETTINGS_DEFAULTS), or None when off.

    Raises ValueError: a non-dict value, an unknown key, an unregistered feature / component, window < 4, long_window not in
    (window, LEVEL_MAX_LONG_WINDOW], clip <= 0, an empty or duplicated feature list."""
    config = CONFIG if config is None else config
    raw = config.get('level_channel')
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"level_channel: None | dict of {sorted(LEVEL_SETTINGS_DEFAULTS)}, not {type(raw).__name__}")
    unknown = sorted(set(raw) - set(LEVEL_SETTINGS_DEFAULTS))
    if unknown:
        raise ValueError(f"level_channel: unknown key(s) {unknown}; allowed: {sorted(LEVEL_SETTINGS_DEFAULTS)}")
    s = {**{k: (list(v) if isinstance(v, list) else v) for k, v in LEVEL_SETTINGS_DEFAULTS.items()}, **raw}
    feats = list(s['features'])
    if not feats or len(set(feats)) != len(feats):
        raise ValueError(f"level_channel.features must be a non-empty list without duplicates, got {feats!r}")
    bad = [f for f in feats if f not in LEVEL_FEATURES]
    if bad:
        raise ValueError(f"level_channel.features: unregistered {bad}; registered: {list(LEVEL_FEATURES)}")
    if s['component'] not in LEVEL_CHANNELS:
        raise ValueError(f"level_channel.component: {' | '.join(map(repr, LEVEL_CHANNELS))} - not {s['component']!r}")
    w, lw = s['window'], s['long_window']
    if isinstance(w, bool) or not isinstance(w, int) or w < 4:
        raise ValueError(f"level_channel.window must be an int >= 4, got {w!r}")
    if isinstance(lw, bool) or not isinstance(lw, int) or not (w < lw <= LEVEL_MAX_LONG_WINDOW):
        raise ValueError(f"level_channel.long_window must be an int in ({w}, {LEVEL_MAX_LONG_WINDOW}], got {lw!r}")
    if isinstance(s['clip'], bool) or not isinstance(s['clip'], (int, float)) or not s['clip'] > 0:
        raise ValueError(f"level_channel.clip must be > 0, got {s['clip']!r}")
    s['features'], s['clip'] = feats, float(s['clip'])
    return s


def level_channel_columns(config: Optional[dict] = None) -> List[str]:
    """Names of the level columns (``LVL_<feature>``); ``[]`` when the channel is off."""
    s = level_channel_settings(config)
    return [] if s is None else [LEVEL_PREFIX + f for f in s['features']]


def add_level_features(df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """``df`` (open/high/low/close/volume, time-ordered) with the raw level columns added (float64, unscaled); unchanged object when off.
    Every value at row t depends on rows <= t only."""
    config = CONFIG if config is None else config
    s = level_channel_settings(config)
    if s is None:
        return df
    return get_level_channel(s['component']).add_features(df, s)


# ══════════════════════════════════════════════════════════════════════════
# the component
# ══════════════════════════════════════════════════════════════════════════
class LevelChannel:
    """Interface of a level side-channel (owner rule: fit on train only -> state; transform; every consumer calls it).

    * ``columns(settings)`` / ``add_features(df, settings)`` - the raw per-bar columns.
    * ``fit(train, feature_order=, timeframes=, settings=)`` -> ``state`` (JSON-safe dict, fitted on the TRAIN split alone).
    * ``transform_inputs(X, state)`` - ``X`` = ``{'X_<tf>': array}`` -> a new dict with the level columns standardized (inputs never mutated).
    * ``stamp(split_or_assets, state)`` - transform a split (or an ``{asset: split}`` dict) and put ``state`` in ``split['level_meta']``.
    * ``prepare_live(dataset, state)`` - the same transform for a live dataset (state required)."""
    name: str = ''

    def columns(self, settings: dict) -> List[str]:
        return [LEVEL_PREFIX + f for f in settings['features']]

    def add_features(self, df: pd.DataFrame, settings: dict) -> pd.DataFrame:
        out = df.copy()
        for f in settings['features']:
            out[LEVEL_PREFIX + f] = LEVEL_FEATURES[f](df, settings['window'], settings['long_window'])
        return out

    def _indices(self, feature_order: List[str], settings: dict) -> Dict[str, int]:
        cols = self.columns(settings)
        missing = [c for c in cols if c not in feature_order]
        if missing:
            raise ValueError(f"level channel: column(s) {missing} are not in the model features {list(feature_order)[:6]}... "
                             "(excluded by exclude_from_features? the channel is all-or-nothing per column list)")
        return {c: list(feature_order).index(c) for c in cols}

    def fit(self, train: Dict, *, feature_order: List[str], timeframes: List[str], settings: dict) -> Dict:
        idx = self._indices(feature_order, settings)
        stats = {}
        for tf in timeframes:
            X = train[f'X_{tf}']
            if len(X) == 0:
                raise ValueError("fit_level_channel: the train split is empty - no statistics")
            cols = list(idx.values())
            a = np.asarray(X[:, -1, cols], dtype='float64')          # the entry-time value of every TRAIN sample
            a = a[np.isfinite(a).all(axis=1)]
            if len(a) == 0:
                raise ValueError(f"fit_level_channel: no finite train samples for {tf}")
            stats[tf] = {'mean': [float(v) for v in a.mean(axis=0)], 'std': [float(max(v, 1e-8)) for v in a.std(axis=0)], 'n': int(len(a))}
        return {'version': 1, 'component': self.name, 'fit_on': 'train', 'columns': list(idx), 'index': list(idx.values()),
                'clip': float(settings['clip']), 'raw_scale': LEVEL_RAW_SCALE, 'stats': stats}

    def transform_inputs(self, X: Dict, state: Dict) -> Dict:
        out = dict(X)
        clip = float(state['clip'])
        for tf, st in state['stats'].items():
            key = f'X_{tf}'
            if key not in X:
                continue
            A = np.array(X[key])                                      # a copy: never write into a (memmapped) source
            idx = state['index']
            mu = np.asarray(st['mean'], dtype='float64')
            sd = np.asarray(st['std'], dtype='float64')
            A[:, :, idx] = np.clip((np.asarray(A[:, :, idx], dtype='float64') - mu) / sd, -clip, clip).astype(A.dtype)
            out[key] = A
        return out

    def _stamp_one(self, split: Dict, state: Dict) -> Dict:
        if 'level_meta' in split:
            raise ValueError("level channel: the split is already stamped with level_meta (double application)")
        out = self.transform_inputs(split, state)
        out['level_meta'] = state
        return out

    def stamp(self, split_or_assets: Dict, state: Dict) -> Dict:
        if state is None or state.get('component') != self.name:
            raise ValueError(f"level channel {self.name!r}: a train-fitted state of this component is required, got {None if state is None else state.get('component')!r}")
        if 'y' in split_or_assets:
            return self._stamp_one(split_or_assets, state)
        return {k: self._stamp_one(v, state) for k, v in split_or_assets.items()}

    def prepare_live(self, dataset: Dict, state: Optional[Dict]) -> Dict:
        if state is None:
            raise ValueError("level channel: a live dataset needs level_meta = train['level_meta'] (train statistics only)")
        out = self._stamp_one(dataset, state)
        out['level_channel'] = dict(dataset['level_channel'], state='fitted') if dataset.get('level_channel') else {'state': 'fitted'}
        return out


def _check_level_channel(name, comp, meta):
    if not isinstance(comp, LevelChannel):
        return [f"must be a LevelChannel instance, got {type(comp).__name__}"]
    return [] if comp.name == name else [f"registered as {name!r} but its .name is {comp.name!r}"]


class LevelV1(LevelChannel):
    name = 'level_v1'


#: registry: component name -> instance (duplicate raises, unknown raises). A new method = a new class + register_level_channel(...).
LEVEL_CHANNELS = _Registry("level channel", _Contract("level channel", optional_meta=(), validator=_check_level_channel))


def register_level_channel(comp: LevelChannel, version: int = 1) -> LevelChannel:
    return LEVEL_CHANNELS.register(comp.name, comp, version=version)


register_level_channel(LevelV1())


def get_level_channel(name: str) -> LevelChannel:
    if name not in LEVEL_CHANNELS:
        raise ValueError(f"level channel: {' | '.join(map(repr, LEVEL_CHANNELS))} - not {name!r}")
    return LEVEL_CHANNELS[name]


# ══════════════════════════════════════════════════════════════════════════
# dataset / split / live glue (every path calls the component, none re-implements it)
# ══════════════════════════════════════════════════════════════════════════
def level_channel_dataset_fields(config: Optional[dict] = None) -> Dict:
    """Keys added to a dataset when the channel is on (``{}`` when off: the old keys, literally)."""
    s = level_channel_settings(config)
    if s is None:
        return {}
    return {'level_channel': {'component': s['component'], 'settings': s, 'columns': level_channel_columns(config), 'state': 'unfitted'}}


def level_channel_unfitted(data: Dict) -> bool:
    return bool(data.get('level_channel')) and data['level_channel'].get('state') == 'unfitted'


def finish_level_channel_many(data: Dict, train: Dict, *others: Dict):
    """After splitting ``data``: fit the registered component on ``train`` ALONE and stamp ``train`` and every split of ``others``
    (a test may be an ``{asset: split}`` dict) with its state. Channel off (no ``data['level_channel']``) -> the splits as they are."""
    if not level_channel_unfitted(data):
        return (train,) + tuple(others)
    meta = data['level_channel']
    comp = get_level_channel(meta['component'])
    state = comp.fit(train, feature_order=data['feature_order'], timeframes=data['timeframes'], settings=meta['settings'])
    return tuple(comp.stamp(sp, state) for sp in (train,) + tuple(others))


def finish_level_channel(data: Dict, train: Dict, val: Dict, test: Dict):
    return finish_level_channel_many(data, train, val, test)


def stamp_level_channel(data: Dict, split_or_assets: Dict, level_meta: Optional[Dict] = None) -> Dict:
    """Apply an existing train state to a split that was not part of the fit (the sealed holdout). Off -> unchanged; on without a state raises."""
    if not level_channel_unfitted(data):
        return split_or_assets
    if level_meta is None:
        raise ValueError("this dataset has a level channel: pass level_meta = train['level_meta'] (train statistics only, nothing is fitted on this split)")
    return get_level_channel(data['level_channel']['component']).stamp(split_or_assets, level_meta)
