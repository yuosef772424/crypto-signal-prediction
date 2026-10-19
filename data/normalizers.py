"""
PURPOSE:  ONE registered Normalizer interface behind every price normalization mode ('window_scale', 'logret_std', 'logret_sigma'; the legacy window-level 'pct_change' resolves to 'window_scale' via normalizer_for_mode): fit on train only -> a plain-dict state, transform the inputs, normalize the targets, and the single target inverse. Consumers (windows, pipeline, split, invert_reg_predictions, live) call get_normalizer / normalizer_for_mode and never branch on a mode name.
TAGS:     Normalizer, NORMALIZERS, get_normalizer, normalizer_for_mode, normalizer_for_data, WindowScale, LogretStd, LogretSigma, fit, transform_inputs, normalize_targets, inverse_targets, state_from_config, price_norm_meta, fit_price_norm, denormalize_price_targets, normalize_price_targets, finish_price_norm, stamp_price_norm, trailing_close_sigma, price_norm_channels, normalization, تطبيع
PITFALLS: Add a normalizer = write a Normalizer subclass and put it in NORMALIZERS (nothing else to edit; its name is the value of CONFIG['price_norm_mode'] once PRICE_NORM_MODES in normalize.py lists it). fit sees TRAIN only; its state is a JSON-safe dict stored on every split (split['price_norm_meta']) and is the only thing the inverse/live path needs. 'window_scale' stores no state in datasets (default outputs stay byte-identical). The target inverse lives in core/price_norm.py (shared with evaluation.decode): never re-derive it. Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.price_norm import logret_forward, logret_inverse
from core.schema import (PRICE_NORM_CHANNELS, PRICE_NORM_CLIP, PRICE_NORM_LOGRET_SIGMA, PRICE_NORM_LOGRET_STD,
                         PRICE_NORM_SIGMA_FLOOR, PRICE_NORM_SIGMA_N, TARGET_COLUMNS)


# ══════════════════════════════════════════════════════════════════════════
# مساعدات مشتركة (عوائد لوغاريتمية نسبة للإغلاق السابق)
# ══════════════════════════════════════════════════════════════════════════
def logret_vs_prev_close(x: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
    """``log(x / prev_close)`` بـ float64؛ غير المنتهي (سعر غير موجب أو NaN) ``0`` لا inf/nan."""
    x = np.asarray(x, dtype='float64')
    p = np.asarray(prev_close, dtype='float64')
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.log(x / p)
    return np.where(np.isfinite(r), r, 0.0)


def trailing_close_sigma(close: np.ndarray, n: int = PRICE_NORM_SIGMA_N, floor: float = PRICE_NORM_SIGMA_FLOOR) -> np.ndarray:
    """``sigma[t]`` = الانحراف المعياري (ddof=1) لآخر ``n`` عوائد إغلاق لوغاريتمية **قبل** الشمعة ``t`` فقط (ماضٍ صِرف:
    عائد الشمعة ``t`` نفسها لا يدخل)، بأرضية ``floor``؛ ``NaN`` حتى يتوفّر ``n`` عائداً. (تعريف V3 في research/norm_probe.)"""
    c = np.asarray(close, dtype='float64')
    rc = np.r_[np.nan, logret_vs_prev_close(c[1:], c[:-1])] if len(c) else c
    s = pd.Series(rc).shift(1).rolling(n).std().to_numpy()
    return np.where(np.isfinite(s), np.maximum(s, floor), np.nan)


def price_norm_channels(feature_order: List[str]) -> List[Tuple[str, int]]:
    """``[(اسم، فهرس)]`` لقنوات open/high/low/close الموجودة في ميزات النموذج — وحدها تتحوّل إلى عوائد في الأوضاع logret_*."""
    return [(n, j) for j, n in enumerate(feature_order) if n in PRICE_NORM_CHANNELS]


def _reg_key(y: Dict, target: str) -> Optional[str]:
    for k in (f'y_{target}_reg', f'{target}_reg'):
        if k in y:
            return k
    return None


def _head_target(head: str) -> str:
    """'high' | 'high_reg' | 'y_high_reg' ← 'high'."""
    h = head[2:] if head.startswith('y_') else head
    return h[:-4] if h.endswith('_reg') else h


def _check_output(output: str, allowed: Tuple[str, ...]) -> None:
    if output not in allowed:
        raise ValueError(f"output: {' | '.join(map(repr, allowed))} — لا {output!r}")


# ══════════════════════════════════════════════════════════════════════════
# الواجهة
# ══════════════════════════════════════════════════════════════════════════
class Normalizer:
    """واجهة تطبيع أسعار الإدخال/الهدف. الكائن بلا حالة؛ الحالة ``state`` قاموس JSON-آمن يُلائَم على train وحده.

    * ``fit(train, feature_order=, timeframes=, clip=)`` → ``state`` (``state['mode'] == name``).
    * ``transform_inputs(X, state)`` — ``X`` قاموس ``{'X_<tf>': array}``؛ يُرجع قاموساً جديداً بعد تطبيق ``state``.
    * ``normalize_targets(y, state)`` — ``y`` قاموس أهداف بوحدتها الخام عند البناء؛ يُرجع القاموس المُقيَّس.
    * ``normalize_prices`` / ``inverse_targets(pred, state, last_close, head, output)`` — السعر ↔ فضاء الهدف
      (``last_close`` هي مرساة الرأس: آخر إغلاق في logret_*، آخر سعر من نوع الرأس نفسه في window_scale).
    * خطافات البناء (windows) والتقسيم (split) والبيانات الحيّة أدناه؛ الافتراضي منها «لا شيء» (= window_scale)."""
    name: str = ''
    supports_cross_sectional = True       # cross_sectional_normalize يعيد تقييس y_*_reg: يرفضه من أهدافه مرتبطة بإحصاءات train
    supports_live = True                  # build_dataset_live يقبل هذا التطبيع

    # ── بناء النوافذ ─────────────────────────────────────────────────────────
    def check_config(self, config: dict, targets, tf_order, features) -> None:
        """يرفع ``ValueError`` لإعداد لا يُعبَّر عنه بتناسق في هذا التطبيع."""

    def lookback_bars(self) -> int:
        """عدد الشموع المتصلة المطلوبة قبل بداية النافذة (0 = لا شرط)."""
        return 0

    def window_context(self, raw_close: np.ndarray):
        """سياق يُحسب مرة لكل أصل من الإغلاق الخام (مثلاً sigma الماضي) ويُمرَّر إلى ``build_inputs``."""
        return None

    def sample_valid(self, last_c, future_close, future_low_min, future_high_max) -> bool:
        return True

    def raw_reg_target(self, future: float, last_c: float) -> Optional[float]:
        """هدف الانحدار الخام لهذا التطبيع؛ None = يبقى بناء ``reg_target_mode`` المعتاد."""
        return None

    def build_inputs(self, windows, features, centers, scales, scaler_type, config, *, tf, is_base, price_df, end_idxs,
                     win_base, i_close_feat, context=None) -> np.ndarray:
        """النوافذ الخام ← X المُطبَّعة عند البناء (قبل تلاؤم حالة train)."""
        return process_windows(windows, features, centers, scales, scaler_type, config=config)

    # ── مجموعة البيانات والتقسيم ─────────────────────────────────────────────
    def dataset_fields(self, config: dict, features: List[str], tf_order: List[str]) -> Dict:
        """مفاتيح تُضاف إلى مجموعة البيانات (فارغة للافتراضي: المفاتيح القديمة حرفياً)."""
        return {}

    def fit_at_split(self, data: Dict) -> bool:
        """هل تُلائَم الحالة على train عند التقسيم (بيانات غير مُلائمة)؟"""
        return False

    def fit(self, train: Dict, *, feature_order: List[str], timeframes: List[str], clip: float = PRICE_NORM_CLIP) -> Dict:
        return {'version': 1, 'mode': self.name}

    def stamp_split(self, data: Dict, split: Dict, state: Optional[Dict]) -> Dict:
        """قسم واحد من ``data`` بعد تطبيق/ختم ``state``؛ الافتراضي: بلا مساس."""
        return split

    def stamp(self, data: Dict, split_or_assets: Dict, state: Optional[Dict] = None) -> Dict:
        """``stamp_split`` على قسم أو قاموس عملات ``{asset: split}``."""
        return split_or_assets

    # ── المدخلات والأهداف ────────────────────────────────────────────────────
    def transform_inputs(self, X: Dict, state: Dict) -> Dict:
        return dict(X)

    def normalize_targets(self, y: Dict, state: Dict) -> Dict:
        return y

    def state_from_config(self, config: dict, scale: Optional[float] = None) -> Dict:
        """حالة العكس حين لا يوجد ``price_norm_meta`` على القسم (window_scale: من CONFIG)."""
        return {'version': 1, 'mode': self.name}

    def anchor_column(self, head: str) -> str:
        """عمود ``last_candles`` الذي يُستعمل مرساةً للرأس."""
        return 'last_close'

    def normalize_prices(self, price, state: Dict, last_close, head: str = 'high') -> np.ndarray:
        raise NotImplementedError

    def inverse_targets(self, pred, state: Dict, last_close, head: str = 'high', output: str = 'price', **kw) -> np.ndarray:
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════
# window_scale (والاسم القديم pct_change على مستوى النافذة): بلا إحصاءات train
# ══════════════════════════════════════════════════════════════════════════
class WindowScale(Normalizer):
    """التطبيع الافتراضي: كل نافذة بمقياسها (``process_windows``، ومنه ``price_norm_mode='pct_change'``). هدف الانحدار
    ``reg_target_mode``: ``'return'`` (آخر سعر من نوع الرأس × (1 + pred / scale)) أو ``'window_scale'`` القديم."""
    name = 'window_scale'

    def dataset_fields(self, config, features, tf_order):
        out = {}
        if price_norm_mode(config) != PRICE_NORM_WINDOW:        # 'pct_change': وحدة الأعمدة السعرية في X
            out['price_norm_mode'] = price_norm_mode(config)
            out['price_pct_clip'] = price_pct_clip(config)
        return out

    def state_from_config(self, config, scale=None):
        return {'version': 1, 'mode': self.name, 'reg_target_mode': config.get('reg_target_mode', 'return'),
                'reg_target_scale': float(config.get('reg_target_scale', 1.0) if scale is None else scale)}

    def anchor_column(self, head):
        col = {'high': 'last_high', 'low': 'last_low', 'close': 'last_close'}.get(_head_target(head))
        if col is None:
            raise ValueError(f"لا مرجع سعر معروف للهدف '{_head_target(head)}'.")
        return col

    def normalize_prices(self, price, state, last_close, head='high'):
        scale = float(state.get('reg_target_scale', 1.0))
        return (np.asarray(price, dtype='float64') / np.asarray(last_close, dtype='float64') - 1.0) * scale

    def inverse_targets(self, pred, state, last_close, head='high', output='price', bases=None, **kw):
        _check_output(output, ('price',))
        scale = float(state.get('reg_target_scale', 1.0))
        preds = np.asarray(pred, dtype='float64') / scale
        last_price = np.asarray(last_close, dtype='float64')
        if state.get('reg_target_mode', 'return') == 'window_scale':
            return last_price + preds * np.asarray(bases, dtype='float64')[:, 1]
        return last_price * (1.0 + preds)


# ══════════════════════════════════════════════════════════════════════════
# logret_*: عوائد log نسبة للإغلاق السابق في قنوات OHLC وأهداف log(price / last_close)
# ══════════════════════════════════════════════════════════════════════════
class _LogretBase(Normalizer):
    supports_cross_sectional = False
    supports_live = False

    def check_config(self, config, targets, tf_order, features):
        """شروط الأوضاع logret_*: ترفع ``ValueError`` بدل تخمين ما لا يُعبَّر عنه بتناسق (انظر docstring prepare_single_asset)."""
        pn_mode = self.name
        if not price_norm_channels(features):
            raise ValueError(f"price_norm_mode={pn_mode!r} يحتاج قناة open/high/low/close واحدة على الأقل في ميزات النموذج "
                             "(exclude_from_features يستبعد open/high/low افتراضياً؛ close موجود)")
        if config.get('x_storage_dtype') and pn_mode == PRICE_NORM_LOGRET_STD:
            raise ValueError("price_norm_mode='logret_std' مع x_storage_dtype: العوائد الخام قبل التقييس تفقد دقّتها في float16؛ "
                             "عطّل x_storage_dtype")
        if config.get('reg_target_mode', 'return') != 'return':
            raise ValueError(f"price_norm_mode={pn_mode!r} يتطلب reg_target_mode='return' (هدفه log(price/last_close) مُقيَّساً)، "
                             f"لا {config.get('reg_target_mode')!r}")
        if float(config.get('reg_target_scale', 1.0)) != 1.0:
            raise ValueError(f"price_norm_mode={pn_mode!r} يتطلب reg_target_scale=1.0 (الهدف مُقيَّس بإحصاءات train لا بمقياس ثابت)، "
                             f"لا {config.get('reg_target_scale')!r}")
        self._check_heads(config, targets, tf_order)

    def _check_heads(self, config, targets, tf_order):
        pass

    def lookback_bars(self):
        return 1

    def sample_valid(self, last_c, future_close, future_low_min, future_high_max):
        return bool(last_c > 0 and future_close > 0 and future_low_min > 0 and future_high_max > 0)   # log غير معرَّف لسعر غير موجب

    def raw_reg_target(self, future, last_c):
        # المرساة last_close لكل الأهداف (كمدخلات النافذة)؛ التقييس بإحصاءات train وحدها يتمّ عند التقسيم
        return float(np.log(future / last_c))

    def build_inputs(self, windows, features, centers, scales, scaler_type, config, *, tf, is_base, price_df, end_idxs,
                     win_base, i_close_feat, context=None):
        X = process_windows(windows, features, centers, scales, scaler_type, config=config)
        self._write_channels(X, windows, is_base, price_df, end_idxs, win_base, price_norm_channels(features),
                             i_close_feat, context, tf)
        return X

    def _write_channels(self, X, raw_win, is_base, price_df, end_idxs, win, channels, i_close_feat, close_sigma, tf):
        """يستبدل قنوات OHLC في ``X`` (نتيجة process_windows) بعوائد ``log(x_t / close_{t-1})``:
        الفريم الأساسي — الإغلاق السابق الفعلي من الأسعار الخام (بما فيه الشمعة قبل النافذة)؛ فريم أعلى — من إغلاق النافذة نفسها
        والصف الأول 0. ``close_sigma`` (logret_sigma) ← ÷ sigma الماضي لكل شمعة ثم قصّ ±PRICE_NORM_CLIP؛ بدونه خام."""
        n = X.shape[0]
        if n == 0:
            return
        if is_base:
            idx = end_idxs[:, None] + np.arange(-(win - 1), 1)[None, :]
            prev = np.asarray(price_df['close'].to_numpy(), dtype='float64')[idx - 1]
            sig = close_sigma[idx] if close_sigma is not None else None
            for name, j in channels:
                x = np.asarray(price_df[name].to_numpy(), dtype='float64')[idx]
                r = logret_vs_prev_close(x, prev)
                if sig is not None:
                    r = np.clip(r / sig, -PRICE_NORM_CLIP, PRICE_NORM_CLIP)
                X[:, :, j] = r.astype(X.dtype)
        else:
            if i_close_feat is None:
                raise ValueError(f"price_norm_mode={self.name!r}: الفريم {tf!r} يحتاج 'close' بين الميزات (الإغلاق السابق داخل نافذته)")
            c = np.asarray(raw_win[:, :, i_close_feat], dtype='float64')
            prev = np.concatenate([np.full((n, 1), np.nan), c[:, :-1]], axis=1)
            for name, j in channels:
                X[:, :, j] = logret_vs_prev_close(np.asarray(raw_win[:, :, j], dtype='float64'), prev).astype(X.dtype)

    # ── الأهداف: مشتركة (إحصاءات الهدف في state['targets']) ──────────────────
    def _target_stats(self, state: Dict, head: str) -> Tuple[float, float]:
        t = _head_target(head)
        st = (state.get('targets') or {}).get(t)
        if st is None:
            raise ValueError(f"price_norm_meta بلا إحصاءات هدف {t!r} (mode={state.get('mode')!r}، الأهداف: "
                             f"{sorted(state.get('targets') or {})}) — 'logret_sigma' بلا أهداف انحدار")
        return float(st['mean']), float(st['std'])

    def normalize_prices(self, price, state, last_close, head='high'):
        """السعر المستقبلي ← فضاء الهدف: ``(log(price / last_close) − mean) / std`` (بدون قصّ)."""
        m, s = self._target_stats(state, head)
        return logret_forward(price, m, s, last_close)

    def inverse_targets(self, pred, state, last_close, head='high', output='price', **kw):
        """**العكس الوحيد** لأهداف الانحدار في الأوضاع logret_*: مخرَج النموذج (فضاء الهدف المُقيَّس) ← ``output``:
        ``'price'``: ``last_close × exp(pred × std + mean)``؛ ``'logret'``: ``pred × std + mean``."""
        _check_output(output, ('price', 'logret'))
        m, s = self._target_stats(state, head)
        return logret_inverse(pred, m, s, last_close, output)

    def state_from_config(self, config, scale=None):
        raise ValueError(f"price_norm_mode={self.name!r}: الأهداف log(price/last_close) مُقيَّسة بإحصاءات train — "
                         "مرّر price_norm_meta=split['price_norm_meta'] (أو استخدم denormalize_price_targets)")

    def _stamp_walk(self, data, split_or_assets, state):
        one = lambda sp: self.stamp_split(data, sp, state)                      # noqa: E731
        return one(split_or_assets) if 'y' in split_or_assets else {k: one(v) for k, v in split_or_assets.items()}

    def stamp(self, data, split_or_assets, state=None):
        return self._stamp_walk(data, split_or_assets, state)


class LogretStd(_LogretBase):
    """قنوات OHLC: ``clip((log(x_t / close_{t-1}) − mean) / std, ±5)`` بإحصاءات **train وحده**؛ أهداف
    ``(log(price / last_close) − mean_h) / std_h`` بإحصاءات train. البيانات UNFITTED (``price_norm_state='unfitted'``) حتى
    يُلائم ``split_data`` الحالة ويختمها في ``split['price_norm_meta']``."""
    name = PRICE_NORM_LOGRET_STD

    def dataset_fields(self, config, features, tf_order):
        return {'price_norm_mode': self.name, 'price_pct_clip': price_pct_clip(config), 'price_norm_state': 'unfitted'}

    def fit_at_split(self, data):
        return data.get('price_norm_state') == 'unfitted'

    def fit(self, train, *, feature_order, timeframes, clip=PRICE_NORM_CLIP):
        """يُلائم الإحصاءات على قسم **train وحده** ويُرجع ``state`` (قاموس JSON-آمن) تُخزَّن على كل قسم:

        * ``inputs[tf]`` — ``columns/index/mean/std`` لقنوات OHLC (متوسط وانحراف عناصر النوافذ، لكل قناة)؛
        * ``targets[t]`` — متوسط وانحراف هدف الانحدار ``log(price / last_close)`` (لكل هدف ``high/low/close``)؛
        * ``mode/clip/anchor`` — ``anchor='last_close'``: الهدف والعكس كلاهما نسبة لآخر إغلاق في النافذة."""
        ch = price_norm_channels(feature_order)
        if not ch:
            raise ValueError("price_norm_mode='logret_std' يحتاج قناة open/high/low/close واحدة على الأقل في ميزات النموذج")
        idx = [j for _, j in ch]
        inputs = {}
        for tf in timeframes:
            X = train[f'X_{tf}']
            if len(X) == 0:
                raise ValueError("fit_price_norm: قسم train فارغ — لا إحصاءات")
            a = np.asarray(X[:, :, idx], dtype='float64')
            inputs[tf] = {'columns': [n for n, _ in ch], 'index': list(idx),
                          'mean': [float(v) for v in a.mean(axis=(0, 1))],
                          'std': [float(max(v, 1e-12)) for v in a.std(axis=(0, 1))]}
        targets = {}
        y = train.get('y', {})
        for t in TARGET_COLUMNS:
            k = _reg_key(y, t)
            if k is not None:
                v = np.asarray(y[k], dtype='float64')
                targets[t] = {'mean': float(v.mean()), 'std': float(max(v.std(), 1e-12))}
        return {'version': 1, 'mode': self.name, 'clip': float(clip), 'anchor': 'last_close', 'inputs': inputs, 'targets': targets}

    def transform_inputs(self, X, state):
        """قنوات OHLC الخام ← ``clip((x − mean) / std, ±clip)``. تنسخ المصفوفات (لا تعدّل المصدر)."""
        out = dict(X)
        clip = float(state['clip'])
        for tf, st in state['inputs'].items():
            key = f'X_{tf}'
            if key not in X:
                continue
            A = np.array(X[key])
            idx = st['index']
            mu = np.asarray(st['mean'], dtype='float64')
            sd = np.asarray(st['std'], dtype='float64')
            A[:, :, idx] = np.clip((np.asarray(A[:, :, idx], dtype='float64') - mu) / sd, -clip, clip).astype(A.dtype)
            out[key] = A
        return out

    def normalize_targets(self, y, state):
        y = dict(y)
        for t, st in state['targets'].items():
            k = _reg_key(y, t)
            if k is not None:
                y[k] = ((np.asarray(y[k], dtype='float64') - st['mean']) / st['std']).astype('float32')
        return y

    def stamp_split(self, data, split, state):
        if not self.fit_at_split(data):                      # مُلائمة سلفاً: يُختم state البيانات فقط
            meta = data['price_norm_meta']
            return dict(split, price_norm_meta=meta, price_norm_mode=self.name)
        if state is None:
            raise ValueError("بيانات بوضع 'logret_std' غير مُلائمة: مرّر price_norm_meta = train['price_norm_meta'] "
                             "(إحصاءات train وحدها تُطبَّق على كل قسم)")
        if state.get('mode') != self.name:
            raise ValueError(f"apply_price_norm لـ {self.name!r} فقط، لا {state.get('mode')!r}")
        if 'price_norm_meta' in split:
            raise ValueError("apply_price_norm: القسم مختوم بـ price_norm_meta سلفاً (تطبيق مزدوج)")
        out = self.transform_inputs(split, state)
        out['y'] = self.normalize_targets(split.get('y', {}), state)
        out['price_norm_meta'] = state
        out['price_norm_mode'] = state['mode']
        return out


class LogretSigma(_LogretBase):
    """قنوات OHLC: ``clip(log(x_t / close_{t-1}) / sigma_t, ±5)`` حيث sigma الانحراف المعياري لآخر 14 عائد إغلاق **قبل**
    الشمعة (ماضٍ صِرف)؛ بلا إحصاءات مُلائمة ولا رؤوس انحدار ولا أكثر من فريم."""
    name = PRICE_NORM_LOGRET_SIGMA

    def _check_heads(self, config, targets, tf_order):
        enabled = [h for h in get_target_heads(targets, config) if h.endswith('_reg')]
        if enabled:
            raise ValueError(f"price_norm_mode='logret_sigma' لا يدعم رؤوس الانحدار {enabled}: هدفها يحتاج sigma لكل عيّنة "
                             "(per-sample sigma) غير محمول عبر التقسيم؛ عطّل رؤوس _reg عبر enabled_heads أو استخدم 'logret_std'")
        if len(tf_order) > 1:
            raise ValueError("price_norm_mode='logret_sigma' يدعم فريماً واحداً (sigma الماضي يحتاج تاريخاً قبل النافذة)")

    def lookback_bars(self):
        return 1 + PRICE_NORM_SIGMA_N

    def window_context(self, raw_close):
        return trailing_close_sigma(raw_close)

    def dataset_fields(self, config, features, tf_order):
        return {'price_norm_mode': self.name, 'price_pct_clip': price_pct_clip(config),
                'price_norm_meta': self.fit({}, feature_order=features, timeframes=list(tf_order), clip=float(PRICE_NORM_CLIP))}

    def fit(self, train, *, feature_order, timeframes, clip=PRICE_NORM_CLIP):
        """``state`` لـ 'logret_sigma': بلا إحصاءات مُلائمة (القسمة على sigma ماضٍ صِرف لكل شمعة) ولا أهداف انحدار."""
        ch = price_norm_channels(feature_order)
        return {'version': 1, 'mode': self.name, 'clip': float(clip), 'anchor': 'last_close',
                'sigma_n': PRICE_NORM_SIGMA_N, 'sigma_floor': PRICE_NORM_SIGMA_FLOOR,
                'inputs': {tf: {'columns': [n for n, _ in ch], 'index': [j for _, j in ch]} for tf in timeframes},
                'targets': {}}

    def stamp_split(self, data, split, state):
        return dict(split, price_norm_meta=data['price_norm_meta'], price_norm_mode=self.name)


#: السجلّ: اسم التطبيع ← كائنه. إضافة تطبيع = صنف جديد + سطر هنا.
NORMALIZERS: Dict[str, Normalizer] = {n.name: n for n in (WindowScale(), LogretStd(), LogretSigma())}


def get_normalizer(name: str) -> Normalizer:
    """تطبيع مسجَّل بالاسم؛ اسم غير مسجَّل يرفع ``ValueError``."""
    if name not in NORMALIZERS:
        raise ValueError(f"normalizer: {' | '.join(map(repr, NORMALIZERS))} — لا {name!r}")
    return NORMALIZERS[name]


def normalizer_for_mode(mode: str) -> Normalizer:
    """تطبيع ``CONFIG['price_norm_mode']``: 'pct_change' (ترميز على مستوى النافذة، بلا إحصاءات) ← 'window_scale'."""
    return get_normalizer(PRICE_NORM_WINDOW if mode == PRICE_NORM_PCT else mode)


def normalizer_for_data(data: Dict) -> Normalizer:
    """تطبيع مجموعة بيانات/قسم من ``price_norm_mode`` المختوم عليه (غيابه = 'window_scale')."""
    return normalizer_for_mode(data.get('price_norm_mode', PRICE_NORM_WINDOW))


# ══════════════════════════════════════════════════════════════════════════
# واجهات عامة متوافقة (غلاف رقيق فوق السجلّ)
# ══════════════════════════════════════════════════════════════════════════
def normalize_price_targets(price: np.ndarray, meta: Dict, last_close: np.ndarray, head: str = 'high') -> np.ndarray:
    """السعر المستقبلي ← فضاء الهدف: ``(log(price / last_close) − mean) / std`` (بدون قصّ). عكسه :func:`denormalize_price_targets`."""
    return get_normalizer(meta['mode']).normalize_prices(price, meta, last_close, head)


def denormalize_price_targets(pred: np.ndarray, meta: Dict, last_close: np.ndarray, head: str = 'high',
                              output: str = 'price') -> np.ndarray:
    """**العكس الوحيد** لأهداف الانحدار في الأوضاع logret_* (غلاف على ``Normalizer.inverse_targets``):

    * ``'price'`` (الافتراضي): ``last_close × exp(pred × std + mean)`` — السعر الحقيقي؛
    * ``'logret'``: ``pred × std + mean`` — العائد اللوغاريتمي نسبة لـ ``last_close``.
    ``head``: ``'high'`` أو ``'high_reg'`` أو ``'y_high_reg'``؛ ``meta`` = ``split['price_norm_meta']``؛ ``last_close`` من
    ``last_candles[:, LAST_COLUMNS.index('last_close')]``. يرفع ``ValueError`` لهدف بلا إحصاءات أو ``output`` مجهول."""
    return get_normalizer(meta['mode']).inverse_targets(pred, meta, last_close, head=head, output=output)


def fit_price_norm(train: Dict, feature_order: List[str], timeframes: List[str], mode: str = PRICE_NORM_LOGRET_STD,
                   clip: float = PRICE_NORM_CLIP) -> Dict:
    """``get_normalizer(mode).fit(...)`` على train وحده."""
    return get_normalizer(mode).fit(train, feature_order=feature_order, timeframes=timeframes, clip=clip)


def price_norm_unfitted(data: Dict) -> bool:
    """True إن كانت ``data`` غير مُلائمة بعد (تُلائَم في ``split_data`` على train)."""
    return normalizer_for_data(data).fit_at_split(data)


def stamp_price_norm(data: Dict, split_or_assets: Dict, meta: Optional[Dict] = None) -> Dict:
    """يطبّق/يختم ``meta`` على قسم (أو قاموس عملات) من ``data`` عبر تطبيقه المسجَّل؛ window_scale ← بلا مساس."""
    return normalizer_for_data(data).stamp(data, split_or_assets, meta)


def finish_price_norm(data: Dict, train: Dict, val: Dict, test: Dict):
    """بعد تقسيم ``data``: يُلائم تطبيقه المسجَّل على ``train`` وحده (إن لزم) ويختم/يطبّق على الأقسام الثلاثة (test قد يكون
    قاموس عملات). window_scale ← الأقسام كما هي. يُرجع ``(train, val, test)``."""
    nz = normalizer_for_data(data)
    state = nz.fit(train, feature_order=data['feature_order'], timeframes=data['timeframes'],
                   clip=PRICE_NORM_CLIP) if nz.fit_at_split(data) else None
    return tuple(nz.stamp(data, sp, state) for sp in (train, val, test))
