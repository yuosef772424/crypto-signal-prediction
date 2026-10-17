"""
PURPOSE:  Latest-samples table (date, signal, confidence, uncertainty, expected move) that works without true targets.
TAGS:     build_latest_table, format_latest_table, print_latest_table, timestamps, latest samples, live table
PITFALLS: Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.

## 6️⃣.٥ جدول آخر العينات (التاريخ · التوقع · الثقة · عدم اليقين) — يعمل بلا أهداف حقيقية
"""
# ─────────────────────────────────────────────────────────────────────────────
# المرحلة 3.5: جدول "آخر العينات" (تقييم/تداول حي) — التاريخ + التوقع + الثقة + عدم اليقين
# ─────────────────────────────────────────────────────────────────────────────
#
# يعمل بلا أهداف حقيقية إطلاقاً: عمود الهدف/النتيجة يظهر فقط إن وُجدت قيمة فعلية
# (وإلا يظهر ⏳ "بلا هدف بعد" — مثل آخر عينة في التداول الحي).
# ─────────────────────────────────────────────────────────────────────────────

_TIME_KEYS = ('timestamps', 'timestamp', 'times', 'time', 'dates', 'date',
              'datetime', 'open_time', 'candle_time')


def _to_datetime_index(values) -> Optional[pd.DatetimeIndex]:
    """
    يحوّل قيم الوقت إلى DatetimeIndex: datetime / نص / epoch (ثوانٍ، مللي، مايكرو، نانو
    — يُكتشف تلقائياً من الحجم). يُرجع None إن تعذّر التحويل أو لم تبدُ القيم كأوقات.
    """
    if values is None:
        return None
    try:
        arr = np.asarray(values)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        if arr.size == 0:
            return None
        if np.issubdtype(arr.dtype, np.datetime64):
            return pd.DatetimeIndex(arr)
        if arr.dtype.kind in 'iuf':
            f = arr.astype(np.float64)
            finite = f[np.isfinite(f)]
            if finite.size == 0:
                return None
            m = abs(float(np.median(finite)))
            unit = 'ns' if m >= 1e17 else 'us' if m >= 1e14 else 'ms' if m >= 1e11 else 's' if m >= 1e8 else None
            if unit is None:
                return None                    # أرقام صغيرة (فهارس) وليست طوابع زمنية
            return pd.DatetimeIndex(pd.to_datetime(f, unit=unit, errors='coerce'))
        idx = pd.DatetimeIndex(pd.to_datetime(arr, errors='coerce'))
        return idx if idx.notna().any() else None
    except Exception:
        return None


def _extract_timestamps(timestamps=None, last_candles=None, timestamp_col=None) -> Optional[pd.DatetimeIndex]:
    """الأولوية: timestamps الصريحة ثم عمود timestamp_col من last_candles."""
    if timestamps is not None:
        return _to_datetime_index(timestamps)
    if last_candles is not None and timestamp_col is not None:
        return _to_datetime_index(_safe_col(last_candles, timestamp_col))
    return None


def _find_timestamps(test_data: Dict, timestamp_key: Optional[str] = None,
                     timestamp_col: Optional[int] = None) -> Optional[pd.DatetimeIndex]:
    """يبحث عن الأوقات داخل بيانات أصل واحد (مفاتيح شائعة أو timestamp_col في last_candles)."""
    keys = [timestamp_key] if timestamp_key else list(_TIME_KEYS)
    for k in keys:
        if k and k in test_data and test_data[k] is not None:
            ts = _to_datetime_index(test_data[k])
            if ts is not None:
                return ts
    return _extract_timestamps(None, test_data.get('last_candles'), timestamp_col)


def _align_timestamps(ts: Optional[pd.DatetimeIndex], n_total: int, k: int) -> Optional[pd.DatetimeIndex]:
    """يُرجع أوقات آخر k عينة: ts بطول n_total الكامل أو بطول k فقط؛ غير ذلك None."""
    if ts is None:
        return None
    if len(ts) == n_total:
        return ts[n_total - k:]
    if len(ts) == k:
        return ts
    warnings.warn(f"طول الأوقات ({len(ts)}) لا يطابق عدد العينات ({n_total}) — سيُعرض رقم العينة بدل التاريخ.")
    return None


# ─── جدول آخر العينات ─────────────────────────────────────────────────────────

def build_latest_table(
    decoded: Dict[str, Dict[str, np.ndarray]],
    target_specs,
    n_display: int = 5,
    timestamps=None,
    asset: Optional[str] = None,
) -> pd.DataFrame:
    """
    جدول (عينة × هدف) لآخر `n_display` عينة من نتائج decode/evaluate.

    الأعمدة: idx, date, [asset], target, kind, signal, confidence_%, uncertainty_%,
             move_% (التحرك المتوقع عن سعر الدخول)، pred (السعر المتوقع)، pred_lo/pred_hi
             (pred ± عدم اليقين)، entry، true، true_label، has_true، ok.
    `ok`: True/False إن وُجد هدف فعلي، None إن لم يوجد بعد (تداول حي).
    `timestamps` يجب أن تطابق (بالترتيب) العينات المُمرَّرة في decoded.
    """
    specs = [s for s in resolve_targets(target_specs) if isinstance(decoded.get(s.name), dict)]
    if not specs or not n_display or n_display <= 0:
        return pd.DataFrame()

    def _n_of(r):
        return len(r['pred_real']) if 'pred_real' in r else len(r['pred_index'])

    n = min(_n_of(decoded[s.name]) for s in specs)
    k = int(min(n_display, n))

    ts = _to_datetime_index(timestamps) if timestamps is not None else None
    if ts is not None:
        ts = ts[len(ts) - n:] if len(ts) >= n else None

    rows = []
    for i in range(n - k, n):
        date = ts[i] if ts is not None else pd.NaT
        for spec in specs:
            r = decoded[spec.name]
            row = {'idx': i, 'date': date, 'target': spec.name, 'kind': spec.kind}
            if asset is not None:
                row['asset'] = asset

            if spec.kind == 'continuous':
                pred = float(r['pred_real'][i])
                entry = float(r['entry_price'][i]) if 'entry_price' in r else float('nan')
                entry_ok = np.isfinite(entry) and entry != 0.0
                move = (pred - entry) / (abs(entry) + 1e-7) * 100 if entry_ok else float('nan')

                unc = r.get('uncertainty_real')
                unc_i = float(unc[i]) if unc is not None else float('nan')
                conf = r.get('confidence')
                conf_i = float(conf[i]) * 100 if conf is not None else float('nan')

                true = r.get('true_real')
                if true is None:
                    true = r.get('target_real')
                true_i = float(true[i]) if true is not None and np.isfinite(true[i]) else float('nan')
                has_true = bool(np.isfinite(true_i))
                ok = bool(np.sign(pred - entry) == np.sign(true_i - entry)) if (has_true and entry_ok) else None

                row.update({
                    'signal': '—' if not np.isfinite(move) else ('▲' if move > 0 else '▼' if move < 0 else '■'),
                    'confidence_%': conf_i,
                    'uncertainty_%': unc_i / (abs(pred) + 1e-7) * 100 if np.isfinite(unc_i) else float('nan'),
                    'move_%': move,
                    'pred': pred, 'pred_lo': pred - unc_i, 'pred_hi': pred + unc_i,
                    'entry': entry if entry_ok else float('nan'),
                    'true': true_i, 'true_label': None, 'has_true': has_true, 'ok': ok,
                })

            else:  # categorical
                proba = float(r['pred_proba'][i]) if 'pred_proba' in r else float('nan')
                conf = r.get('confidence')
                conf_i = float(conf[i]) * 100 if conf is not None else proba * 100
                has_true = 'true_label' in r
                ok = bool(r['pred_index'][i] == r['true_index'][i]) if 'true_index' in r else None
                row.update({
                    'signal': str(r['pred_label'][i]),
                    'confidence_%': conf_i, 'uncertainty_%': float('nan'), 'move_%': float('nan'),
                    'pred': float('nan'), 'pred_lo': float('nan'), 'pred_hi': float('nan'),
                    'entry': float('nan'), 'true': float('nan'),
                    'true_label': str(r['true_label'][i]) if has_true else None,
                    'has_true': has_true, 'ok': ok,
                })
            rows.append(row)

    return pd.DataFrame(rows)


def _fmt_price(x) -> str:
    if x is None or not np.isfinite(x):
        return '—'
    a = abs(x)
    if a >= 1000:
        return f"{x:,.2f}"
    if a >= 1:
        return f"{x:.4f}"
    if a >= 0.01:
        return f"{x:.5f}"
    return f"{x:.7f}"


def _fmt_num(x, fmt: str) -> str:
    return '—' if (x is None or not np.isfinite(x)) else format(x, fmt)


def format_latest_table(df: pd.DataFrame) -> pd.DataFrame:
    """يحوّل جدول build_latest_table إلى نصوص مضغوطة جاهزة للطباعة."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=df.index)
    if df['date'].notna().any():
        out['date'] = df['date'].map(lambda d: d.strftime('%Y-%m-%d %H:%M') if pd.notna(d) else '—')
    else:
        out['#'] = df['idx']
    if 'asset' in df.columns:
        out['asset'] = df['asset']
    out['target'] = df['target']
    out['signal'] = df['signal']
    out['conf'] = df['confidence_%'].map(lambda v: _fmt_num(v, '.1f') + ('%' if np.isfinite(v) else ''))
    out['unc'] = df['uncertainty_%'].map(lambda v: _fmt_num(v, '.2f') + ('%' if np.isfinite(v) else ''))
    out['move'] = df['move_%'].map(lambda v: _fmt_num(v, '+.3f') + ('%' if np.isfinite(v) else ''))
    out['pred'] = [_fmt_price(p) if k == 'continuous' else '—' for p, k in zip(df['pred'], df['kind'])]
    if (df['kind'] == 'continuous').any():
        out['entry'] = df['entry'].map(_fmt_price)
    if df['has_true'].any():
        out['true'] = [
            tl if (k == 'categorical' and isinstance(tl, str)) else _fmt_price(t)
            for t, tl, k in zip(df['true'], df['true_label'], df['kind'])
        ]
    if df['ok'].notna().any() or df['has_true'].any():
        out['ok'] = df['ok'].map(
            lambda v: '⏳' if (v is None or (isinstance(v, float) and np.isnan(v))) else ('✅' if bool(v) else '❌')
        )
    return out


def _latest_legend(df: pd.DataFrame) -> str:
    txt = ("signal: ▲ صعود متوقع | ▼ هبوط  ·  conf: الثقة  ·  unc: عدم اليقين كنسبة من السعر المتوقع"
           "  ·  move: التحرك المتوقع عن سعر الدخول")
    if df is not None and not df.empty and (df['ok'].notna().any() or df['has_true'].any()):
        txt += "  ·  ⏳: لا يوجد هدف فعلي بعد"
    return txt


def print_latest_table(df: pd.DataFrame, title: Optional[str] = None, legend: bool = True, indent: str = "  "):
    if df is None or df.empty:
        print(indent + "(لا توجد عينات للعرض)")
        return
    if title:
        print(title)
    txt = format_latest_table(df).to_string(index=False)
    print("\n".join(indent + ln for ln in txt.split("\n")))
    if legend:
        print(indent + _latest_legend(df))
