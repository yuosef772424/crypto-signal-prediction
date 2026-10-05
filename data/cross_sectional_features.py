"""
PURPOSE:  Cross-sectional features computed across all coins at the same timestamp: momentum rank, market breadth, momentum orthogonalised to NATR.
TAGS:     cross-sectional features, momentum rank, market breadth, momentum_orth_natr, NATR, rank residual
PITFALLS: Runs its _test_*() checks at load time. Needs every coin's returns at once (build_dataset_from_loader does a first pass, _load_cross_asset_frames). Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.
"""
# @title
"""
رتبة الزخم المقطعية (Cross-Sectional Momentum Rank).

فكرة مختلفة عن السياق السوقي أعلاه (عملة مرجعية واحدة): هنا نسأل "كيف أدى
هذا الأصل مقارنةً بكل الأصول الأخرى عند نفس اللحظة بالضبط؟" — رتبة مئوية
(0=الأسوأ أداءً بين كل الأصول في تلك اللحظة، 1=الأفضل) بدل العائد المطلق،
تلتقط دوراناً قطاعياً (بعض العملات تقود الحركة، أخرى تتبعها) قد لا يظهر في
عائد الأصل بمفرده. **سببية بالكامل**: عائد الأفق h عند الزمن t يعتمد فقط
على الماضي، والمقارنة بين الأصول تتم عند نفس t بالضبط — لا معلومة مستقبلية.

مُعطَّلة افتراضياً (``momentum_rank.enabled=False``). تحتاج رؤية عوائد كل الأصول
معاً دفعة واحدة، خلافاً للسياق السوقي (مرجع واحد يمكن تحميله بمعزل):
``build_dataset_from_preloaded`` يراها في ``data``، و``build_dataset_from_loader``
يحمّل high/low/close لكل العملات في مرور أول (:func:`_load_cross_asset_frames`).
"""


def momentum_rank_columns(config: Optional[dict] = None) -> List[str]:
    """أسماء أعمدة رتبة الزخم — قائمة ثابتة من الإعدادات وحدها، لتُدرَج في
    feature_order تلقائياً عبر infer_feature_columns."""
    config = CONFIG if config is None else config
    mr = config.get('momentum_rank') or {}
    if not mr.get('enabled', False):
        return []
    horizons = mr.get('horizons') or [24]
    return [f'MOM_RANK_{h}' for h in horizons]


def build_momentum_rank(data: Optional[Dict] = None,
                        config: Optional[dict] = None) -> Optional[Dict[str, Dict[str, pd.DataFrame]]]:
    """يحسب رتبة الزخم المقطعية لكل الأصول معاً — مرّة واحدة قبل حلقة
    المعالجة. ``None`` إن كان ``momentum_rank`` معطَّلاً أو ``data`` غائبة.
    ``data`` = ``{أصل: {فريم: DataFrame}}`` يكفيه عمود ``close``.

    Returns:
        ``{اسم_الأصل: {فريم: DataFrame(أعمدة MOM_RANK_h)}}`` — قيمة مختلفة
        لكل أصل (خلافاً لـ:func:`build_market_context` التي تُرجع قيمة
        مرجع واحدة مشتركة للجميع).
    """
    config = CONFIG if config is None else config
    mr = config.get('momentum_rank') or {}
    if not mr.get('enabled', False) or not data:
        return None
    horizons = mr.get('horizons') or [24]
    tf_order = config['tf_order']
    out: Dict[str, Dict[str, pd.DataFrame]] = {asset: {} for asset in data}
    for tf in tf_order:
        rets: Dict[str, pd.DataFrame] = {}
        for asset, tf_dict in data.items():
            if tf not in tf_dict or 'close' not in tf_dict[tf]:
                continue
            close = tf_dict[tf]['close'].astype('float64')
            log_close = np.log(close.clip(lower=1e-12))
            frame = pd.DataFrame(index=close.index)
            for h in horizons:
                frame[f'MOM_RANK_{h}'] = (log_close - log_close.shift(int(h))) * 100.0
            rets[asset] = frame
        if not rets:
            continue
        for h in horizons:
            col = f'MOM_RANK_{h}'
            combined = pd.concat({a: r[col] for a, r in rets.items()}, axis=1)
            rank_pct = combined.rank(axis=1, pct=True, na_option='keep')
            for asset in rets:
                out.setdefault(asset, {}).setdefault(tf, pd.DataFrame(index=rank_pct.index))
                out[asset][tf][col] = rank_pct[asset] if asset in rank_pct.columns else np.nan
    return out


def add_momentum_rank(dfs: Dict[str, pd.DataFrame],
                      asset_rank_dfs: Optional[Dict[str, pd.DataFrame]],
                      config: Optional[dict] = None) -> Dict[str, pd.DataFrame]:
    """يُلحق أعمدة MOM_RANK_ (محسوبة مسبقاً لهذا الأصل تحديداً عبر
    :func:`build_momentum_rank`) إلى ``dfs`` — محاذاة على فهرس كل فريم،
    ``fillna(0.5)`` (رتبة محايدة: لا أفضل ولا أسوأ) عند غياب القيمة (بداية
    السلسلة قبل توفّر أفق h، أو أصل غاب عن حساب الرتب المقطعية)."""
    cols = momentum_rank_columns(config)
    if not cols:
        return dfs
    out = {}
    for tf, df in dfs.items():
        d = df.copy()
        asset_rank = (asset_rank_dfs or {}).get(tf)
        for c in cols:
            if asset_rank is not None and c in asset_rank.columns:
                d[c] = asset_rank[c].reindex(d.index).fillna(0.5)
            else:
                d[c] = 0.5
        out[tf] = d
    return out


def _test_momentum_rank():
    """اختبار ذاتي لرتبة الزخم المقطعية: (أ) مُعطَّلة افتراضياً (قائمة
    أعمدة فارغة)، (ب) الرتبة تطابق ترتيب الأداء الفعلي يدوياً على 3 أصول
    صناعية بأداء معروف مسبقاً (صاعد بقوة/مستقرّ/هابط بقوة) — الأصل الأقوى
    أداءً يجب أن يحصل على أعلى رتبة (قريبة من 1) والأضعف أدنى رتبة (قريبة
    من 0)، (ج) سببية: الرتبة عند أي نقطة لا تتغيّر إن قُطعت السلسلة بعدها."""
    import numpy as np, pandas as pd
    n = 60
    idx = pd.date_range('2024-01-01', periods=n, freq='D')
    rng = np.random.default_rng(8)

    def make_df(drift):
        close = 100 + np.cumsum(np.full(n, drift) + rng.normal(0, 0.1, n))
        high = close + rng.random(n) * 0.2
        low = close - rng.random(n) * 0.2
        open_ = close + rng.normal(0, 0.05, n)
        volume = rng.random(n) * 100 + 10
        return pd.DataFrame({'open': open_, 'high': high, 'low': low,
                             'close': close, 'volume': volume}, index=idx)

    data = {
        'STRONG_UP': {'1D': make_df(1.0)},
        'FLAT': {'1D': make_df(0.0)},
        'STRONG_DOWN': {'1D': make_df(-1.0)},
    }

    cfg_off = {**CONFIG, 'momentum_rank': {'enabled': False}}
    assert momentum_rank_columns(cfg_off) == [], 'مُعطَّلة افتراضياً يجب أن تُعطي قائمة فارغة'
    assert build_momentum_rank(data=data, config=cfg_off) is None, 'مُعطَّلة يجب أن تُرجع None'

    cfg_on = {**CONFIG, 'tf_order': ['1D'], 'momentum_rank': {'enabled': True, 'horizons': [24]}}
    assert momentum_rank_columns(cfg_on) == ['MOM_RANK_24']

    rank_dfs = build_momentum_rank(data=data, config=cfg_on)
    assert rank_dfs is not None
    last = {a: rank_dfs[a]['1D']['MOM_RANK_24'].iloc[-1] for a in data}
    assert last['STRONG_UP'] > last['FLAT'] > last['STRONG_DOWN'], \
        f'الترتيب يجب أن يطابق الأداء الفعلي: {last}'
    assert np.isclose(last['STRONG_UP'], 1.0) and np.isclose(last['STRONG_DOWN'], 1 / 3), \
        f'3 أصول: الأقوى يجب أن يحصل على رتبة 1.0 بالضبط، الأضعف على 1/3: {last}'

    # سببية: تقطيع السلسلة عند k لا يُغيّر الرتبة عند تلك النقطة
    for k in (40, 50, 59):
        data_k = {a: {'1D': df['1D'].iloc[:k]} for a, df in data.items()}
        rank_k = build_momentum_rank(data=data_k, config=cfg_on)
        for a in data:
            full_val = rank_dfs[a]['1D']['MOM_RANK_24'].iloc[k - 1]
            trunc_val = rank_k[a]['1D']['MOM_RANK_24'].iloc[-1]
            assert np.isclose(full_val, trunc_val, equal_nan=True), \
                f'تسرّب معلومة مستقبلية عند k={k} للأصل {a}: {full_val} != {trunc_val}'

    out = add_momentum_rank(data['STRONG_UP'], rank_dfs['STRONG_UP'], cfg_on)
    assert 'MOM_RANK_24' in out['1D'].columns
    assert out['1D']['MOM_RANK_24'].notna().all()
    assert np.isclose(out['1D']['MOM_RANK_24'].iloc[-1], 1.0)

    print('✅ _test_momentum_rank: مُعطَّلة افتراضياً + رتبة تطابق الأداء '
          'الفعلي على 3 أصول صناعية + سببية مُتحقَّقة + إلحاق صحيح')


_test_momentum_rank()


def market_breadth_columns(config: Optional[dict] = None) -> List[str]:
    """أسماء أعمدة اتساع السوق (Market Breadth) — قائمة ثابتة من الإعدادات
    وحدها، بنفس دور momentum_rank_columns."""
    config = CONFIG if config is None else config
    mb = config.get('market_breadth') or {}
    if not mb.get('enabled', False):
        return []
    horizons = mb.get('horizons') or [1]
    return [f'MKT_BREADTH_{h}' for h in horizons]


def build_market_breadth(data: Optional[Dict] = None,
                         config: Optional[dict] = None) -> Optional[Dict[str, pd.DataFrame]]:
    """يحسب اتساع السوق (نسبة الأصول الصاعدة) لكل لحظة — قيمة **واحدة
    مشتركة** لكل الأصول (خلافاً لـ:func:`build_momentum_rank` التي تُنتج
    رتبة مختلفة لكل أصل)، بروح :func:`build_market_context` لكن تحتاج رؤية
    كل الأصول معاً (لا مرجعاً واحداً): ``data`` من ``build_dataset_from_preloaded``
    أو من مرور التحميل الأول في ``build_dataset_from_loader``. ``MKT_BREADTH_h`` = نسبة الأصول (0..1)
    التي عائدها على أفق h موجب، في نفس اللحظة بالضبط.

    Returns:
        ``{فريم: DataFrame(عمود MKT_BREADTH_h واحد لكل أفق)}`` — نفس القيمة
        تُستخدَم لكل الأصول عبر :func:`add_market_breadth`.
    """
    config = CONFIG if config is None else config
    mb = config.get('market_breadth') or {}
    if not mb.get('enabled', False) or not data:
        return None
    horizons = mb.get('horizons') or [1]
    tf_order = config['tf_order']
    out: Dict[str, pd.DataFrame] = {}
    for tf in tf_order:
        rets: Dict[str, pd.Series] = {}
        for asset, tf_dict in data.items():
            if tf not in tf_dict or 'close' not in tf_dict[tf]:
                continue
            close = tf_dict[tf]['close'].astype('float64')
            rets[asset] = close.pct_change(1)
        if not rets:
            continue
        ret_frame = pd.concat(rets, axis=1)
        frame = pd.DataFrame(index=ret_frame.index)
        for h in horizons:
            h_ret = ret_frame.rolling(int(h)).sum() if h > 1 else ret_frame
            # ✅ عملة بلا بيانات بعد (لم تُدرَج، أو قبل اكتمال أفق h) لا تُعدّ «غير صاعدة»:
            # NaN يُستبعد من المتوسط. سابقاً كان (NaN > 0) = False يخفض الاتساع في بدايات
            # التاريخ بقدر العملات غير المُدرَجة. لحظة بلا أي عملة ⇒ NaN ⇒ 0.5 المحايدة في
            # add_market_breadth.
            # astype قبل where: (bool).where(...) يُنتج object (خليط bool/NaN)، فيصير عمود الاتساع object ويحذّر
            # pandas من خفض النوع في fillna داخل add_market_breadth.
            advancing = (h_ret > 0).astype('float64').where(h_ret.notna())
            frame[f'MKT_BREADTH_{h}'] = advancing.mean(axis=1, skipna=True).astype('float64')
        out[tf] = frame
    return out


def add_market_breadth(dfs: Dict[str, pd.DataFrame],
                       breadth_dfs: Optional[Dict[str, pd.DataFrame]],
                       config: Optional[dict] = None) -> Dict[str, pd.DataFrame]:
    """يُلحق أعمدة MKT_BREADTH_ (نفس القيمة لكل الأصول عند نفس اللحظة) إلى
    ``dfs`` — محاذاة ``reindex(..., method='ffill')``، محايدة 0.5 (لا
    اتساع ولا انكماش) عند الغياب."""
    cols = market_breadth_columns(config)
    if not cols:
        return dfs
    out = {}
    for tf, df in dfs.items():
        d = df.copy()
        breadth = (breadth_dfs or {}).get(tf)
        if breadth is not None:
            aligned = breadth.reindex(d.index, method='ffill')
            for c in cols:
                d[c] = aligned[c].astype('float64').fillna(0.5) if c in aligned.columns else 0.5
        else:
            for c in cols:
                d[c] = 0.5
        out[tf] = d
    return out


def _test_market_breadth():
    """اختبار ذاتي لاتساق السوق: (أ) مُعطَّلة افتراضياً، (ب) 3 أصول
    اثنان صاعدان وواحد هابط عند لحظة معيّنة يجب أن يُعطيا 2/3≈0.667 بالضبط،
    (ج) القيمة مشتركة (نفس الرقم) لكل الأصول الثلاثة عند نفس اللحظة، (د)
    سببية: تقطيع السلسلة لا يُغيّر القيمة عند نقطة سابقة."""
    import numpy as np, pandas as pd
    n = 40
    idx = pd.date_range('2024-01-01', periods=n, freq='D')
    rng = np.random.default_rng(16)

    def make_df(rets):
        close = 100 * np.cumprod(1 + rets / 100.0)
        high = close + 0.1
        low = close - 0.1
        volume = rng.random(n) * 100 + 10
        return pd.DataFrame({'open': close, 'high': high, 'low': low, 'close': close,
                             'volume': volume}, index=idx)

    rets_up = np.full(n, 1.0)
    rets_up2 = np.full(n, 0.5)
    rets_down = np.full(n, -1.0)
    data = {'A': {'1D': make_df(rets_up)}, 'B': {'1D': make_df(rets_up2)},
           'C': {'1D': make_df(rets_down)}}

    cfg_off = {**CONFIG, 'market_breadth': {'enabled': False}}
    assert market_breadth_columns(cfg_off) == []
    assert build_market_breadth(data=data, config=cfg_off) is None

    cfg_on = {**CONFIG, 'tf_order': ['1D'], 'market_breadth': {'enabled': True, 'horizons': [1]}}
    assert market_breadth_columns(cfg_on) == ['MKT_BREADTH_1']

    breadth_dfs = build_market_breadth(data=data, config=cfg_on)
    assert breadth_dfs is not None
    val = breadth_dfs['1D']['MKT_BREADTH_1'].iloc[-1]
    assert np.isclose(val, 2.0 / 3.0), f'توقّعت 2/3، حصلت على {val}'

    out_a = add_market_breadth({'1D': data['A']['1D']},
                               breadth_dfs, config=cfg_on)['1D']
    out_c = add_market_breadth({'1D': data['C']['1D']}, breadth_dfs, config=cfg_on)['1D']
    assert np.isclose(out_a['MKT_BREADTH_1'].iloc[-1], out_c['MKT_BREADTH_1'].iloc[-1]), \
        'القيمة يجب أن تكون مشتركة (متطابقة) لكل الأصول عند نفس اللحظة'
    # عمود object (خليط bool/NaN) كان يُطلق FutureWarning في fillna على Colab
    assert breadth_dfs['1D']['MKT_BREADTH_1'].dtype == np.float64, breadth_dfs['1D'].dtypes
    assert out_a['MKT_BREADTH_1'].dtype == np.float64, out_a['MKT_BREADTH_1'].dtype

    for k in (25, 35, n):
        data_k = {a: {'1D': df['1D'].iloc[:k]} for a, df in data.items()}
        breadth_k = build_market_breadth(data=data_k, config=cfg_on)
        full_val = breadth_dfs['1D']['MKT_BREADTH_1'].iloc[k - 1]
        trunc_val = breadth_k['1D']['MKT_BREADTH_1'].iloc[-1]
        assert np.isclose(full_val, trunc_val, equal_nan=True), \
            f'تسرّب معلومة مستقبلية عند k={k}: {full_val} != {trunc_val}'

    print('✅ _test_market_breadth: مُعطَّلة افتراضياً + نسبة صحيحة (2/3) + '
          'قيمة مشتركة لكل الأصول + سببية مُتحقَّقة')


_test_market_breadth()


# ══════════════════════════════════════════════════════════════════════════
# الزخم المتعامد مع التقلّب (MOM_ORTH_NATR) — أول ميزة مقبولة خارج العيّنة
# ══════════════════════════════════════════════════════════════════════════
# اختبار مسجَّل مسبقاً (docs/research/preregistration_momentum_orth_natr_holdout.md)
# على 38 نافذة لم تُلمَس: IC = −0.109 مع هبوط الـlow النسبي، 84% معنوية، 38/38
# نافذة بنفس الإشارة. المعنى: أصل زخمه أعلى من أقرانه **ذوي نفس التقلّب** في
# نفس اليوم يهبط أقلّ من السوق في الأفق التالي. الزخم الخام وحده فشل خارج
# العيّنة — التعامد مع NATR_14 هو ما يكشف الإشارة (الزخم والتقلّب يتقاطعان
# مقطعياً فيُخفي أحدهما الآخر).
#
# الحساب (سببي بالكامل، يطابق تعريف الاختبار):
#   1. RET_h = عائد لوغاريتمي ×100 لكل أفق h ∈ horizons (نفس add_custom_features).
#   2. z_h = RET_h / (1.4826·وسيط|RET_h| على آخر norm_window شمعة) — نفس تحويل
#      sign_robust في process_windows (تراجع إلى الانحراف المعياري إن انعدم
#      الوسيط)، ثم قصّ ±clip_abs.
#   3. لكل لحظة: z_h ÷ انحرافه المعياري عبر الأصول، ثم المتوسط على الآفاق.
#   4. لكل لحظة: بواقي انحدار رتبة (3) على رتبة NATR عبر أصول تلك اللحظة،
#      مقسومة على نصف عدد الأصول ⇒ بمقياس ±1 تقريباً.
# لحظة بأقلّ من min_assets أصلاً ⇒ NaN (يُملأ صفراً محايداً عند الإلحاق).


def momentum_orth_natr_columns(config: Optional[dict] = None) -> List[str]:
    """اسم عمود الزخم المتعامد مع التقلّب — [] إن كان معطَّلاً."""
    config = CONFIG if config is None else config
    mo = config.get('momentum_orth_natr') or {}
    return ['MOM_ORTH_NATR'] if mo.get('enabled', False) else []


def _natr_raw(df: pd.DataFrame, length: int) -> pd.Series:
    """NATR بنفس حساب pandas_ta المستخدم في add_features (df.ta.natr)."""
    out = df[['high', 'low', 'close']].astype('float64').ta.natr(length=int(length))
    return pd.Series(np.asarray(out, dtype='float64'), index=df.index)


def _rowwise_rank_residual(Y: np.ndarray, Xc: np.ndarray, min_n: int) -> np.ndarray:
    """لكل صفّ (لحظة): بواقي انحدار رتبة Y على رتبة Xc عبر الأعمدة الصالحة
    (مع ثابت)، مقسومة على n/2. صفوف بأقلّ من min_n قيمة صالحة ⇒ NaN."""
    out = np.full(Y.shape, np.nan)
    for i in range(Y.shape[0]):
        m = np.isfinite(Y[i]) & np.isfinite(Xc[i])
        n = int(m.sum())
        if n < min_n:
            continue
        ry = pd.Series(Y[i, m]).rank().to_numpy()
        rx = pd.Series(Xc[i, m]).rank().to_numpy()
        rx_c = rx - rx.mean()
        denom = float(rx_c @ rx_c)
        beta = float(rx_c @ (ry - ry.mean())) / denom if denom > 0 else 0.0
        out[i, m] = (ry - ry.mean() - beta * rx_c) / (n / 2.0)
    return out


def build_momentum_orth_natr(data: Optional[Dict] = None,
                             config: Optional[dict] = None) -> Optional[Dict[str, Dict[str, pd.DataFrame]]]:
    """يحسب MOM_ORTH_NATR لكل الأصول معاً (مثل build_momentum_rank) — ``None``
    إن كانت معطَّلة أو ``data`` غائبة. يُرجع ``{أصل: {فريم: DataFrame}}``."""
    config = CONFIG if config is None else config
    mo = config.get('momentum_orth_natr') or {}
    if not mo.get('enabled', False) or not data:
        return None
    horizons = [int(h) for h in (mo.get('horizons') or [1, 3, 6, 12, 24])]
    natr_len = int(mo.get('natr_length', 14))
    min_assets = int(mo.get('min_assets', 5))
    clip = float(config.get('clip_abs', 5.0))
    eps = float(config.get('eps', 1e-8))
    out: Dict[str, Dict[str, pd.DataFrame]] = {asset: {} for asset in data}
    for tf in config['tf_order']:
        norm_window = int((config.get('window_sizes') or {}).get(tf, 32))
        z_by_h = {h: {} for h in horizons}
        natr = {}
        for asset, tf_dict in data.items():
            df = tf_dict.get(tf) if isinstance(tf_dict, dict) else None
            if df is None or not {'high', 'low', 'close'} <= set(df.columns):
                continue
            log_close = np.log(df['close'].astype('float64').clip(lower=1e-12))
            for h in horizons:
                r = (log_close - log_close.shift(h)) * 100.0
                mag = 1.4826 * r.abs().rolling(norm_window, min_periods=norm_window).median()
                fb = r.rolling(norm_window, min_periods=norm_window).std(ddof=0)
                mag = mag.where(mag >= eps, np.maximum(fb, eps))
                z_by_h[h][asset] = (r / mag).clip(-clip, clip)
            natr[asset] = _natr_raw(df, natr_len)
        if not natr:
            continue
        natr_df = pd.concat(natr, axis=1).sort_index()
        assets, idx = list(natr_df.columns), natr_df.index
        parts = []
        for h in horizons:
            zh = pd.concat(z_by_h[h], axis=1).reindex(index=idx, columns=assets)
            sd = zh.std(axis=1, ddof=0)
            parts.append(zh.div(sd.where(sd > eps), axis=0))
        mom = sum(p.to_numpy() for p in parts) / len(parts)     # NaN إن غاب أيّ أفق
        res = _rowwise_rank_residual(mom, natr_df.to_numpy(), min_assets)
        res_df = pd.DataFrame(res, index=idx, columns=assets)
        for asset in assets:
            out.setdefault(asset, {})[tf] = pd.DataFrame({'MOM_ORTH_NATR': res_df[asset]})
    return out


def add_momentum_orth_natr(dfs: Dict[str, pd.DataFrame],
                           asset_dfs: Optional[Dict[str, pd.DataFrame]],
                           config: Optional[dict] = None) -> Dict[str, pd.DataFrame]:
    """يُلحق MOM_ORTH_NATR (محسوبة مسبقاً لهذا الأصل) — ``fillna(0.0)``
    (محايد: لا زخم زائد ولا ناقص عن أقرانه) عند الغياب (``asset_dfs=None``، أو
    لحظة بأقل من ``min_assets`` أصلاً)."""
    cols = momentum_orth_natr_columns(config)
    if not cols:
        return dfs
    out = {}
    for tf, df in dfs.items():
        d = df.copy()
        src = (asset_dfs or {}).get(tf)
        if src is not None and 'MOM_ORTH_NATR' in src.columns:
            d['MOM_ORTH_NATR'] = src['MOM_ORTH_NATR'].reindex(d.index).fillna(0.0)
        else:
            d['MOM_ORTH_NATR'] = 0.0
        out[tf] = d
    return out


def _test_momentum_orth_natr():
    """(أ) معطَّلة ⇒ لا أعمدة و None. (ب) متعامدة فعلاً مع NATR داخل كل لحظة
    (ارتباط رتب ≈ 0). (ج) أصل بنفس تقلّب أقرانه وزخم أعلى يحصل على قيمة أعلى.
    (د) سببية: قطع السلسلة بعد k لا يُغيّر القيمة عند k."""
    import numpy as np, pandas as pd
    n = 120
    idx = pd.date_range('2024-01-01', periods=n, freq='D')
    rng = np.random.default_rng(11)

    def make_df(drift, vol):
        ret = drift + rng.normal(0, vol, n)
        close = 100 * np.exp(np.cumsum(ret))
        spread = np.abs(rng.normal(0, vol, n)) + vol / 2
        return pd.DataFrame({'open': close, 'high': close * (1 + spread), 'low': close * (1 - spread),
                             'close': close, 'volume': rng.random(n) * 100 + 10}, index=idx)

    specs = {f'A{i}': (d, v) for i, (d, v) in enumerate(
        [(0.004, 0.01), (0.0, 0.01), (-0.004, 0.01), (0.016, 0.04), (0.0, 0.04), (-0.016, 0.04),
         (0.002, 0.02), (-0.002, 0.02)])}
    data = {a: {'1D': make_df(d, v)} for a, (d, v) in specs.items()}
    cfg_off = {**CONFIG, 'tf_order': ['1D'], 'momentum_orth_natr': {'enabled': False}}
    assert momentum_orth_natr_columns(cfg_off) == []
    assert build_momentum_orth_natr(data=data, config=cfg_off) is None
    cfg = {**CONFIG, 'tf_order': ['1D'], 'window_sizes': {'1D': 32},
           'momentum_orth_natr': {'enabled': True, 'horizons': [1, 3, 6, 12, 24]}}
    assert momentum_orth_natr_columns(cfg) == ['MOM_ORTH_NATR']
    res = build_momentum_orth_natr(data=data, config=cfg)
    M = pd.concat({a: res[a]['1D']['MOM_ORTH_NATR'] for a in data}, axis=1)
    N = pd.concat({a: _natr_raw(data[a]['1D'], 14) for a in data}, axis=1)
    valid = M.dropna(how='any').index
    assert len(valid) > 40, f'لحظات صالحة قليلة: {len(valid)}'
    # البواقي متعامدة خطياً مع رتبة NATR داخل كل لحظة (ارتباط بيرسون مع الرتبة = 0 بالضبط)
    rhos = [np.corrcoef(M.loc[t].to_numpy(), N.loc[t].rank().to_numpy())[0, 1] for t in valid]
    assert np.nanmax(np.abs(rhos)) < 1e-9, f'غير متعامدة مع NATR: {np.nanmax(np.abs(rhos))}'
    assert float(M.loc[valid].abs().max().max()) <= 2.0   # بواقي رتب ÷ (n/2): بمقياس ±1 تقريباً
    late = valid[-20:]
    assert M.loc[late, 'A0'].mean() > M.loc[late, 'A2'].mean(), 'الزخم الأعلى بنفس التقلّب يجب أن يعلو'
    assert M.loc[late, 'A3'].mean() > M.loc[late, 'A5'].mean()
    for k in (70, 100):
        cut = {a: {'1D': d['1D'].iloc[:k]} for a, d in data.items()}
        rk = build_momentum_orth_natr(data=cut, config=cfg)
        for a in data:
            full_v = res[a]['1D']['MOM_ORTH_NATR'].iloc[k - 1]
            cut_v = rk[a]['1D']['MOM_ORTH_NATR'].iloc[-1]
            assert np.isclose(full_v, cut_v, equal_nan=True), f'تسرّب مستقبلي عند k={k} ({a})'
    added = add_momentum_orth_natr(data['A0'], res['A0'], cfg)
    assert added['1D']['MOM_ORTH_NATR'].notna().all()
    assert (add_momentum_orth_natr(data['A0'], None, cfg)['1D']['MOM_ORTH_NATR'] == 0.0).all()
    print('✅ _test_momentum_orth_natr: متعامدة مع NATR داخل كل لحظة + ترتيب الزخم '
          'بنفس التقلّب صحيح + سببية مُتحقَّقة + إلحاق/قيمة محايدة صحيحان')


_test_momentum_orth_natr()
