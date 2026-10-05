"""
PURPOSE:  Phase-2 glue: locates tools/intraday_features.py, resolves the phase2_data toggles, caches archive reads and adds the 15m intraday and futures-metrics features.
TAGS:     phase 2, phase2_data, intraday_features, 15m, futures metrics, require_phase2_data, resolve_phase2_toggles, module_dirs
PITFALLS: All arithmetic lives in tools/intraday_features.py (tested by tests/test_intraday_features.py); this module only finds paths, toggles, caches and fills neutrally. Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.
"""
# ══════════════════════════════════════════════════════════════════════════
# المرحلة ٢: أرشيف Binance الكامل (شموع 15m + تمويل + OI + futures_metrics)
# ══════════════════════════════════════════════════════════════════════════
# المنطق الحسابي كله في tools/intraday_features.py (مُختبَر في tests/test_intraday_features.py)؛
# هنا فقط: إيجاد المسارات، مفاتيح التشغيل، التخزين المؤقت، والملء المحايد (NaN ← 0 + علم توفّر).
import importlib.util as _ilu
import threading as _threading
import sys as _sys

_PHASE2_MOD: Dict[str, Any] = {"mod": None, "tried": False}
_PHASE2_LOCK = _threading.Lock()
_PHASE2_CACHE: "Dict[tuple, Any]" = {}
_PHASE2_CACHE_MAX = 32
_PHASE2_KEYS = ('use_intraday_15m', 'use_futures_metrics')


def _phase2_cfg(config: Optional[dict] = None) -> dict:
    config = CONFIG if config is None else config
    return config.get('phase2_data') or {}


def _phase2_module(config: Optional[dict] = None):
    """يستورد tools/intraday_features.py من المستودع (cwd، أو مسارات phase2_data['module_dirs'])."""
    if _PHASE2_MOD["tried"]:
        return _PHASE2_MOD["mod"]
    with _PHASE2_LOCK:
        if _PHASE2_MOD["tried"]:
            return _PHASE2_MOD["mod"]
        dirs = [os.getcwd()] + list(_phase2_cfg(config).get('module_dirs') or [])
        for d in dirs:
            path = Path(d) / 'tools' / 'intraday_features.py'
            if path.exists():
                spec = _ilu.spec_from_file_location('intraday_features', path)
                mod = _ilu.module_from_spec(spec)
                _sys.modules['intraday_features'] = mod
                spec.loader.exec_module(mod)
                _PHASE2_MOD["mod"] = mod
                print(f"🧩 المرحلة ٢: tools/intraday_features.py ← {path}")
                break
        _PHASE2_MOD["tried"] = _PHASE2_MOD["mod"] is not None   # لم يوجد ⇒ يُعاد البحث لاحقاً
    return _PHASE2_MOD["mod"]


def _phase2_root(config: Optional[dict] = None, mount: bool = False) -> Optional[Path]:
    """جذر المجلدات: phase2_data['data_root'] إن حُدِّد، وإلا MyDrive (مُركَّب مسبقاً — أو يُركَّب إن mount)."""
    p2 = _phase2_cfg(config)
    if p2.get('data_root'):
        return Path(p2['data_root'])
    if _DRIVE_ROOT.get("root") is not None:
        return _DRIVE_ROOT["root"]
    mp = Path((CONFIG if config is None else config).get('drive_mount_point', '/content/drive')) / 'MyDrive'
    if mp.is_dir():
        return mp
    return mount_drive(config=config) if mount else None


def _phase2_dirs(key: str, config: Optional[dict] = None) -> List[str]:
    config = CONFIG if config is None else config
    p2 = _phase2_cfg(config)
    if key == 'use_intraday_15m':
        return [p2.get('intraday_dir', 'history_15m')]
    return [(config.get('funding_rate') or {}).get('drive_dir', 'funding_rate'),
            (config.get('open_interest') or {}).get('drive_dir', 'open_interest'),
            p2.get('metrics_dir', 'futures_metrics')]


def _phase2_detect(key: str, config: Optional[dict] = None, mount: bool = False) -> bool:
    """'auto' ← True إن وُجد أيّ مجلد من مجلدات المفتاح (غير فارغ) تحت الجذر، والوحدة قابلة للاستيراد."""
    root = _phase2_root(config, mount=mount)
    if root is None:
        return False
    found = False
    for sub in _phase2_dirs(key, config):
        d = root / sub
        try:
            if d.is_dir() and next(d.iterdir(), None) is not None:
                found = True
                break
        except OSError:
            continue
    return bool(found and _phase2_module(config) is not None)


def _phase2_on(config: Optional[dict], key: str) -> bool:
    """USE_INTRADAY_15M / USE_FUTURES_METRICS الفعّال: True/False صريح، أو 'auto'/None ← كشف المجلد."""
    v = _phase2_cfg(config).get(key, False)
    if v in ('auto', None):
        return _phase2_detect(key, config)
    return bool(v) and _phase2_module(config) is not None


def resolve_phase2_toggles(config: Optional[dict] = None, verbose: bool = True) -> Dict[str, bool]:
    """يثبّت 'auto' إلى True/False **قبل** بناء البيانات (يُستدعى في بداية build_dataset*) — فتتطابق
    feature_order وما تُضيفه الخطافات فعلاً حتى لو رُكِّب Drive بعد اشتقاق الميزات أول مرة."""
    config = CONFIG if config is None else config
    p2 = config.get('phase2_data')
    if not isinstance(p2, dict):
        return {}
    changed, res = False, {}
    for key in _PHASE2_KEYS:
        v = p2.get(key, False)
        if v in ('auto', None):
            v = _phase2_detect(key, config, mount=True)
            p2[key] = v
            changed = True
        elif v and _phase2_module(config) is None:
            print(f"⚠️ {key}=True لكن tools/intraday_features.py غير موجود (أضف مسار المستودع إلى "
                  f"phase2_data['module_dirs']) — يُعطَّل.")
            p2[key] = v = False
            changed = True
        res[key] = bool(v)
    if changed:
        config['feature_order'] = None          # يُعاد اشتقاقها بالمفاتيح المثبّتة
    if verbose:
        print(f"🧩 المرحلة ٢: USE_INTRADAY_15M={res.get('use_intraday_15m')} | "
              f"USE_FUTURES_METRICS={res.get('use_futures_metrics')} | الجذر={_phase2_root(config)}")
    return res


def require_phase2_data(keys=_PHASE2_KEYS, config: Optional[dict] = None) -> Dict[str, bool]:
    """يثبّت مفاتيح المرحلة ٢ على True **أو يرفع خطأً**: الوحدة tools/intraday_features.py وكل مجلدات كل مفتاح
    (history_15m؛ funding_rate/open_interest/futures_metrics) يجب أن توجد غير فارغة تحت الجذر.

    لماذا: مع 'auto' يؤدّي غياب مجلد (Drive لم يُركَّب، اسم مختلف) أو الوحدة إلى False بصمت، فتُبنى البيانات بـ23 ميزة
    بدل 43، أو بمجلد ناقص تُبنى بأعمدة أصفار — بناء ساعات لا يُكتشف خطؤه إلا عند التدريب. إعداد 1h_s8 يستدعيها.
    """
    config = CONFIG if config is None else config
    problems = []
    if _phase2_module(config) is None:
        problems.append("tools/intraday_features.py غير موجود (cwd أو phase2_data['module_dirs'])")
    root = _phase2_root(config, mount=True)
    if root is None:
        problems.append("جذر المجلدات غير موجود (phase2_data['data_root'] أو MyDrive)")
    else:
        for key in keys:
            for sub in _phase2_dirs(key, config):
                d = root / sub
                try:
                    ok = d.is_dir() and next(d.iterdir(), None) is not None
                except OSError:
                    ok = False
                if not ok:
                    problems.append(f"{key}: المجلد {d} غير موجود أو فارغ")
    if problems:
        raise RuntimeError("❌ بيانات المرحلة ٢ مطلوبة لهذا الإعداد وغير متاحة — لا بناء بميزات ناقصة:\n   • "
                           + "\n   • ".join(problems))
    p2 = config.setdefault('phase2_data', {})
    for key in keys:
        p2[key] = True
    config['feature_order'] = None
    return {key: True for key in keys}


def _phase2_cached(key: tuple, fn):
    with _PHASE2_LOCK:
        if key in _PHASE2_CACHE:
            return _PHASE2_CACHE[key]
    val = fn()
    with _PHASE2_LOCK:
        _PHASE2_CACHE[key] = val
        while len(_PHASE2_CACHE) > _PHASE2_CACHE_MAX:
            _PHASE2_CACHE.pop(next(iter(_PHASE2_CACHE)))
    return val


def _phase2_file(config: dict, sub: str, symbol: str, pattern: str = "{name}.csv.gz") -> Optional[Path]:
    root = _phase2_root(config, mount=True)
    mod = _phase2_module(config)
    return None if (root is None or mod is None) else mod.find_file(root / sub, symbol, pattern)


def load_intraday_daily_frame(symbol: str, config: Optional[dict] = None) -> Optional[pd.DataFrame]:
    """ميزات يومية من history_15m/<S>.csv.gz (NaN حيث لا تغطية) — مخزّنة مؤقتاً لكل رمز، فتشترك فيها
    خطافات EFF_RATIO/VWAP_DEVIATION/VOL_CONC/ITD_ بقراءة ملف واحدة."""
    config = CONFIG if config is None else config
    p2 = _phase2_cfg(config)
    path = _phase2_file(config, p2.get('intraday_dir', 'history_15m'), symbol,
                        p2.get('intraday_pattern', '{name}.csv.gz'))
    if path is None:
        return None
    mod = _phase2_module(config)
    kw = dict(min_coverage=float(p2.get('min_coverage', 0.75)),
              trades_z_window=int(p2.get('trades_z_window', 30)), topk=int(p2.get('topk_bars', 4)))
    key = ('itd', symbol, str(path), path.stat().st_mtime, tuple(sorted(kw.items())))
    return _phase2_cached(key, lambda: mod.daily_intraday_features(mod.read_klines(path), **kw))


def _phase2_intraday_aligned(symbol: str, index, tf: str, config: dict) -> Optional[pd.DataFrame]:
    daily = load_intraday_daily_frame(symbol, config)
    if daily is None:
        return None
    mod = _phase2_module(config)
    return mod.align_daily_to_bars(daily, index, tf, pd.Timedelta(_phase2_cfg(config).get('max_age', '1D')))


def _phase2_slot_features(dfs: Dict[str, pd.DataFrame], symbol: str, config: dict,
                          col: str, flag: str) -> Dict[str, pd.DataFrame]:
    """فتحة قائمة (EFF_RATIO_24H/VWAP_DEVIATION/VOL_CONC_HHI) من شموع 15m الحقيقية: قيمة اليوم المكتمل
    الأخير عند إغلاق الشمعة (بلا تسرّب)، NaN ← 0 + علم التوفّر."""
    out: Dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        d = df.copy()
        aligned = _phase2_intraday_aligned(symbol, d.index, tf, config)
        vals = aligned[col] if aligned is not None else pd.Series(np.nan, index=d.index)
        d[flag] = vals.notna().astype('float64').to_numpy()
        d[col] = vals.fillna(0.0).to_numpy()
        out[tf] = d
    return out


def intraday_15m_columns(config: Optional[dict] = None) -> List[str]:
    """أعمدة ITD_ الجديدة (نسبة taker، الصفقات، التقلّب المحقَّق، تركّز أعلى k شموع) — حين USE_INTRADAY_15M."""
    config = CONFIG if config is None else config
    if not _phase2_on(config, 'use_intraday_15m'):
        return []
    return ['ITD_TAKER_BUY_RATIO', 'ITD_TRADES_LOG', 'ITD_TRADES_Z', 'ITD_RVOL', 'ITD_VOL_TOPK',
            'ITD_available']


def add_intraday_15m_features(dfs: Dict[str, pd.DataFrame], symbol: str,
                              config: Optional[dict] = None) -> Dict[str, pd.DataFrame]:
    """يُلحق أعمدة ITD_ من history_15m — مُعطَّل ⇒ ``dfs`` كما هي."""
    config = CONFIG if config is None else config
    cols = intraday_15m_columns(config)
    if not cols:
        return dfs
    out: Dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        d = df.copy()
        aligned = _phase2_intraday_aligned(symbol, d.index, tf, config)
        for c in cols:
            if c == 'ITD_available':
                continue
            vals = aligned[c] if aligned is not None else pd.Series(np.nan, index=d.index)
            d[c] = vals.fillna(0.0).to_numpy()
        d['ITD_available'] = (aligned['ITD_available'].fillna(0.0).to_numpy()
                              if aligned is not None else 0.0)
        out[tf] = d
    return out


def futures_metrics_columns(config: Optional[dict] = None) -> List[str]:
    """أعمدة نسب long/short وtaker من futures_metrics — حين USE_FUTURES_METRICS."""
    config = CONFIG if config is None else config
    if not _phase2_on(config, 'use_futures_metrics'):
        return []
    return ['LSR_TOP_ACCT', 'LSR_TOP_POS', 'LSR_GLOBAL', 'LSR_GLOBAL_chg_1', 'TAKER_LSR_1d', 'MET_available']


def _phase2_futures_frames(symbol: str, config: dict):
    """(funding, oi_series, metrics) الخام لرمز من مجلدات Drive (None لما لا يوجد)، مخزّنة مؤقتاً."""
    mod = _phase2_module(config)
    p2 = _phase2_cfg(config)
    fr_dir = (config.get('funding_rate') or {}).get('drive_dir', 'funding_rate')
    oi_dir = (config.get('open_interest') or {}).get('drive_dir', 'open_interest')
    paths = (_phase2_file(config, fr_dir, symbol), _phase2_file(config, oi_dir, symbol),
             _phase2_file(config, p2.get('metrics_dir', 'futures_metrics'), symbol))
    key = ('fut', symbol) + tuple((str(p), p.stat().st_mtime) if p else None for p in paths)

    def _load():
        fr = mod.read_archive(paths[0], ['funding_rate']) if paths[0] else None
        met = mod.read_archive(paths[2], list(mod.METRICS_COLS)) if paths[2] else None
        if paths[1]:
            oi = mod.read_archive(paths[1], ['open_interest'])['open_interest']
        elif met is not None:
            oi = met['sum_open_interest']
        else:
            oi = None
        return fr, oi, met
    return _phase2_cached(key, _load)


def _add_funding_oi_features_phase2(dfs: Dict[str, pd.DataFrame], symbol: str, config: dict,
                                    use_fr: bool, use_oi: bool) -> Dict[str, pd.DataFrame]:
    """نسخة المرحلة ٢ من add_funding_oi_features: كل قيمة = آخر ما هو متاح عند **إغلاق** الشمعة
    (لا عند فتحها كالنسخة القديمة)، مع مجاميع التمويل 1d/3d وأعمدة futures_metrics وأعلام توفّر."""
    mod = _phase2_module(config)
    fr_raw, oi_raw, met_raw = _phase2_futures_frames(symbol, config)
    p2 = _phase2_cfg(config)
    max_age = pd.Timedelta(p2.get('max_age', '1D'))
    z_win = int((config.get('funding_rate') or {}).get('zscore_window', 90))
    oi_k = int((config.get('open_interest') or {}).get('change_period', 1))
    met_cols = futures_metrics_columns(config)
    out: Dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        d = df.copy()
        parts = []
        if use_fr:
            parts.append(mod.funding_bar_features(fr_raw, d.index, tf, zscore_window=z_win, max_age=max_age))
        if use_oi:
            parts.append(mod.oi_bar_features(oi_raw, d.index, tf, change_bars=oi_k, max_age=max_age))
        if met_cols:
            parts.append(mod.metrics_bar_features(met_raw, d.index, tf, max_age=max_age))
        for part in parts:
            flags = [c for c in part.columns if c.endswith('_available')]
            filled = mod.fill_neutral(part, flags)
            for c in filled.columns:
                d[c] = filled[c].to_numpy()
        out[tf] = d
    return out
