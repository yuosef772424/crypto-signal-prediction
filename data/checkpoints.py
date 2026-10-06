"""
PURPOSE:  Per-asset resume checkpoints for build_dataset_from_loader, guarded by a fingerprint of every setting that changes the output.
TAGS:     checkpoints, resume, checkpoint_dir, fingerprint, clear_checkpoint, _checkpoint_fingerprint
PITFALLS: A setting that changes the data must enter _checkpoint_fingerprint, or an old checkpoint is silently merged with new-shape data. Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.
"""
# @title
"""
نقاط استئناف لكل أصل — تحمي من فقدان بيانات مُعالَجة عند انقطاع الجلسة
(نفاد الرام، انقطاع الاتصال، إعادة تشغيل يدوية) في ``build_dataset_from_loader``.

**الفكرة:** كل أصل يُحفَظ على القرص (Drive عادة) فور معالجته، لا فقط في نهاية
البناء كله. عند إعادة التشغيل، الأصول المحفوظة سابقاً تُحمَّل من القرص مباشرة
(بلا شبكة ولا إعادة حساب)، ويُعالَج فقط ما تبقّى.

⚠️ **الحماية من التقادم:** نقطة استئناف محفوظة بإعدادات مختلفة (نافذة، ميزات،
أهداف...) عن التشغيل الحالي خطر حقيقي — دمج بيانات بشكل قديم مع أخرى بشكل
جديد صامتاً. لهذا تُحفَظ **بصمة** الإعدادات المؤثّرة مع كل نقطة استئناف
وتُقارَن عند التحميل: أي اختلاف يُسقط نقطة الاستئناف (تُعاد معالجة ذلك الأصل)
بدل دمج غير متجانس بصمت.

**والبيانات نفسها جزء من البصمة** (:func:`_asset_checkpoint_fingerprint`): محتوى الملف الخام للعملة، وحجم/زمن
تعديل ملفات المرحلة ٢ التي تُقرأ لها، ومحتوى بيانات الأقران حين تُفعَّل ميزة عابرة للأصول أو سياق سوقي. كانت البصمة
للإعدادات وحدها، فإعادة جلب التاريخ إلى المجلد نفسه ثم البناء بنفس ``checkpoint_dir`` تُرجع المصفوفات القديمة بلا
أحدث العيّنات (docs/research/audit/r2_01_stale_pipeline_checkpoint.py). الثمن قراءة الملف الخام قبل الاستئناف
(ثوانٍ لكل عملة)؛ المُوفَّر هو الحساب الثقيل (المؤشرات والنوافذ والتطبيع).
أرشيفات المرحلة ١ القديمة (``load_funding_open_interest``، أرشيف كفاءة كوفمان المحلي) خارج البصمة: بعد تحديثها
استدعِ :func:`clear_checkpoint`.
"""


def _checkpoint_fingerprint(config: dict, features: List[str], tail: int = 0,
                            universe: Optional[List[str]] = None) -> str:
    """بصمة كل ما يؤثّر في شكل/معنى بيانات عملة واحدة مُعالَجة — تُحفَظ مع كل
    نقطة استئناف. أي تغيير في هذه القيم يُبطل نقاط الاستئناف القديمة تلقائياً.

    ``universe``: أسماء كل العملات المطلوبة — تدخل البصمة فقط حين تُفعَّل ميزة عابرة
    للأصول (قيمتها لعملة ما تتغيّر بتغيّر أقرانها)."""
    import hashlib
    parts = {
        'feature_order': list(features),
        'tf_order': list(config['tf_order']),
        'window_sizes': {tf: config['window_sizes'][tf] for tf in config['tf_order']},
        'forecast_horizon': config['forecast_horizon'],
        'targets': list(config['targets']),
        'enabled_heads': config.get('enabled_heads', {}),
        'scaler_type': config['scaler_type'],
        'stride': config['stride'],
        'reg_target_mode': config.get('reg_target_mode', 'return'),
        'market_context': config.get('market_context', {}),
        'funding_rate': config.get('funding_rate', {}),
        'open_interest': config.get('open_interest', {}),
        'tail': tail,
        # نوع تطبيع كل ميزة: تغيّر تصنيف ميزة (مثل أعلام *_available ← BINARY_FLAG) يُغيّر X المحفوظة.
        'feature_kinds': [classify_feature(f) for f in features],
        # 1: نافذة الإدخال win شمعة متتالية بالضبط (النوافذ العابرة لفجوة تُحذف).
        'window_contiguity': 1,
        # 2: هدف كل عيّنة من الشموع عند ts + 1..h بالضبط (عيّنات الفجوات تُحذف) — نقاط الاستئناف السابقة بُنيت
        #    بـ «الصف التالي» فتُبطَل.
        'label_contiguity': 2,
    }
    # يدخل البصمة فقط حين ≠ 1 — فلا تُبطَل نقاط استئناف بُنيت قبل وجود المفتاح.
    if float(config.get('reg_target_scale', 1.0)) != 1.0:
        parts['reg_target_scale'] = float(config['reg_target_scale'])
    # شبكة نهايات النوافذ تغيّر العيّنات فقط حين stride > 1 — فلا تُبطَل نقاط يومي stride=1.
    if int(config['stride']) > 1 and config.get('align_windows_to_grid', True):
        parts['window_grid'] = str(config.get('window_grid_anchor', '1970-01-01'))
    # الفريمات الأعلى بوضع 'closed' وتخزين X بدقّة أقلّ يغيّران X المحفوظة — يدخلان البصمة حين يؤثّران فقط، فلا تُبطَل نقاط
    # الاستئناف القائمة (فريم واحد: الوضع بلا أثر أصلاً).
    if _higher_tf_closed(config):
        parts['higher_tf_mode'] = 'closed'
    if config.get('x_storage_dtype'):
        parts['x_storage_dtype'] = str(config['x_storage_dtype'])
    # تطبيع الأعمدة السعرية بالتغيّر النسبي يغيّر X المحفوظة — يدخل البصمة حين ≠ الافتراضي فقط.
    if price_norm_mode(config) != PRICE_NORM_WINDOW:
        parts['price_norm_mode'] = price_norm_mode(config)
        parts['price_pct_clip'] = price_pct_clip(config)
    # فلاتر العيّنات تغيّر أيّ العيّنات تُحفَظ — تدخل البصمة حين تُستعمل فقط.
    if config.get('sample_filters'):
        parts['sample_filters'] = _sample_filters_signature(list(config['sample_filters']))
    # الميزات العابرة للأصول: نقاط الاستئناف القديمة من مسار التحميل تحمل قيمها
    # المحايدة الثابتة (قبل _load_cross_asset_frames) ⇒ يجب أن تُبطَل.
    cross = {k: config.get(k) for k in ('momentum_rank', 'market_breadth', 'momentum_orth_natr')
             if (config.get(k) or {}).get('enabled', False)}
    if cross:
        parts['cross_asset'] = cross
        parts['universe'] = sorted(universe or [])
        if str(config.get('cross_asset_dtype') or 'float64') != 'float64':     # يدخل حين ≠ الافتراضي فقط
            parts['cross_asset_dtype'] = str(config['cross_asset_dtype'])
    # المرحلة ٢ تُدخَل في البصمة فقط حين تُفعَّل — فلا تُبطَل نقاط الاستئناف القديمة بلا داعٍ.
    _p2 = config.get('phase2_data') or {}
    if any(_p2.get(k) not in (False, None, 'auto') for k in ('use_intraday_15m', 'use_futures_metrics')):
        parts['phase2_data'] = {k: v for k, v in _p2.items() if k != 'module_dirs'}
    blob = json.dumps(parts, sort_keys=True, default=str).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()[:16]


def _frame_signature(obj) -> str:
    """بصمة محتوى (sha256) لإطار أو سلسلة أو قاموس متداخل منها: القيم والفهرس وأسماء الأعمدة.
    ``pd.util.hash_pandas_object`` متّجهة، فتكلفتها أجزاء من الثانية لعشرات آلاف الصفوف."""
    import hashlib
    h = hashlib.sha256()

    def _walk(o):
        if isinstance(o, dict):
            for k in sorted(o, key=str):
                h.update(f"<{k}>".encode('utf-8'))
                _walk(o[k])
        elif isinstance(o, (pd.DataFrame, pd.Series)):
            h.update(repr((o.shape, list(o.columns) if isinstance(o, pd.DataFrame) else o.name)).encode('utf-8'))
            h.update(pd.util.hash_pandas_object(o, index=True).to_numpy().tobytes())
        else:
            h.update(repr(o).encode('utf-8'))
    _walk(obj)
    return h.hexdigest()[:16]


def _phase2_asset_signature(name: str, config: dict) -> Optional[list]:
    """(مجلد، ملف، حجم، زمن تعديل) لكل ملف مرحلة ٢ يُقرأ لهذه العملة — ``None`` إن كانت المرحلة ٢ معطّلة."""
    keys = [k for k in _PHASE2_KEYS if (_phase2_cfg(config).get(k) is True)]
    if not keys:
        return None
    p2 = _phase2_cfg(config)
    out = []
    for key in keys:
        for sub in _phase2_dirs(key, config):
            pattern = (p2.get('intraday_pattern', '{name}.csv.gz') if key == 'use_intraday_15m'
                       else '{name}.csv.gz')
            path = _phase2_file(config, sub, name, pattern)
            st = path.stat() if path is not None else None
            out.append([sub, path.name if path is not None else None,
                        st.st_size if st else None, st.st_mtime_ns if st else None])
    return out


def _asset_checkpoint_fingerprint(base_fp: str, raw_df: pd.DataFrame, name: str, config: dict,
                                  shared_sig: Optional[str] = None) -> str:
    """بصمة نقطة استئناف عملة واحدة = بصمة الإعدادات + محتوى ملفها الخام (بعد tail) + ملفات المرحلة ٢ لها +
    ``shared_sig`` (بيانات الأقران للميزات العابرة للأصول والسياق السوقي، تُحسب مرة لكل بناء)."""
    import hashlib
    blob = json.dumps([base_fp, _frame_signature(raw_df), _phase2_asset_signature(name, config), shared_sig],
                      default=str).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()[:16]


def _checkpoint_paths(checkpoint_dir, name: str) -> Tuple[Path, Path]:
    d = Path(checkpoint_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.npz", d / f"{name}.meta.json"


def _save_asset_checkpoint(checkpoint_dir, name: str, X_tf: Dict[str, np.ndarray],
                           y_t: Dict[str, np.ndarray], bases: np.ndarray,
                           last: np.ndarray, fingerprint: str) -> None:
    """يحفظ نتيجة أصل واحد على القرص — كتابة ذرّية (ملف مؤقت ثم إعادة تسمية)
    فلا يُخلَّف ملف نصفي تالف لو انقطعت الجلسة أثناء الحفظ نفسه."""
    npz_path, meta_path = _checkpoint_paths(checkpoint_dir, name)
    payload = {f"X__{tf}": arr for tf, arr in X_tf.items()}
    payload.update({f"y__{h}": arr for h, arr in y_t.items()})
    payload["bases"] = bases
    payload["last"] = last

    # ✅ الاسم المؤقّت ينتهي بـ .npz أيضاً: np.savez تُلحق ".npz" تلقائياً إن لم
    # ينتهِ الاسم بها فعلاً — ".npz.tmp" كان سيُصبح فعلياً ".npz.tmp.npz"،
    # فيفشل replace() لاحقاً (يبحث عن اسم لم يُكتَب قط).
    tmp_npz = npz_path.with_name(f"{name}.tmp.npz")
    np.savez(tmp_npz, **payload)
    tmp_npz.replace(npz_path)

    tmp_meta = meta_path.with_suffix(".json.tmp")
    tmp_meta.write_text(json.dumps({"fingerprint": fingerprint, "n_samples": int(bases.shape[0])}))
    tmp_meta.replace(meta_path)


def _load_asset_checkpoint(checkpoint_dir, name: str, fingerprint: str):
    """يحمّل نقطة استئناف أصل — ``None`` إن لم تُوجد، أو وُجدت ببصمة مختلفة
    (إعدادات تغيّرت)، أو تعذّرت قراءتها (ملف تالف من انقطاع سابق)."""
    npz_path, meta_path = _checkpoint_paths(checkpoint_dir, name)
    if not (npz_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") != fingerprint:
            return None
        with np.load(npz_path) as z:
            X_tf = {k[3:]: z[k] for k in z.files if k.startswith("X__")}
            y_t = {k[3:]: z[k] for k in z.files if k.startswith("y__")}
            bases = z["bases"]
            last = z["last"]
        return X_tf, y_t, bases, last
    except Exception:                                            # noqa: BLE001
        return None


def clear_checkpoint(checkpoint_dir, names: Optional[List[str]] = None) -> int:
    """يحذف نقاط استئناف محفوظة — كل الأصول (``names=None``) أو أصول بعينها.
    يُرجع عدد الأصول المحذوفة.

    بصمة الإعدادات (:func:`_checkpoint_fingerprint`) تُبطل نقاط الاستئناف
    تلقائياً عند تغيّر ما يؤثّر في البيانات — هذه الدالة للتحكّم اليدوي
    الإضافي (مثلاً: إعادة تحميل عملة واحدة تشكّ ببياناتها الخام).
    """
    d = Path(checkpoint_dir)
    if not d.exists():
        return 0
    removed = 0
    for npz in d.glob("*.npz"):
        name = npz.stem
        if names is not None and name not in names:
            continue
        npz.unlink(missing_ok=True)
        meta = d / f"{name}.meta.json"
        if meta.exists():
            meta.unlink()
        removed += 1
    return removed
