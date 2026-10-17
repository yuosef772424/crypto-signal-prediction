"""
PURPOSE:  Optional cross-sectional normalisation of regression targets across coins at the same timestamp (Qlib CSZScoreNorm/CSRankNorm spirit) and its inverse.
TAGS:     cross-sectional normalization, cross_sectional_normalize, invert_cross_sectional, zscore, rank, targets
PITFALLS: Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.

## 15-ب) تطبيع مقطعي عبر الأصول (اختياري — بروح Qlib CSZScoreNorm/CSRankNorm)
"""
# @title
"""
تطبيع مقطعي (cross-sectional) لأهداف الانحدار عبر الأصول — بروح معالجات
Qlib ``CSZScoreNorm``/``CSRankNorm`` على حقل ``label`` (ليست محاكاة حرفية
لكودها، بل نفس الفكرة مُعاد بناؤها هنا).

**الفرق عن كل تطبيع آخر في هذا الدفتر:** التطبيع في خلايا التطبيع (٩) وخلية
النوافذ (١١) **زمني** — كل عيّنة تُطبَّع بإحصاء *تاريخها هي* (نافذتها أو آخر
إغلاقها). هذا التطبيع **مقطعي**: يجمع عيّنات *كل الأصول* التي تشارك نفس
الطابع الزمني، ويقيس كل عيّنة نسبة إلى توزيع عوائد *أقرانها في تلك اللحظة*.

فرق عملي: عملة بتقلّب يومي 1% وأخرى بتقلّب 8% — عائد خام 2% يعني حركة قوية
جداً للأولى وعادية للثانية، لكن العائد الخام وحده (reg_target_mode='return')
لا يميّز بينهما. التطبيع المقطعي يقيس كلتيهما نسبة إلى توزيع العوائد الفعلي
في نفس اليوم عبر كل الأصول، فتصير "1.2" تعني «أفضل من الأقران بمقدار 1.2
انحراف معياري ذلك اليوم» بصرف النظر عن مستوى تقلّب العملة نفسها.
"""


def cross_sectional_normalize(dataset: Dict, heads: Optional[List[str]] = None,
                              method: Optional[str] = None,
                              min_assets: Optional[int] = None,
                              clip: Optional[float] = None,
                              config: Optional[dict] = None,
                              verbose: bool = True) -> Dict:
    """يُطبِّع رؤوس انحدار عائدية مقطعياً عبر الأصول عند كل لحظة زمنية.

    ⚠️ **تغيير دلالي، لا مجرد تحسين مقياس:** الرأس بعد هذا التطبيع يتنبأ بأداء
    *نسبي* (تفوّق/تأخّر عن الأقران في نفس اللحظة) لا بعائد *مطلق* — مناسب
    لاستراتيجية اختيار الأفضل بين عدة عملات في نفس اللحظة، لا للسؤال المطلق
    "كم سيتحرّك BTC غداً؟". للتفسير المطلق استخدم عوائد ``reg_target_mode=
    'return'`` وحدها بلا هذه الخطوة.

    ⚠️ **غير متاح حرفياً في التنبؤ الحيّ:** حتى مع ``method='zscore'``
    (القابل للعكس تاريخياً عبر :func:`invert_cross_sectional`)، عكس تنبؤ
    *مستقبلي* إلى عائد مطلق يحتاج متوسط/انحراف عوائد الأقران في تلك اللحظة —
    وهي غير معروفة قبل إغلاقها. هذا تحديداً سبب كون الرأس بعد هذا التطبيع
    إشارة ترتيب حيّة، لا توقعاً مطلقاً حيّاً.

    Args:
        heads: رؤوس الانحدار المطلوب تطبيعها (افتراضياً كل رؤوس reg المُفعَّلة).
        method: ``'zscore'`` (افتراضي، قابل للعكس عبر :func:`invert_cross_sectional`)
            أو ``'rank'`` (رتبة موحّدة إلى ``[-1, 1]``؛ أقوى أمام قيمة شاذة
            واحدة داخل اليوم، لكن بلا عكس دقيق لعائد مطلق — ترتيبي بالتصميم).
        min_assets: أقل عدد أصول عند نفس اللحظة ليُحسب التطبيع؛ دون ذلك يبقى
            العائد الخام كما هو — مجموعة أقران أصغر من هذا لا تعطي إحصاءً
            موثوقاً (انحراف معياري على عيّنتين أو ثلاث ضجيج بحت لا إشارة).
        clip: قصّ بعد ``'zscore'`` فقط (``'rank'`` محصورة أصلاً في ``[-1, 1]``).

    Returns:
        نسخة من ``dataset`` (**لا يُعدَّل الأصل**): رؤوس ``heads`` مُستبدَلة
        بالقيم المُطبَّعة، مع ``dataset['cs_norm_stats']`` — لكل رأس: الطريقة
        والحدّ الأدنى والقصّ، وإحصاء كل لحظة طُبِّعت (لازم لعكس ``'zscore'``
        عبر :func:`invert_cross_sectional`؛ اللحظات التي لم تُطبَّع — أصول
        أقل من ``min_assets`` — غائبة عن الإحصاء عمداً، فتبقى عائداً خاماً
        عند العكس أيضاً).

    يفترض ``reg_target_mode='return'`` (العائد المباشر) — تطبيقه على
    ``'window_scale'`` يخلط مرجعين مختلفي المعنى (IQR نافذة الأصل الواحد +
    توزيع الأقران في نفس اللحظة) فيُرفَض بخطأ صريح بدل نتيجة مضلِّلة بصمت.
    """
    config = CONFIG if config is None else config
    if config.get('reg_target_mode', 'return') != 'return':
        raise ValueError(
            "cross_sectional_normalize يتطلب reg_target_mode='return' — "
            "تطبيقه على 'window_scale' يخلط مرجع IQR النافذة بتوزيع الأقران.")

    cs_cfg = config.get('cs_norm', {}) or {}
    heads = list(heads) if heads is not None else get_reg_heads(config)
    method = method or cs_cfg.get('method', 'zscore')
    min_assets = int(min_assets if min_assets is not None else cs_cfg.get('min_assets', 5))
    clip = float(clip if clip is not None else cs_cfg.get('clip', 5.0))
    if method not in ('zscore', 'rank'):
        raise ValueError(f"method غير معروفة: '{method}' (المتاح: zscore, rank)")

    heads = [h for h in heads if h in dataset]
    out = dict(dataset)
    if not heads or 'last_candles' not in dataset or len(dataset['last_candles']) == 0:
        out['cs_norm_stats'] = dict(dataset.get('cs_norm_stats', {}))
        return out

    ts_all = np.asarray(dataset['last_candles'])[:, TS_COL].astype('int64')
    order = np.argsort(ts_all, kind='stable')
    ts_sorted = ts_all[order]
    uniq_ts, start_pos, counts = np.unique(ts_sorted, return_index=True, return_counts=True)

    all_stats: Dict[str, dict] = dict(dataset.get('cs_norm_stats', {}))
    for h in heads:
        raw = np.asarray(dataset[h], dtype='float64')
        y_sorted = raw[order]
        normed = y_sorted.copy()
        by_ts: Dict[int, Tuple[float, float, int]] = {}
        n_small = 0
        for t, pos, cnt in zip(uniq_ts, start_pos, counts):
            sl = slice(int(pos), int(pos) + int(cnt))
            grp = y_sorted[sl]
            if cnt < min_assets:
                n_small += cnt
                continue
            if method == 'rank':
                ranks = grp.argsort(kind='stable').argsort(kind='stable').astype('float64')
                denom = max(cnt - 1, 1)
                normed[sl] = (ranks / denom) * 2.0 - 1.0        # [-1, 1]، بلا حاجة لقصّ
            else:
                mu, sd = float(grp.mean()), float(grp.std())
                if sd < EPS:
                    normed[sl] = 0.0
                else:
                    normed[sl] = np.clip((grp - mu) / sd, -clip, clip)
                by_ts[int(t)] = (mu, sd, int(cnt))
        result = np.empty_like(raw)
        result[order] = normed
        out[h] = result.astype('float32')
        all_stats[h] = {'method': method, 'min_assets': min_assets, 'clip': clip, 'by_ts': by_ts}
        if verbose:
            total = len(raw)
            print(f"📐 [{h}] تطبيع مقطعي ({method}): "
                  f"{total - n_small:,}/{total:,} عيّنة طُبِّعت نسبة لأقرانها "
                  f"(الحدّ الأدنى {min_assets} أصلاً) | {n_small:,} بقيت عائداً خاماً "
                  f"(لحظات بأصول أقل من الحدّ)")
    out['cs_norm_stats'] = all_stats
    return out


def invert_cross_sectional(preds: np.ndarray, timestamps: np.ndarray, head: str,
                           dataset: Dict) -> np.ndarray:
    """يعكس تنبؤات مُطبَّعة مقطعياً (``method='zscore'`` فقط) إلى عائد خام.

    يستخدم إحصاء *نفس اللحظات* المحفوظ في ``dataset['cs_norm_stats'][head]``
    عند بناء بيانات التدريب — مناسب لتقييم نموذج على بيانات تاريخية (الإحصاء
    معروف بأثر رجعي)، **لا للتنبؤ الحيّ**: عائد الأقران الفعلي "اليوم" غير
    معروف قبل إغلاق اليوم نفسه — راجع تنبيه :func:`cross_sectional_normalize`.

    لحظة لم تُطبَّع أصلاً (أصول أقل من ``min_assets`` عند البناء) تُعاد كما
    هي (كانت عائداً خاماً بلا تطبيع، فتبقى كذلك). ``method='rank'`` ترتيبي
    بلا عكس دقيق فيُرفَض بخطأ صريح.
    """
    stats = dataset.get('cs_norm_stats', {}).get(head)
    if stats is None:
        raise ValueError(f"لا إحصاء تطبيع مقطعي محفوظ للرأس '{head}' — "
                         "طبّق cross_sectional_normalize أولاً.")
    if stats['method'] != 'zscore':
        raise ValueError(
            f"عكس دقيق متاح فقط لـ method='zscore' (المحفوظ: '{stats['method']}'). "
            "'rank' ترتيبي بالتصميم، بلا معنى لعكسه إلى عائد مطلق.")
    by_ts = stats['by_ts']
    preds = np.asarray(preds, dtype='float64')
    ts = np.asarray(timestamps).astype('int64')
    out = preds.copy()
    for i, t in enumerate(ts):
        rec = by_ts.get(int(t))
        if rec is not None:
            mu, sd, _ = rec
            out[i] = preds[i] * sd + mu
    return out
