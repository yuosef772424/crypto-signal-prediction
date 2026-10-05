"""
PURPOSE:  Trade selection: keep only trades whose expected move exceeds the uncertainty (select_and_rank_trades) and print them.
TAGS:     select_and_rank_trades, print_trade_selection, edge vs uncertainty, min_edge_ratio, trade selection
PITFALLS: A self-test of the per-target fairness runs at load. Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.

## 1️⃣9️⃣ 🎯 اختيار وتصنيف الصفقات: التحرك المتوقع مقابل عدم اليقين (Edge vs Uncertainty)

**الفكرة**: صفقة "تستحق الثقة" فقط إن كان **التحرك المتوقع** (المسافة بين
السعر المتوقع وسعر الدخول) **أكبر من عدم اليقين** المصاحب للتنبؤ — أي أن
الإشارة تتجاوز الضجيج الإحصائي. أي صفقة يكون فيها التحرك المتوقع أصغر من (أو
يساوي) عدم اليقين تُستبعد تلقائياً بالكامل لأنها غير موثوقة (النموذج نفسه غير
متأكد أن الاتجاه المتوقع حقيقي وليس ضوضاء).

`select_and_rank_trades` تعمل تلقائياً على:
- ناتج `build_flat_dataframe` (بيانات اختبار/باكتست تاريخية بوحدات حقيقية)
- ناتج `build_latest_table` / `predict_latest_v4` / `predict_latest_all_assets`
  (تقرير التداول الحي بالنسبة المئوية)

وتكتشف الأعمدة المناسبة تلقائياً حسب مصدر البيانات (أو تقبلها صراحةً)، ثم
تُرتّب الصفقات الناجية من الفلتر حسب: **الثقة** فقط، أو **حجم التحرك** فقط،
أو **الاثنين معاً** (نقاط مركّبة تمنع صفقة قوية في معيار واحد وضعيفة جداً في
الآخر من الصعود للقمة ظلماً).
"""
# ═══════════════════════════════════════════════════════════════════════════
# 🎯 select_and_rank_trades: فلترة الصفقات (تحرك متوقع > عدم يقين) + ترتيبها
# ═══════════════════════════════════════════════════════════════════════════

# كل عنصر: (عمود التحرك, عمود عدم اليقين, عمود الثقة) — يجب أن تكون الثلاثة
# بنفس "الفضاء" (وحدات حقيقية معاً، أو نسبة مئوية معاً) حتى تُقارَن بصحة.
_COMPATIBLE_TRADE_COLUMNS = [
    ('predicted_change', 'uncertainty', 'confidence'),   # build_flat_dataframe (وحدات حقيقية)
    ('move_%', 'uncertainty_%', 'confidence_%'),         # build_latest_table / predict_latest_v4 (نسبة مئوية)
]


def _detect_trade_columns(df: pd.DataFrame, move_col=None, uncertainty_col=None, confidence_col=None):
    '''يختار ثلاثية (move, uncertainty, confidence) متوافقة: إمّا الثلاثة صراحةً معاً، أو اكتشاف تلقائي لزوج معروف.'''
    any_explicit = any(c is not None for c in (move_col, uncertainty_col, confidence_col))
    if any_explicit:
        missing = [n for n, c in (('move_col', move_col), ('uncertainty_col', uncertainty_col),
                                   ('confidence_col', confidence_col)) if c is None]
        if missing:
            raise ValueError(f"عند تحديد أي عمود يدوياً يجب تحديد الثلاثة معاً: {missing} مفقودة.")
        for c in (move_col, uncertainty_col, confidence_col):
            if c not in df.columns:
                raise KeyError(f"العمود '{c}' غير موجود. الأعمدة المتاحة: {list(df.columns)}")
        return move_col, uncertainty_col, confidence_col

    for m, u, c in _COMPATIBLE_TRADE_COLUMNS:
        if m in df.columns and u in df.columns and c in df.columns:
            return m, u, c

    raise KeyError(
        "تعذّر اكتشاف أعمدة (تحرك / عدم يقين / ثقة) متوافقة تلقائياً ضمن الصيغ المعروفة "
        f"{_COMPATIBLE_TRADE_COLUMNS}. مرّرها صراحةً (الثلاثة معاً) عبر "
        "move_col / uncertainty_col / confidence_col.\n"
        f"الأعمدة المتاحة فعلياً: {list(df.columns)}"
    )


def select_and_rank_trades(
    df: pd.DataFrame,
    move_col: Optional[str] = None,
    uncertainty_col: Optional[str] = None,
    confidence_col: Optional[str] = None,
    min_edge_ratio: float = 1.0,
    sort_by: str = 'both',
    ascending: bool = False,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    '''
    يُبقي فقط الصفقات التي |التحرك المتوقع| > عدم اليقين × min_edge_ratio (الإشارة
    أكبر من الضجيج)، ويستبعد كل الباقي تماماً (لا يظهر إطلاقاً في الناتج). ثم
    يرتّب الصفقات الناجية حسب sort_by.

    تعمل تلقائياً على ناتج build_flat_dataframe (باكتست) أو build_latest_table/
    predict_latest_v4/predict_latest_all_assets (لايف) — تكتشف الأعمدة المناسبة
    تلقائياً، أو مرّرها صراحةً (الثلاثة معاً) إن كان لديك تسمية مختلفة.

    Args:
        min_edge_ratio: الحد الأدنى لنسبة (|move| / عدم اليقين) لقبول الصفقة.
            1.0 (افتراضي) = التحرك المتوقع أكبر من عدم اليقين بمقدار مرة واحدة
            على الأقل. أكبر من 1.0 أكثر تشدداً (هامش أمان إضافي، صفقات أقل
            وأنقى)، أصغر من 1.0 أكثر تساهلاً (صفقات أكثر).
        sort_by:
            'confidence' -> الأعلى ثقة أولاً.
            'move'       -> الأكبر تحركاً متوقعاً أولاً (بالقيمة المطلقة).
            'both'       -> (افتراضي) نقاط مركّبة = متوسط الرتبة المئوية للثقة
                            والرتبة المئوية لحجم التحرك؛ يمنع صفقة متفوقة في
                            معيار واحد فقط من التصدّر ظلماً.
            'edge'       -> الأعلى في نسبة (|move| / عدم اليقين) نفسها (أنقى صفقة
                            إحصائياً، بغض النظر عن حجم التحرك أو الثقة المطلقة).
        top_n: أعد فقط أفضل top_n صفقة بعد الترتيب (None = كل الصفقات الناجية).

    Returns:
        DataFrame (نسخة مُرشَّحة ومرتَّبة من df) + أعمدة إضافية:
        edge_ratio (|move|/عدم اليقين)، direction ('up'/'down')، rank_score.
        فارغ إن لم تنجُ أي صفقة من الفلتر.
    '''
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    work = df.copy()
    if 'kind' in work.columns:
        work = work[work['kind'] == 'continuous'].copy()
    if work.empty:
        return work

    mcol, ucol, ccol = _detect_trade_columns(work, move_col, uncertainty_col, confidence_col)

    move = pd.to_numeric(work[mcol], errors='coerce')
    unc = pd.to_numeric(work[ucol], errors='coerce')
    conf = pd.to_numeric(work[ccol], errors='coerce')

    valid = np.isfinite(move) & np.isfinite(unc) & (unc > 0)
    work, move, unc, conf = work.loc[valid].copy(), move[valid], unc[valid], conf[valid]

    edge_ratio = move.abs() / unc
    keep = edge_ratio > float(min_edge_ratio)
    work = work.loc[keep].copy()

    if work.empty:
        print(f"⚠️ لا توجد صفقات يتجاوز فيها |التحرك المتوقع| عدم اليقين × {min_edge_ratio} (0 من {int(valid.sum())}).")
        return work

    work['edge_ratio'] = edge_ratio.loc[keep].values
    work['direction'] = np.where(move.loc[keep].values > 0, '▲', '▼')

    conf_kept, move_kept = conf.loc[keep], move.loc[keep].abs()
    if sort_by == 'confidence':
        work['rank_score'] = conf_kept.values
    elif sort_by == 'move':
        work['rank_score'] = move_kept.values
    elif sort_by == 'edge':
        work['rank_score'] = work['edge_ratio'].values
    elif sort_by == 'both':
        # ✅ رتبة مئوية *داخل كل هدف على حدة* لا على المجموع المُجمَّع: الثقة/الحجم
        # لكل هدف (close/high/low) لهما رأس NIG/تصنيف منفصل تماماً، فمقياسهما غير
        # قابل للمقارنة المباشرة بين الأهداف (اكتُشف عملياً: هدف "high" أظهر ثقة
        # أعلى بكثير من "close"/"low" رغم ترابط ضعيف بين ثقته ودقّته الفعلية —
        # راجع تقرير المعايرة — فرتبة مئوية مُجمَّعة عبر كل الأهداف معاً كانت
        # تُصعّد صفقات "high" ظلماً بلا علاقة بجودتها الفعلية). التجميع لكل هدف
        # على حدة يُطبَّق فقط حين تتعدّد الأهداف فعلياً في work['target']؛ هدف
        # واحد (الاستخدام المُوصى به في أمثلة القسم) يُعيد نفس السلوك القديم تماماً.
        if 'target' in work.columns and work['target'].nunique() > 1:
            groups = work['target'].values
            conf_pct = conf_kept.groupby(groups).rank(pct=True, na_option='bottom')
            move_pct = move_kept.groupby(groups).rank(pct=True, na_option='bottom')
        else:
            conf_pct = conf_kept.rank(pct=True, na_option='bottom')
            move_pct = move_kept.rank(pct=True, na_option='bottom')
        work['rank_score'] = ((conf_pct.values + move_pct.values) / 2.0)
    else:
        raise ValueError("sort_by يجب أن تكون إحدى: 'confidence' | 'move' | 'both' | 'edge'")

    work = work.sort_values('rank_score', ascending=ascending).reset_index(drop=True)
    return work.head(int(top_n)) if top_n is not None else work


def _test_select_and_rank_trades_per_target_fairness():
    # يثبت أن رتبة مئوية مجمَّعة عبر أهداف مختلفة لا تُرقّي هدفاً
    # واحداً مجرّد ارتفاع نطاق ثقته عن بقية الأهداف (بلا تحسُّن حقيقي في الدقة)،
    # بخلاف السلوك القديم الذي كان يُصعّد ذلك الهدف 100% من أعلى N صفقة بلا استثناء.
    rng = np.random.default_rng(0)
    n = 200
    rows = []
    for target, conf_level in (("high", 0.85), ("close", 0.30), ("low", 0.25)):
        move = rng.normal(0, 1, n)
        unc = np.abs(rng.normal(1, 0.2, n)) + 0.1
        conf = np.clip(conf_level + rng.normal(0, 0.05, n), 0.01, 0.99)
        for m, u, c in zip(move, unc, conf):
            rows.append({"target": target, "predicted_change": m, "uncertainty": u,
                        "confidence": c, "kind": "continuous"})
    df = pd.DataFrame(rows)
    selected = select_and_rank_trades(df, min_edge_ratio=1.0, sort_by='both', top_n=30)
    counts = selected['target'].value_counts()
    assert len(counts) == 3, (
        f"الرتبة المئوية المجمَّعة ما زالت تُرقّي هدفاً واحداً بسبب اختلاف مقياس الثقة (الموجود: {dict(counts)})")
    assert counts.max() <= 15, (
        f"هدف واحد يستحوذ أكثر من نصف أفضل 30 صفقة رغم تساوي الجودة الفعلية بين الأهداف الثلاثة في البيانات التركيبية (الموجود: {dict(counts)})")
    print("✅ select_and_rank_trades: الرتبة المئوية لـ sort_by='both' عند تعدّد الأهداف محسوبة داخل كل هدف على حدة، لا تُرقّي هدفاً واحداً بسبب تضخم ثقته الخام.")
    return True


_test_select_and_rank_trades_per_target_fairness()


# ═══════════════════════════════════════════════════════════════════════════
# 🖨️ print_trade_selection: عرض مضغوط لناتج select_and_rank_trades
# ═══════════════════════════════════════════════════════════════════════════

def _trade_accuracy_pct(series: pd.Series) -> Tuple[float, int]:
    '''معدّل الإصابة % من عمود ok/direction_correct/correct (يتجاهل None/NaN بأمان).'''
    s = series.dropna()
    if len(s) == 0:
        return float('nan'), 0
    return float(s.astype(bool).mean()) * 100.0, len(s)


def print_trade_selection(
    result: pd.DataFrame,
    original: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
    max_rows: Optional[int] = 20,
    indent: str = "  ",
) -> None:
    '''
    يطبع ناتج select_and_rank_trades بشكل مضغوط (رقم، تاريخ/عملة/هدف، اتجاه،
    ثقة، عدم يقين، تحرك، edge، سعر متوقع/دخول، صح/خطأ إن توفّر هدف فعلي).

    إن مُرِّر original (الـ DataFrame *قبل* الفلترة، بنفس عمود النتيجة
    direction_correct/correct/ok)، يُطبع أسفل الجدول مقارنة معدل الإصابة قبل
    وبعد الفلترة + Top-N المعروضة — لإظهار أداء الصفقات المعروضة مقارنةً بالمجموعة الأصلية.
    '''
    if result is None or result.empty:
        if title:
            print(title)
        print(indent + "(لا توجد صفقات تجتاز الفلتر)")
        return

    if title:
        print(title)

    show = result.head(max_rows) if max_rows else result
    mcol, ucol, ccol = _detect_trade_columns(show)

    out = pd.DataFrame(index=show.index)
    out['#'] = np.arange(1, len(show) + 1)
    if 'date' in show.columns and show['date'].notna().any():
        out['date'] = show['date'].map(lambda d: d.strftime('%Y-%m-%d %H:%M') if pd.notna(d) else '—')
    elif 'idx' in show.columns:
        out['#idx'] = show['idx']
    if 'asset' in show.columns:
        out['asset'] = show['asset']
    out['target'] = show['target']
    out['dir'] = show['direction']

    conf_vals = show[ccol].astype(float)
    conf_pct = conf_vals * 100.0 if conf_vals.max(skipna=True) <= 1.5 else conf_vals
    out['conf%'] = conf_pct.map(lambda v: _fmt_num(v, '.1f'))
    out['unc'] = show[ucol].map(lambda v: _fmt_num(v, '.4g'))
    out['move'] = show[mcol].map(lambda v: _fmt_num(v, '+.3f'))
    out['edge_x'] = show['edge_ratio'].map(lambda v: _fmt_num(v, '.2f'))
    if 'pred' in show.columns:
        out['pred'] = show['pred'].map(_fmt_price)
    if 'entry' in show.columns:
        out['entry'] = show['entry'].map(_fmt_price)

    ok_col = next((c for c in ('direction_correct', 'ok', 'correct') if c in show.columns), None)
    if ok_col:
        out['ok'] = show[ok_col].map(
            lambda v: '⏳' if (v is None or (isinstance(v, float) and np.isnan(v))) else ('✅' if bool(v) else '❌')
        )

    print("\n".join(indent + ln for ln in out.to_string(index=False).split("\n")))
    if max_rows and len(result) > max_rows:
        print(indent + f"… و{len(result) - max_rows} صفقة أخرى (max_rows=None لعرض الكل)")

    ok_col = next((c for c in ('direction_correct', 'correct', 'ok') if c in result.columns), None)
    if ok_col and original is not None and ok_col in original.columns:
        base_acc, base_n = _trade_accuracy_pct(original[ok_col])
        sel_acc, sel_n = _trade_accuracy_pct(result[ok_col])
        if base_n and sel_n:
            print(f"\n{indent}📈 معدل الإصابة: قبل الفلترة {base_acc:.1f}% (n={base_n})  ->  "
                  f"بعد الفلترة + Top-N {sel_acc:.1f}% (n={sel_n}, {sel_n / base_n * 100:.0f}% من الصفقات)")


# ══════════════════════════════════════════════════════════════════════════
# 🚀 أمثلة استخدام
# ══════════════════════════════════════════════════════════════════════════
#
# # 1) على نتائج باكتست (test_all_assets_v4 → build_flat_dataframe):
# flat_df = build_flat_dataframe(results_df, target_specs='close')
# selected = select_and_rank_trades(flat_df, min_edge_ratio=1.0, sort_by='both', top_n=20)
# print_trade_selection(selected, original=flat_df,
#                        title="\n🎯 أفضل 20 صفقة (تحرك متوقع > عدم يقين، مرتّبة بالثقة+الحجم):")
#
# # 2) بترتيب حسب الثقة فقط، أو حسب حجم التحرك فقط، أو حسب نقاء الإشارة (edge):
# by_conf = select_and_rank_trades(flat_df, sort_by='confidence')
# by_move = select_and_rank_trades(flat_df, sort_by='move')
# by_edge = select_and_rank_trades(flat_df, sort_by='edge')
#
# # 3) مباشرة على تقرير التداول الحي (نفس الدالة، بلا أي تغيير):
# live_table = predict_latest_all_assets(model, test_dict, ['1h', '4h', '1D'],
#                                        target_specs='close', n_display=10, verbose=False)
# live_selected = select_and_rank_trades(live_table, min_edge_ratio=1.2, sort_by='confidence')
# print_trade_selection(live_selected, title="\n🔴 صفقات لايف مؤهّلة (تحرك > 1.2x عدم اليقين):")
