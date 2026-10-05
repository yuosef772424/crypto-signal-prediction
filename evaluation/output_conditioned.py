"""
PURPOSE:  Output-conditioned analysis: success rate by predicted-move type, confidence bucket and consecutive batch.
TAGS:     analyze_by_predicted_movement, analyze_by_confidence_bucket, analyze_by_batch, buckets
PITFALLS: Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.

## 1️⃣1️⃣ 🎯 تقرير: تحليل موجَّه للمخرجات (Output-Conditioned Analysis)

نسبة النجاح حسب نوع/حجم الحركة المتوقعة، حسب فئة الثقة، وحسب **دفعات متتالية** من العينات (لاكتشاف انحدار الأداء في دفعة معيّنة) — كل جدول كبير يُحفظ تلقائياً في ملف.
"""
# ═══════════════════════════════════════════════════════════════════════════
# 🎯 تحليل موجَّه للمخرجات (Output-Conditioned Analysis)
# ═══════════════════════════════════════════════════════════════════════════
#
# يجيب على أسئلة من نوع:
#   "ما نسبة نجاح التوقع عندما يتوقع النموذج حركة صعود بأكثر من 2%؟"
#   "هل هناك دفعة (batch) معيّنة من العينات حدث فيها انحدار واضح في الأداء
#    مقارنة بغيرها؟"
# ═══════════════════════════════════════════════════════════════════════════

def analyze_by_predicted_movement(
    df: pd.DataFrame,
    magnitude_col: str = 'predicted_change_pct',
    bins: Optional[List[float]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    نسبة نجاح التوقع مقسّمة حسب "نوع الحركة المتوقعة" (اتجاه × حجم الحركة).

    مثال على القراءة: "عندما توقع النموذج صعوداً بين 0.5% و 2%، نجح في
    الاتجاه 61% من المرات (على 340 عينة)".
    """
    if magnitude_col not in df.columns:
        raise ValueError(f"العمود '{magnitude_col}' غير موجود — هذا التحليل يتطلب أهدافاً مستمرة (أسعار).")

    if bins is None:
        bins = [-np.inf, -2, -0.5, 0, 0.5, 2, np.inf]
    labels = [f"({bins[i]:.1f}%, {bins[i+1]:.1f}%]" for i in range(len(bins) - 1)]

    work = df.copy()
    work['direction'] = np.where(work[magnitude_col] > 0, 'صعود', np.where(work[magnitude_col] < 0, 'هبوط', 'ثابت'))
    work['movement_bucket'] = pd.cut(work[magnitude_col], bins=bins, labels=labels)

    grouped = work.groupby(['target', 'direction', 'movement_bucket'], observed=True).agg(
        عدد=('correct', 'size'),
        نسبة_نجاح=('correct', 'mean'),
        متوسط_الثقة=('confidence', 'mean'),
    ).reset_index()
    grouped['نسبة_نجاح'] = (grouped['نسبة_نجاح'] * 100).round(2)
    grouped['متوسط_الثقة'] = grouped['متوسط_الثقة'].round(3)
    grouped = grouped[grouped['عدد'] > 0].sort_values(['target', 'direction'])

    if verbose:
        print("=" * 100)
        print("🎯 نسبة النجاح حسب نوع/حجم الحركة المتوقعة")
        print("=" * 100)
        save_or_print(grouped, 'movement_analysis', out_dir=DEFAULT_OUTPUT_DIR)

        best = grouped.loc[grouped['نسبة_نجاح'].idxmax()]
        worst = grouped.loc[grouped['نسبة_نجاح'].idxmin()]
        print(f"\n✅ أفضل نطاق حركة: {best['target']} / {best['direction']} {best['movement_bucket']} "
              f"→ نجاح {best['نسبة_نجاح']}% ({int(best['عدد'])} عينة)")
        print(f"❌ أسوأ نطاق حركة: {worst['target']} / {worst['direction']} {worst['movement_bucket']} "
              f"→ نجاح {worst['نسبة_نجاح']}% ({int(worst['عدد'])} عينة)")

    return grouped


def analyze_by_confidence_bucket(df: pd.DataFrame, n_buckets: int = 5, verbose: bool = True) -> pd.DataFrame:
    """نسبة النجاح حسب فئة الثقة (buckets متساوية العدد) — لكل target."""
    work = df.dropna(subset=['confidence']).copy()
    work['conf_bucket'] = pd.qcut(work['confidence'], q=n_buckets, duplicates='drop')

    grouped = work.groupby(['target', 'conf_bucket'], observed=True).agg(
        عدد=('correct', 'size'),
        نسبة_نجاح=('correct', 'mean'),
        متوسط_الثقة=('confidence', 'mean'),
    ).reset_index()
    grouped['نسبة_نجاح'] = (grouped['نسبة_نجاح'] * 100).round(2)

    if verbose:
        print("\n" + "=" * 100)
        print("📊 نسبة النجاح حسب فئة الثقة")
        print("=" * 100)
        save_or_print(grouped, 'confidence_bucket_analysis', out_dir=DEFAULT_OUTPUT_DIR)

    return grouped


def analyze_by_batch(
    df: pd.DataFrame,
    batch_size: int = 100,
    target: Optional[str] = None,
    z_threshold: float = 1.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    تحليل الأداء عبر دفعات متتالية من العينات (بترتيبها كما وردت) لاكتشاف
    "انحدار" (regression) في دفعة معيّنة مقارنة ببقية الدفعات.

    الطريقة: تقسيم العينات إلى دفعات بحجم batch_size، حساب نسبة النجاح
    ومتوسط الخطأ لكل دفعة، ثم حساب Z-score لكل دفعة بالنسبة لمتوسط/انحراف
    كل الدفعات — أي دفعة بـ |Z| >= z_threshold تُعتبر "شاذة" (انحدار أو تحسّن
    غير معتاد) وتتم المقارنة صراحة بينها وبين أفضل دفعة.
    """
    work = df if target is None else df[df['target'] == target]
    work = work.reset_index(drop=True)
    work['batch_id'] = work.index // batch_size

    agg_dict = {'correct': 'mean', 'confidence': 'mean'}
    if 'abs_error' in work.columns:
        agg_dict['abs_error'] = 'mean'

    batch_stats = work.groupby('batch_id').agg(**{
        'عدد': ('correct', 'size'),
        'نسبة_نجاح': ('correct', 'mean'),
        'متوسط_الثقة': ('confidence', 'mean'),
        **({'متوسط_الخطأ': ('abs_error', 'mean')} if 'abs_error' in work.columns else {}),
    }).reset_index()

    mean_wr, std_wr = batch_stats['نسبة_نجاح'].mean(), batch_stats['نسبة_نجاح'].std()
    batch_stats['z_score'] = (batch_stats['نسبة_نجاح'] - mean_wr) / (std_wr + 1e-9)
    batch_stats['حالة'] = np.where(
        batch_stats['z_score'] <= -z_threshold, '⚠️ انحدار',
        np.where(batch_stats['z_score'] >= z_threshold, '✅ تحسّن', 'طبيعي')
    )
    batch_stats['نسبة_نجاح'] = (batch_stats['نسبة_نجاح'] * 100).round(2)

    if verbose:
        print("\n" + "=" * 100)
        print(f"📦 تحليل الأداء عبر الدفعات (batch_size={batch_size}"
              + (f", target={target}" if target else "") + ")")
        print("=" * 100)
        save_or_print(batch_stats.round(4), f"batch_performance{'_' + target if target else ''}", out_dir=DEFAULT_OUTPUT_DIR)

        anomalies = batch_stats[batch_stats['حالة'] != 'طبيعي']
        best_batch = batch_stats.loc[batch_stats['نسبة_نجاح'].idxmax()]
        if not anomalies.empty:
            print(f"\n🔎 دفعات شاذة مقارنة بأفضل دفعة (#{int(best_batch['batch_id'])}, "
                  f"{best_batch['نسبة_نجاح']}% نجاح):")
            for _, row in anomalies.iterrows():
                diff = best_batch['نسبة_نجاح'] - row['نسبة_نجاح']
                print(f"   • الدفعة #{int(row['batch_id'])}: {row['نسبة_نجاح']}% نجاح "
                      f"({row['حالة']}) — أقل من الأفضل بـ {diff:.2f} نقطة مئوية")
        else:
            print("\n✅ لا توجد دفعات شاذة — الأداء مستقر عبر كل الدفعات")

    return batch_stats
