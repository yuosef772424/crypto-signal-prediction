"""
PURPOSE:  Automatic pattern discovery: decision tree, random-forest importances, K-Means clusters, isolation-forest anomalies per head.
TAGS:     detect_success_failure_patterns, decision tree, feature importance, kmeans, isolation forest, pattern discovery, per_head
PITFALLS: Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.

## 1️⃣2️⃣ 🔬 تقرير: اكتشاف الأنماط آلياً (Pattern Discovery)

شجرة قرار مفسِّرة، أهمية الخصائص (Random Forest) + رسم بياني، تجميع سلوكي (K-Means) + إسقاط PCA، واكتشاف شذوذ (Isolation Forest) — لفهم متى ولماذا تكون توقعات النموذج متميزة أو خاطئة.

يعمل لكل رأس (high/low/close) على حدة افتراضياً (`per_head=True`) ويطبع لكل رأس خط الأساس الساذج والدقة الموزونة؛ `per_head=False` يُبقي الجدول المجمَّع. `p_up` (احتمال صعود رأس التصنيف) خاصية مرشّحة حين يتوفّر.
"""
# ═══════════════════════════════════════════════════════════════════════════
# 🔬 اكتشاف الأنماط آلياً في مخرجات النموذج (Pattern Discovery)
# ═══════════════════════════════════════════════════════════════════════════
#
# التحليل يجري لكل رأس (high/low/close) على حدة افتراضياً (per_head=True): الرؤوس لها علامة تغيّر متوقَّع
# مختلفة بنيوياً في entry_range (high موجب دائماً، low سالب دائماً)، فجدولها المجمَّع يجعل الشجرة تنقسم
# على «أي رأس» لا على نجاح/فشل. per_head=False يُبقي الجدول المجمَّع القديم.
#
# يستخدم خوارزميات تعلّم آلي بسيطة وقابلة للتفسير لاكتشاف الأنماط التي
# تجعل توقعات النموذج "متميزة" (صحيحة بثقة) أو "خاطئة" (نمط فشل متكرر):
#
#   1) شجرة قرار ضحلة  → قواعد "إن-فإن" مقروءة بشرياً تفسّر متى ينجح/يفشل
#   2) Random Forest    → أهمية كل خاصية (Feature Importance) في التمييز
#   3) K-Means Clustering → تجميع التوقعات في "أنماط سلوك" ووصف كل نمط
#   4) Isolation Forest → اكتشاف التوقعات الشاذة (Outliers) ومقارنة أدائها
# ═══════════════════════════════════════════════════════════════════════════

def _detect_patterns_single(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    max_tree_depth: int = 4,
    n_clusters: int = 4,
    top_n_assets_onehot: int = 8,
    verbose: bool = True,
    head: Optional[str] = None,
) -> Dict:
    """
    نواة detect_success_failure_patterns: جدول واحد (رأس واحد أو مجمَّع). `head` يسمّي الرأس في العناوين
    وأسماء الملفات المحفوظة (بلاه: الأسماء القديمة نفسها) كي لا يكتب رأسٌ فوق ملفات آخر.

    Returns:
        dict: {
            'tree_rules': str,                  # قواعد شجرة القرار كنص مقروء
            'feature_importance': DataFrame,     # ترتيب الخصائص حسب الأهمية
            'clusters': DataFrame,               # وصف كل عنقود سلوكي
            'clustered_df': DataFrame,           # البيانات الأصلية + عمود cluster
            'anomalies': DataFrame,              # التوقعات الشاذة وأداؤها
        }
    """
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    work = df.copy()
    if 'predicted_change_pct' not in work.columns and 'pct_error' in work.columns:
        work['predicted_change_pct'] = 0.0  # عمود بديل إن غاب (بيانات فئوية بحتة)

    auto_features = feature_cols is None
    if auto_features:
        candidate_cols = ['confidence', 'uncertainty', 'aleatoric', 'epistemic', 'predicted_change_pct']
        feature_cols = [c for c in candidate_cols if c in work.columns]
        # عدم اليقين بوحدة السعر غير قابل للمقارنة بين العملات (عنقود = عملة واحدة غالية): النسخة النسبية لسعر الدخول إن وُجدت
        feature_cols = [f'{c}_ret' if c in ('uncertainty', 'aleatoric', 'epistemic') and f'{c}_ret' in work.columns
                        and work[f'{c}_ret'].notna().all() else c for c in feature_cols]

    work = work.dropna(subset=feature_cols + ['correct'])
    if len(work) < 20:
        raise ValueError("عدد العينات الصالحة قليل جداً (<20) لاكتشاف أنماط موثوقة.")

    # p_up (احتمال صعود رأس التصنيف): خاصية مرشّحة حين تتوفّر لكل العيّنات — بلا NaN يُسقط صفوفاً. في الأوضاع التي
    # لا تقرأ chicks فيها رأس التصنيف يكون العمود كله NaN فلا يدخل، وتبقى الخصائص كما كانت.
    if auto_features and 'p_up' in work.columns and work['p_up'].notna().all():
        feature_cols = feature_cols + ['p_up']
    tag = f"_{head}" if head else ""
    head_txt = f" — الرأس: {head}" if head else ""

    # ترميز أهم N أصول كـ one-hot (يبقي الباقي كفئة 'أخرى' لتفادي انفجار الأبعاد)
    if 'asset' in work.columns:
        top_assets = work['asset'].value_counts().nlargest(top_n_assets_onehot).index
        work['asset_grouped'] = np.where(work['asset'].isin(top_assets), work['asset'], 'أخرى')
        asset_dummies = pd.get_dummies(work['asset_grouped'], prefix='asset')
        X = pd.concat([work[feature_cols].reset_index(drop=True), asset_dummies.reset_index(drop=True)], axis=1)
    else:
        X = work[feature_cols].reset_index(drop=True)

    y = work['correct'].astype(int).reset_index(drop=True)

    results = {'head': head, 'n': int(len(work)), 'features': list(feature_cols)}

    # ── 1) شجرة قرار مفسِّرة ────────────────────────────────────────────────
    tree = DecisionTreeClassifier(max_depth=max_tree_depth, min_samples_leaf=max(10, len(X) // 50),
                                   class_weight='balanced', random_state=42)
    tree.fit(X, y)
    tree_rules = export_text(tree, feature_names=list(X.columns))
    results['tree_rules'] = tree_rules
    results['tree_accuracy'] = float(tree.score(X, y))
    # ✅ class_weight='balanced' يُحسّن عمد ترجيح الفئة الأقلّ على حساب دقة خامة
    # ممكنة على الفئة الأكثر — فقد تقلّ tree_accuracy (دقة خام غير موزونة)
    # عن خط أساس "توقّع الفئة الأكثر دائماً" بلا أن يعني ذلك فشل الشجرة —
    # الدقة الموزونة (balanced_accuracy_score) هي المقياس المطابق فعلياً لما دُرّبت
    # عليه الشجرة.
    results['tree_naive_baseline_accuracy'] = float(max(y.mean(), 1 - y.mean()))
    results['tree_balanced_accuracy'] = float(balanced_accuracy_score(y, tree.predict(X)))

    # ── 2) أهمية الخصائص عبر Random Forest ─────────────────────────────────
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced',
                                 random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importance_df = pd.DataFrame({
        'الخاصية': X.columns,
        'الأهمية': rf.feature_importances_,
    }).sort_values('الأهمية', ascending=False).reset_index(drop=True)
    results['feature_importance'] = importance_df

    # ── 3) تجميع سلوكي (Clustering) ─────────────────────────────────────────
    cluster_features = [c for c in feature_cols if c in work.columns]
    scaler = StandardScaler()
    Xc = scaler.fit_transform(work[cluster_features])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = km.fit_predict(Xc)

    clustered = work.copy()
    clustered['cluster'] = cluster_labels

    cluster_summary = clustered.groupby('cluster').agg(
        عدد=('correct', 'size'),
        نسبة_نجاح=('correct', 'mean'),
        **{f'متوسط_{c}': (c, 'mean') for c in cluster_features},
    ).reset_index()
    cluster_summary['نسبة_نجاح'] = (cluster_summary['نسبة_نجاح'] * 100).round(2)
    cluster_summary = cluster_summary.sort_values('نسبة_نجاح', ascending=False)
    results['clusters'] = cluster_summary
    results['clustered_df'] = clustered

    # ── 4) اكتشاف الشذوذ (نتائج "متميزة" بمعنى غير اعتيادية) ────────────────
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    anomaly_flag = iso.fit_predict(work[cluster_features])  # -1 = شاذ
    work_anom = work.copy()
    work_anom['is_anomaly'] = anomaly_flag == -1
    anomalies = work_anom[work_anom['is_anomaly']]
    results['anomalies'] = anomalies

    if verbose:
        print("=" * 100)
        print(f"🔬 اكتشاف الأنماط آلياً في مخرجات النموذج{head_txt}")
        print("=" * 100)

        print(f"\n🌳 1) قواعد شجرة القرار (دقة خام: {results['tree_accuracy']*100:.1f}%، خط أساس ساذج: {results['tree_naive_baseline_accuracy']*100:.1f}%، دقة موزونة: {results['tree_balanced_accuracy']*100:.1f}%):")
        print("-" * 60)
        save_or_print(tree_rules, f'decision_tree_rules{tag}', out_dir=DEFAULT_OUTPUT_DIR)

        print(f"\n📊 2) أهمية الخصائص في التمييز بين النجاح والفشل:")
        print("-" * 60)
        save_or_print(importance_df, f'feature_importance{tag}', out_dir=DEFAULT_OUTPUT_DIR)
        print(f"   → أهم خاصية تفسّر نجاح/فشل التوقع: '{importance_df.iloc[0]['الخاصية']}'")
        plot_feature_importance(importance_df, save_dir=DEFAULT_OUTPUT_DIR, filename=f'feature_importance{tag}')

        print(f"\n🧩 3) الأنماط السلوكية المكتشفة (K-Means, k={n_clusters}):")
        print("-" * 60)
        save_or_print(cluster_summary.round(4), f'cluster_summary{tag}', out_dir=DEFAULT_OUTPUT_DIR)
        best_c = cluster_summary.iloc[0]
        worst_c = cluster_summary.iloc[-1]
        print(f"   ✅ أفضل نمط: العنقود #{int(best_c['cluster'])} بنجاح {best_c['نسبة_نجاح']}%")
        print(f"   ❌ أسوأ نمط: العنقود #{int(worst_c['cluster'])} بنجاح {worst_c['نسبة_نجاح']}%")
        try:
            plot_clusters_2d(clustered, cluster_features, save_dir=DEFAULT_OUTPUT_DIR, filename=f'clusters_pca{tag}')
        except Exception as e:
            print(f"   ⚠️ تعذر رسم العناقيد: {e}")

        print(f"\n🚨 4) التوقعات الشاذة (Isolation Forest): {len(anomalies)} من {len(work)} "
              f"({len(anomalies)/len(work)*100:.1f}%)")
        print("-" * 60)
        if len(anomalies) > 0:
            normal_acc = work_anom[~work_anom['is_anomaly']]['correct'].mean() * 100
            anom_acc = anomalies['correct'].mean() * 100
            print(f"   • دقة التوقعات العادية: {normal_acc:.1f}%")
            print(f"   • دقة التوقعات الشاذة:  {anom_acc:.1f}%")
            save_or_print(anomalies, f'anomalous_predictions{tag}', out_dir=DEFAULT_OUTPUT_DIR)
            if anom_acc < normal_acc - 10:
                print(f"   ⚠️  التوقعات الشاذة أقل دقة بوضوح — قد تحتاج فلترة أو مراجعة يدوية")
            elif anom_acc > normal_acc + 10:
                print(f"   💡 التوقعات الشاذة أعلى دقة رغم ندرتها — تستحق دراسة (فرصة محتملة)")
            else:
                print(f"   ℹ️  لا فرق واضح في الدقة بين الشاذ والعادي")

    return results


def _pattern_summary(per_head: Dict[str, Dict]) -> pd.DataFrame:
    """صف لكل رأس: خط الأساس الساذج (حصة الفئة الأكثر) مقابل الدقة الموزونة للشجرة (خط أساسها الساذج 50% بالتعريف)."""
    rows = []
    for h, r in per_head.items():
        imp = r['feature_importance']
        rows.append({
            'head': h, 'n': r['n'],
            'naive_baseline_accuracy': r['tree_naive_baseline_accuracy'],
            'tree_accuracy': r['tree_accuracy'],
            'tree_balanced_accuracy': r['tree_balanced_accuracy'],
            'top_feature': imp.iloc[0]['الخاصية'] if len(imp) else None,
            'features': ','.join(r['features']),
        })
    return pd.DataFrame(rows)


def detect_success_failure_patterns(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    max_tree_depth: int = 4,
    n_clusters: int = 4,
    top_n_assets_onehot: int = 8,
    verbose: bool = True,
    per_head: bool = True,
    head_col: str = 'target',
) -> Dict:
    """
    اكتشاف أنماط النجاح/الفشل (شجرة قرار + أهمية الخصائص + عناقيد + شذوذ) — لكل رأس (`head_col`، أي high/low/close)
    على حدة افتراضياً، أو على الجدول المجمَّع كله بـ `per_head=False` (السلوك القديم).

    لماذا لكل رأس: في entry_range يكون predicted_change_pct موجباً دائماً لـ high وسالباً دائماً لـ low، فالشجرة على
    الجدول المجمَّع تنقسم على «أي رأس» لا على النجاح. ولكل رأس نسبة نجاح أساسية مختلفة جداً (high قرب 100%)، لذا
    يُبلَّغ لكل رأس خط الأساس الساذج (حصة الفئة الأكثر) والدقة الموزونة (داخل العيّنة، خط أساسها 50%).

    الخصائص الافتراضية: confidence/uncertainty/aleatoric/epistemic/predicted_change_pct، و`p_up` (احتمال صعود رأس
    التصنيف لهذا الرأس) حين يتوفّر لكل عيّناته. رأس بلا عيّنات صالحة كافية (<20) يُسجَّل في 'skipped' بدل إفشال الباقي.

    Returns:
        per_head=True:
            'mode': 'per_head'، 'per_head': {رأس: نتيجة النواة}، 'heads': [...]، 'skipped': {رأس: سبب}،
            'summary': DataFrame (لكل رأس: n، خط الأساس الساذج، الدقة الخام والموزونة، أهم خاصية).
            وإن كان الجدول برأس واحد فقط أو بلا عمود `head_col` فحقول النواة نفسها (tree_rules/feature_importance/
            clusters/clustered_df/anomalies/tree_*) على المستوى الأعلى أيضاً — فيبقى التوافق مع من يقرأها مباشرة.
        per_head=False: حقول النواة على المستوى الأعلى، و'mode': 'pooled'، و'summary' بصف واحد 'all'.
        حقول النواة: tree_rules, tree_accuracy, tree_naive_baseline_accuracy, tree_balanced_accuracy,
        feature_importance, clusters, clustered_df, anomalies (+ head, n, features).
    """
    kw = dict(feature_cols=feature_cols, max_tree_depth=max_tree_depth, n_clusters=n_clusters,
              top_n_assets_onehot=top_n_assets_onehot, verbose=verbose)
    heads = list(pd.unique(df[head_col].dropna())) if head_col in df.columns else []

    if not per_head or len(heads) <= 1:
        sub = df if not heads or not per_head else df[df[head_col] == heads[0]]
        res = _detect_patterns_single(sub, **kw)
        name = str(heads[0]) if (per_head and heads) else 'all'
        res['head'] = name
        out = {**res, 'mode': 'per_head' if per_head else 'pooled', 'summary': _pattern_summary({name: res}),
               'skipped': {}}
        if per_head:
            out.update(per_head={name: res}, heads=[name])
        return out

    per, skipped = {}, {}
    for h in heads:
        try:
            per[str(h)] = _detect_patterns_single(df[df[head_col] == h], head=str(h), **kw)
        except ValueError as e:
            skipped[str(h)] = str(e)
            if verbose:
                print(f"\n⚠️ الرأس {h}: تخطي اكتشاف الأنماط — {e}")
    if not per:
        raise ValueError(f"لا رأس فيه عيّنات صالحة كافية لاكتشاف الأنماط: {skipped}")
    summary = _pattern_summary(per)
    if verbose:
        print("=" * 100)
        print("🔬 ملخص اكتشاف الأنماط لكل رأس — خط الأساس الساذج (الفئة الأكثر) مقابل الدقة الموزونة")
        print("=" * 100)
        print(summary.drop(columns=['features']).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("   الدقة الموزونة تُقاس داخل العيّنة (الشجرة تُدرَّب وتُقاس على البيانات نفسها) وخط أساسها الساذج 0.5؛ "
              "رأس دقته الخام بمستوى خط الأساس ودقته الموزونة قرب 0.5 لا نمط فيه.")
    return {'mode': 'per_head', 'per_head': per, 'heads': list(per), 'skipped': skipped, 'summary': summary}
