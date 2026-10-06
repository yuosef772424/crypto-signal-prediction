"""
PURPOSE:  Validated input-output pattern discovery: raw X-window snapshot features vs prediction success with train/validation split, binomial and permutation tests.
TAGS:     extract_input_snapshot_features, analyze_input_output_validated_patterns, run_input_output_pattern_discovery, plot_pattern_validation, input patterns, permutation test, binomial test
PITFALLS: Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.

## 2️⃣0️⃣ 🧬 تحليل المدخلات ↔ المخرجات مع التحقق الإحصائي (Validated Input-Output Pattern Discovery)

يجيب على السؤال: **ما الأنماط في المدخلات الخام (نوافذ X) التي ترتبط فعلاً بنجاح
التوقع، وما الأنماط التي تبدو مقنعة لكنها عشوائية/حفظ زائد؟**

على عكس `detect_success_failure_patterns` (١٢) الذي يعمل على خصائص *مُشتَقّة من
المخرجات* (الثقة، عدم اليقين، حجم الحركة المتوقعة)، القسم هنا يعمل على
**المدخلات الخام** نفسها (نوافذ X التي تدخل النموذج) ويضيف خطوة **تحقق إحصائية
إلزامية** (تقسيم زمني train/validation + Binomial Test + Permutation Test) قبل
تصنيف أي نمط كـ"جيد" أو "سيء" — أي نمط لا يتكرر خارج بيانات اكتشافه يُصنَّف
صراحة "🎲 غير موثوق" ولا يظهر ضمن القوائم النهائية.

**الدوال:**
1. `extract_input_snapshot_features` — يحوّل نوافذ X الخام إلى جدول خصائص واحد (صف/عينة).
2. `analyze_input_output_validated_patterns` — القلب: شجرة قرار + Binomial Test لكل ورقة + Permutation Test لكل خاصية مفردة.
3. `run_input_output_pattern_discovery` — غلاف سريع يربط الخطوتين أعلاه مباشرة بأصل من `test_dict` (تنبؤ + تقييم + تحليل أنماط في استدعاء واحد).
"""
# ═══════════════════════════════════════════════════════════════════════════
# 🧬 بصمة المدخلات: تحويل نوافذ X الخام إلى جدول خصائص واحد (صف لكل عينة)
# ═══════════════════════════════════════════════════════════════════════════

def extract_input_snapshot_features(
    X_inputs: Tuple[np.ndarray, ...],
    timeframe_names: List[str],
    feature_names: Optional[List[Optional[List[str]]]] = None,
    agg: Tuple[str, ...] = ('last', 'mean', 'std'),
    max_features_per_tf: Optional[int] = None,
) -> pd.DataFrame:
    """
    يحوّل نوافذ مدخلات النموذج الخام (X_inputs — نفس ما يدخل predict_batch_v4)
    إلى جدول واحد "بصمة مدخلات" (Input Snapshot) بصف واحد لكل عينة، ليصلح
    كمدخل لأي تحليل إحصائي/تعلّم آلي لاحق (خاصة analyze_input_output_validated_patterns
    و run_input_output_pattern_discovery أدناه).

    لكل إطار زمني ولكل خاصية داخل النافذة تُحسب إحصاءات ملخِّصة عبر محور الزمن
    (آخر قيمة/متوسط/انحراف معياري...) بدل تمرير النافذة الكاملة — لأن الهدف هنا
    تفسير الأنماط، لا إعادة تدريب النموذج.

    Args:
        X_inputs: tuple بنفس طول/ترتيب timeframe_names؛ كل عنصر [N, window, n_features]
                  أو [N, n_features] (بلا نافذة زمنية).
        timeframe_names: أسماء الأطر الزمنية بنفس ترتيب X_inputs، مثل ['1h','4h','1D'].
        feature_names: أسماء خصائص كل إطار (قائمة قوائم بنفس طول timeframe_names)؛
                       إن غابت (أو غاب عنصر لإطار معيّن) تُولَّد أسماء عامة f0..fK.
        agg: الإحصاءات المُحسَبة عبر محور النافذة الزمنية — من
             {'last','mean','std','min','max'}.
        max_features_per_tf: تحديد عدد أول N خاصية من كل إطار (لتفادي انفجار الأبعاد
                              إن كانت الخصائص كثيرة جداً)؛ None = كل الخصائص.

    Returns:
        DataFrame بأسماء أعمدة مقروءة مثل '1D_rsi_last', '4h_close_mean', ...
        (N صف بنفس ترتيب X_inputs — أي بنفس ترتيب last_candles/base_params).
    """
    if len(X_inputs) != len(timeframe_names):
        raise ValueError("طول X_inputs يجب أن يطابق طول timeframe_names.")

    all_cols: Dict[str, np.ndarray] = {}

    for tf_idx, (tf_name, X_tf) in enumerate(zip(timeframe_names, X_inputs)):
        arr = np.asarray(X_tf)
        if arr.ndim == 2:          # [N, features] بلا نافذة زمنية → نافذة طولها 1
            arr = arr[:, None, :]
        elif arr.ndim != 3:
            raise ValueError(f"شكل غير مدعوم لـ X_inputs['{tf_name}']: {arr.shape}")

        n_feat = arr.shape[-1]
        if max_features_per_tf is not None:
            n_feat = min(n_feat, max_features_per_tf)
            arr = arr[..., :n_feat]

        names_tf = None
        if feature_names is not None and tf_idx < len(feature_names) and feature_names[tf_idx] is not None:
            names_tf = list(feature_names[tf_idx])[:n_feat]
        if not names_tf or len(names_tf) != n_feat:
            names_tf = [f"f{i}" for i in range(n_feat)]

        for fi, fname in enumerate(names_tf):
            series = arr[:, :, fi].astype(np.float64)  # [N, window]
            if 'last' in agg:
                all_cols[f"{tf_name}_{fname}_last"] = series[:, -1]
            if 'mean' in agg:
                all_cols[f"{tf_name}_{fname}_mean"] = np.nanmean(series, axis=1)
            if 'std' in agg:
                all_cols[f"{tf_name}_{fname}_std"] = np.nanstd(series, axis=1)
            if 'min' in agg:
                all_cols[f"{tf_name}_{fname}_min"] = np.nanmin(series, axis=1)
            if 'max' in agg:
                all_cols[f"{tf_name}_{fname}_max"] = np.nanmax(series, axis=1)

    return pd.DataFrame(all_cols)


# ═══════════════════════════════════════════════════════════════════════════
# 🔬 ربط المدخلات بالمخرجات + التحقق الإحصائي (Validated Input-Output Patterns)
# ═══════════════════════════════════════════════════════════════════════════
#
# الفكرة: أي شجرة قرار أو ترابط إحصائي سيجد "أنماطاً" حتى في بيانات عشوائية
# تماماً إن قِسناها على نفس البيانات التي اكتُشفت منها (overfitting). لذلك كل
# نمط هنا يُختبر على بيانات لم يرَها الاكتشاف (تقسيم زمني train/validation)
# ويُخضَع لاختبار إحصائي (Binomial Test / Permutation Test) قبل أن يُصنَّف
# "جيداً" أو "سيئاً" — وإلا يُصنَّف صراحة كنمط "غير موثوق/يشبه العشوائية".
# ═══════════════════════════════════════════════════════════════════════════

def analyze_input_output_validated_patterns(
    input_features: pd.DataFrame,
    outcome: Union[pd.Series, np.ndarray],
    timestamps: Optional[Union[pd.Series, np.ndarray]] = None,
    validation_frac: float = 0.3,
    max_tree_depth: int = 3,
    min_leaf_samples: int = 30,
    n_permutations: int = 300,
    alpha: float = 0.05,
    consistency_gap_max: float = 0.15,
    verbose: bool = True,
) -> Dict:
    """
    يربط "بصمة المدخلات" (input_features — جدول خصائص خام، صف لكل عينة، مثل
    ناتج extract_input_snapshot_features) بنتيجة التوقع (outcome — 0/1: هل كان
    صحيحاً؟) ثم يتحقق إحصائياً من كل نمط مُكتشَف قبل تصنيفه، لتفادي أنماط
    عشوائية تبدو مقنعة على بيانات التدريب لكنها لا تتكرر خارجها.

    المنهجية:
      1) تقسيم **زمني** (لا عشوائي) train/validation — يحاكي واقع التداول
         (اكتشاف على الماضي، تحقق على المستقبل الأقرب؛ يفترض أن صفوف
         input_features/outcome مرتّبة زمنياً أصلاً، أو تُرتَّب عبر timestamps
         إن مُرِّرت).
      2) شجرة قرار ضحلة على train فقط → قواعد "إن-فإن" (كل ورقة = نمط مرشّح).
      3) لكل قاعدة: نسبة النجاح على train **و** على validation منفصلتين.
      4) اختبار ذو حدين (Binomial Test) يقارن نجاح القاعدة في validation
         بمعدل النجاح الأساسي (baseline) في نفس الفترة → p-value.
      5) القاعدة تُصنَّف "✅ نمط موثوق جيد" فقط إن كانت دلالتها الإحصائية
         موجودة في validation (p<alpha) **و** فجوة train/validation صغيرة
         (< consistency_gap_max) — أي لم "تحفظ" الشجرة صدفة من التدريب.
         نفس المنطق بالعكس لـ"❌ نمط موثوق سيء". أي قاعدة تفشل أحد الشرطين
         تُصنَّف "🎲 غير موثوق / يشبه العشوائية" ولا يُعتمَد عليها.
      6) بالتوازي: لكل خاصية مفردة (عمود عددي) اختبار تبديل (Permutation Test)
         مستقل عن الشجرة — يخلط outcome عشوائياً n_permutations مرة ويقارن
         الارتباط الحقيقي بتوزيع الارتباطات العشوائية، ثم يتحقق أن الارتباط
         **يتكرر بنفس الاتجاه** على validation قبل اعتباره حقيقياً.

    Args:
        input_features: DataFrame خصائص خام (عمودي رقمية)، صف لكل عينة.
        outcome: مصفوفة/عمود 0/1 (أو bool) — هل كان التوقع صحيحاً لكل عينة؟
        timestamps: (اختياري) لترتيب العينات زمنياً قبل التقسيم؛ إن غابت
                    يُفترض أن الترتيب الحالي زمني أصلاً (كما تُبنى بيانات هذا
                    المشروع عبر last_candles/base_params).
        validation_frac: نسبة عينات التحقق (من نهاية السلسلة الزمنية).
        max_tree_depth / min_leaf_samples: ضبط شجرة القرار الاستكشافية.
        n_permutations: عدد تكرارات اختبار التبديل لكل خاصية.
        alpha: مستوى الدلالة الإحصائية المطلوب (افتراضياً 5%).
        consistency_gap_max: أقصى فرق مقبول بين نسبة نجاح train وvalidation
                              لاعتبار القاعدة مستقرة (وليست حفظاً زائداً).

    Returns:
        dict: {
            'rules_df': DataFrame,             # كل ورقة/قاعدة + إحصاءات + الحكم
            'feature_stats_df': DataFrame,      # كل خاصية + ارتباط + p-value + الحكم
            'tree_rules_text': str,
            'baseline_rate': float, 'n_train': int, 'n_val': int,
            'best_patterns': DataFrame,         # أنماط موثوقة جيدة فقط (مرتّبة)
            'worst_patterns': DataFrame,        # أنماط موثوقة سيئة فقط
            'unreliable_patterns': DataFrame,   # أنماط فشلت التحقق (تُستبعَد من القرار)
        }
    """
    from sklearn.tree import DecisionTreeClassifier, export_text
    from scipy.stats import binomtest, pointbiserialr

    X_all = input_features.reset_index(drop=True).copy()
    y_all = pd.Series(np.asarray(outcome, dtype=np.float64), name='outcome').reset_index(drop=True)

    mask = y_all.notna() & X_all.notna().all(axis=1)
    X_all, y_all = X_all[mask].reset_index(drop=True), y_all[mask].reset_index(drop=True)

    if timestamps is not None:
        ts = pd.Series(np.asarray(timestamps)[np.asarray(mask)]).reset_index(drop=True)
        order = np.argsort(ts.values, kind='stable')
        X_all, y_all = X_all.iloc[order].reset_index(drop=True), y_all.iloc[order].reset_index(drop=True)

    n = len(X_all)
    if n < 50:
        raise ValueError(f"عدد العينات الصالحة قليل جداً ({n} < 50) لتحقق إحصائي موثوق.")

    n_val = max(int(round(n * validation_frac)), min_leaf_samples * 2)
    n_train = n - n_val
    if n_train < min_leaf_samples * 4:
        raise ValueError("عدد عينات التدريب غير كافٍ بعد الفصل الزمني — قلّل validation_frac أو min_leaf_samples.")

    X_train, X_val = X_all.iloc[:n_train].reset_index(drop=True), X_all.iloc[n_train:].reset_index(drop=True)
    y_train, y_val = y_all.iloc[:n_train].reset_index(drop=True), y_all.iloc[n_train:].reset_index(drop=True)
    baseline_rate = float(y_val.mean())  # معدل النجاح الأساسي في فترة التحقق نفسها (مقارنة عادلة)

    # ── 1) شجرة قرار على train فقط ──────────────────────────────────────────
    tree = DecisionTreeClassifier(max_depth=max_tree_depth, min_samples_leaf=min_leaf_samples,
                                   class_weight='balanced', random_state=42)
    tree.fit(X_train, y_train)
    tree_rules_text = export_text(tree, feature_names=list(X_all.columns))

    leaf_train = tree.apply(X_train)
    leaf_val = tree.apply(X_val)

    rule_rows = []
    for leaf_id in np.unique(leaf_train):
        tr_mask = leaf_train == leaf_id
        n_tr = int(tr_mask.sum())
        if n_tr < min_leaf_samples:
            continue
        rate_tr = float(y_train[tr_mask].mean())

        va_mask = leaf_val == leaf_id
        n_va = int(va_mask.sum())
        if n_va == 0:
            rule_rows.append({
                'leaf_id': int(leaf_id), 'n_train': n_tr, 'نسبة_نجاح_train': round(rate_tr * 100, 2),
                'n_val': 0, 'نسبة_نجاح_val': float('nan'), 'فجوة_train_val': float('nan'),
                'p_value': float('nan'), 'الحكم': "🎲 غير موثوق (لا عينات مطابقة في validation)",
            })
            continue

        rate_va = float(y_val[va_mask].mean())
        k_success = int(y_val[va_mask].sum())
        alt = 'greater' if rate_tr >= baseline_rate else 'less'
        p_val = float(binomtest(k_success, n_va, baseline_rate, alternative=alt).pvalue)
        gap = abs(rate_tr - rate_va)

        if p_val < alpha and gap <= consistency_gap_max and rate_va > baseline_rate:
            verdict = "✅ نمط موثوق جيد"
        elif p_val < alpha and gap <= consistency_gap_max and rate_va < baseline_rate:
            verdict = "❌ نمط موثوق سيء"
        else:
            verdict = "🎲 غير موثوق / يشبه العشوائية"

        rule_rows.append({
            'leaf_id': int(leaf_id), 'n_train': n_tr, 'نسبة_نجاح_train': round(rate_tr * 100, 2),
            'n_val': n_va, 'نسبة_نجاح_val': round(rate_va * 100, 2),
            'فجوة_train_val': round(gap * 100, 2), 'p_value': round(p_val, 4), 'الحكم': verdict,
        })

    rules_df = pd.DataFrame(rule_rows).sort_values('نسبة_نجاح_val', ascending=False, na_position='last').reset_index(drop=True)

    # ── 2) اختبار تبديل لكل خاصية مفردة (مستقل عن الشجرة) ───────────────────
    numeric_cols = X_all.select_dtypes(include=[np.number]).columns
    rng = np.random.default_rng(42)
    feat_rows = []
    for col in numeric_cols:
        v_train, v_val = X_train[col].values, X_val[col].values
        if np.std(v_train) < 1e-12:
            continue
        r_train, _ = pointbiserialr(y_train.values, v_train)

        null_r = np.empty(n_permutations)
        y_shuf = y_train.values.copy()
        for i in range(n_permutations):
            rng.shuffle(y_shuf)
            null_r[i], _ = pointbiserialr(y_shuf, v_train)
        p_perm = float(np.mean(np.abs(null_r) >= abs(r_train)))

        r_val = float(pointbiserialr(y_val.values, v_val)[0]) if np.std(v_val) > 1e-12 else float('nan')
        replicates = (not np.isnan(r_val)) and (np.sign(r_val) == np.sign(r_train)) and (abs(r_val) > 0.05)

        if p_perm < alpha and replicates:
            verdict = "✅ خاصية مؤثرة وموثوقة"
        elif p_perm < alpha and not replicates:
            verdict = "🎲 دالة على train فقط (لا تتكرر) — غالباً صدفة/overfitting"
        else:
            verdict = "◻️ غير مؤثرة إحصائياً"

        feat_rows.append({
            'الخاصية': col, 'ارتباط_train': round(float(r_train), 4),
            'ارتباط_val': round(r_val, 4) if not np.isnan(r_val) else float('nan'),
            'p_value_permutation': round(p_perm, 4), 'الحكم': verdict,
        })

    feature_stats_df = pd.DataFrame(feat_rows).sort_values('p_value_permutation').reset_index(drop=True)

    best_patterns = rules_df[rules_df['الحكم'] == "✅ نمط موثوق جيد"].reset_index(drop=True)
    worst_patterns = rules_df[rules_df['الحكم'] == "❌ نمط موثوق سيء"].reset_index(drop=True)
    unreliable_patterns = rules_df[rules_df['الحكم'].str.startswith("🎲")].reset_index(drop=True)

    results = {
        'rules_df': rules_df, 'feature_stats_df': feature_stats_df, 'tree_rules_text': tree_rules_text,
        'baseline_rate': baseline_rate, 'n_train': n_train, 'n_val': n_val,
        'best_patterns': best_patterns, 'worst_patterns': worst_patterns, 'unreliable_patterns': unreliable_patterns,
    }

    if verbose:
        print("=" * 100)
        print("🔬 تحليل المدخلات ↔ المخرجات مع التحقق الإحصائي (Validated Pattern Discovery)")
        print("=" * 100)
        print(f"عينات: train={n_train} | validation={n_val} | معدل النجاح الأساسي (val) = {baseline_rate*100:.1f}%")

        print(f"\n🌳 قواعد شجرة القرار (على train فقط):")
        print("-" * 60)
        save_or_print(tree_rules_text, 'input_output_tree_rules', out_dir=DEFAULT_OUTPUT_DIR)

        print(f"\n📋 كل الأنماط (أوراق الشجرة) مع نتيجة التحقق على validation:")
        print("-" * 60)
        save_or_print(rules_df, 'input_output_rules_validated', out_dir=DEFAULT_OUTPUT_DIR)

        print(f"\n✅ أفضل الأنماط الموثوقة ({len(best_patterns)}):")
        for _, r in best_patterns.head(5).iterrows():
            print(f"   • ورقة #{r['leaf_id']}: نجاح_val={r['نسبة_نجاح_val']}% (train={r['نسبة_نجاح_train']}%) "
                  f"| n_val={r['n_val']} | p={r['p_value']}")
        if best_patterns.empty:
            print("   (لا يوجد نمط اجتاز التحقق الإحصائي كـ'جيد موثوق')")

        print(f"\n❌ أسوأ الأنماط الموثوقة ({len(worst_patterns)}):")
        for _, r in worst_patterns.head(5).iterrows():
            print(f"   • ورقة #{r['leaf_id']}: نجاح_val={r['نسبة_نجاح_val']}% (train={r['نسبة_نجاح_train']}%) "
                  f"| n_val={r['n_val']} | p={r['p_value']}")
        if worst_patterns.empty:
            print("   (لا يوجد نمط اجتاز التحقق الإحصائي كـ'سيء موثوق')")

        print(f"\n🎲 {len(unreliable_patterns)} نمط غير موثوق — لا يُعتمَد عليه (يشبه الصدفة/الحفظ الزائد)")

        print(f"\n🧪 إحصاءات الخصائص الفردية (اختبار تبديل، {n_permutations} تكرار):")
        print("-" * 60)
        save_or_print(feature_stats_df, 'input_feature_permutation_stats', out_dir=DEFAULT_OUTPUT_DIR)
        reliable_feats = feature_stats_df[feature_stats_df['الحكم'] == "✅ خاصية مؤثرة وموثوقة"]
        if len(reliable_feats):
            top = reliable_feats.iloc[0]
            print(f"   → أهم خاصية مفردة موثوقة: '{top['الخاصية']}' "
                  f"(ارتباط train={top['ارتباط_train']}, val={top['ارتباط_val']})")
        else:
            print("   → لا توجد خاصية مفردة ذات دلالة إحصائية تتكرر في train وvalidation معاً.")

        try:
            plot_pattern_validation(rules_df, save_dir=DEFAULT_OUTPUT_DIR)
        except Exception as e:
            print(f"   ⚠️ تعذر رسم مخطط الأنماط: {e}")

    return results


def plot_pattern_validation(rules_df: pd.DataFrame, save_dir: str = DEFAULT_OUTPUT_DIR) -> Optional[str]:
    """رسم شريطي: نسبة نجاح كل نمط (ورقة) على train مقابل validation، بلون يعكس الحكم النهائي."""
    df = rules_df.dropna(subset=['نسبة_نجاح_val']).sort_values('نسبة_نجاح_val', ascending=False)
    if df.empty:
        return None

    color_map = {"✅ نمط موثوق جيد": "#2ca02c", "❌ نمط موثوق سيء": "#d62728",
                 "🎲 غير موثوق / يشبه العشوائية": "#7f7f7f"}
    colors = df['الحكم'].map(color_map).fillna("#7f7f7f")

    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.6), 5))
    x = np.arange(len(df))
    ax.bar(x - 0.18, df['نسبة_نجاح_train'], width=0.36, color='#9ecae1', label='Train success %')
    ax.bar(x + 0.18, df['نسبة_نجاح_val'], width=0.36, color=colors, label='Validation success %')
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{int(l)}" for l in df['leaf_id']], rotation=0)
    ax.set_ylabel('Success rate (%)')
    ax.set_xlabel('Leaf / pattern id')
    ax.set_title('Input→Output Patterns: Train vs Validation success rate')
    ax.legend()
    fig.tight_layout()

    path = None
    if save_dir:
        fig_dir = os.path.join(save_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        path = os.path.join(fig_dir, 'pattern_validation.png')
        fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# 🚀 غلاف سريع: مدخلات أصل واحد → تقييم → تحليل أنماط مُتحقَّق منها
# ═══════════════════════════════════════════════════════════════════════════

def run_input_output_pattern_discovery(
    model,
    test_dict: Dict,
    asset: str,
    timeframes: List[str],
    target_specs,
    feature_names: Optional[List[Optional[List[str]]]] = None,
    agg: Tuple[str, ...] = ('last', 'mean', 'std'),
    validation_frac: float = 0.3,
    batch_size: int = 256,
    timestamp_key: Optional[str] = None,
    timestamp_col: Optional[int] = None,
    verbose: bool = True,
    **analyze_kwargs,
) -> Dict[str, Dict]:
    """
    غلاف سريع يربط مدخلات نموذج أصل واحد الخام (X_inputs) بنتائج تقييمه
    (عبر predict_with_evaluation_v4) ثم يمرّرها إلى
    analyze_input_output_validated_patterns لكل هدف على حدة — أي دالة واحدة
    تُجيب على: "ما الأنماط في المدخلات الخام التي تُميَّز فعلاً بين توقعات
    صحيحة وتوقعات خاطئة/عشوائية، بعد التحقق الإحصائي؟"

    ملاحظة: تعمل على أصل واحد لأن ترتيب X_inputs زمني ومستقل لكل أصل (لا يجوز
    خلط أصول مختلفة في نفس التقسيم الزمني train/validation). لتحليل عدة أصول
    استدعِ الدالة لكل أصل على حدة، ثم قارن جداول rules_df/feature_stats_df يدوياً
    (أو اجمعها إن أردتَ حكماً عاماً عبر الأصول).

    Args:
        test_dict / timeframes / target_specs: نفس اصطلاحات test_all_assets_v4.
        feature_names: أسماء خصائص كل إطار زمني (تُمرَّر كما هي إلى
                       extract_input_snapshot_features) — مهم لقراءة الأنماط
                       (مثل 'rsi', 'close', 'volume'...) بدل f0..fK.
        باقي المعاملات: انظر extract_input_snapshot_features و
                        analyze_input_output_validated_patterns.

    Returns:
        dict: {target_name: نتيجة analyze_input_output_validated_patterns} —
              هدف واحد فقط إن فشل التحليل عليه يُستبعَد مع طباعة تحذير.
    """
    if asset not in test_dict:
        raise KeyError(f"الأصل '{asset}' غير موجود في test_dict.")

    test_data = test_dict[asset]
    X_inputs = tuple(test_data[f'X_{tf}'] for tf in timeframes)
    base_params = test_data['base_params']
    last_candles = test_data.get('last_candles')

    specs = resolve_targets(target_specs)
    y_true = {f'y_{s.name}': test_data['y'][s.name] for s in specs if s.name in test_data.get('y', {})}

    ts = _align_timestamps(_find_timestamps(test_data, timestamp_key, timestamp_col),
                            len(base_params), len(base_params))

    evaluated = predict_with_evaluation_v4(
        model, X_inputs, specs, base_params=base_params, last_candles=last_candles,
        y_true=y_true, batch_size=batch_size, verbose=False, timestamps=ts, asset=asset,
    )

    input_features = extract_input_snapshot_features(X_inputs, timeframes, feature_names=feature_names, agg=agg)

    out: Dict[str, Dict] = {}
    for spec in specs:
        data = evaluated.get(spec.name)
        if not isinstance(data, dict):
            continue

        outcome_col = 'direction_correct' if 'direction_correct' in data else ('correct' if 'correct' in data else None)
        if outcome_col is None:
            if verbose:
                print(f"⚠️ لا يوجد عمود نتيجة (direction_correct/correct) للهدف '{spec.name}' — تخطّي (يحتاج y_true).")
            continue

        outcome = np.asarray(data[outcome_col], dtype=np.float64)
        n = min(len(input_features), len(outcome))
        has_truth = np.asarray(data.get('has_truth', np.ones(n, dtype=bool)))[:n]

        feats = input_features.iloc[:n][has_truth].reset_index(drop=True)
        oc = outcome[:n][has_truth]
        ts_valid = np.asarray(ts)[:n][has_truth] if ts is not None else None

        if verbose:
            print(f"\n{'#' * 100}\n🎯 الهدف: {spec.name}  |  الأصل: {asset}\n{'#' * 100}")
        try:
            out[spec.name] = analyze_input_output_validated_patterns(
                feats, oc, timestamps=ts_valid, validation_frac=validation_frac, verbose=verbose,
                **analyze_kwargs,
            )
        except ValueError as e:
            print(f"   ⚠️ تعذّر التحليل للهدف '{spec.name}': {e}")

    return out


# ══════════════════════════════════════════════════════════════════════════
# 🚀 أمثلة استخدام
# ══════════════════════════════════════════════════════════════════════════
#
# # 1) الطريقة السريعة: أصل واحد من test_dict مباشرة (تنبؤ + تقييم + تحليل أنماط):
# patterns = run_input_output_pattern_discovery(
#     model, test_dict, asset='ENSUSDT', timeframes=['1h', '4h', '1D'],
#     target_specs='close',
#     feature_names=[['rsi','close','volume','ema','atr', ...]] * 3,  # اختياري لكن مفيد جداً للقراءة
#     validation_frac=0.3,
# )
# best = patterns['close']['best_patterns']     # أفضل الأنماط الموثوقة
# worst = patterns['close']['worst_patterns']   # أسوأ الأنماط الموثوقة
# feats = patterns['close']['feature_stats_df'] # كل خاصية مفردة + دلالتها
#
# # 2) الطريقة اليدوية (تحكّم كامل بمصدر outcome/input_features):
# X_inputs = tuple(test_dict['ENSUSDT'][f'X_{tf}'] for tf in ['1h', '4h', '1D'])
# input_features = extract_input_snapshot_features(X_inputs, ['1h', '4h', '1D'])
# # outcome: أي عمود 0/1 من نتيجة evaluate_predictions_v4 (direction_correct / correct)
# result = analyze_input_output_validated_patterns(input_features, outcome, validation_frac=0.3)
