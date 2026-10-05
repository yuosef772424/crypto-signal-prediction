"""
PURPOSE:  run_full_analysis: the whole pipeline plus every report and plot in one call, big outputs saved to analysis_outputs/.
TAGS:     run_full_analysis, full analysis, all reports
PITFALLS: Tests exclude this module on the general load and exec it explicitly (load_into(ns, exclude=...)). Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.

## 1️⃣6️⃣ 🚀 الدالة الرئيسية الموحّدة v2: كل التحليلات + الرسوم + إدارة المخرجات

`run_full_analysis` تُشغّل خط الأنابيب كاملاً ثم كل التقارير والرسوم أعلاه في استدعاء واحد، وتحفظ كل مخرج كبير في `analysis_outputs/`.
"""
# ═══════════════════════════════════════════════════════════════════════════
# 🚀 الدالة الرئيسية الموحّدة v2: كل التحليلات + الرسوم + إدارة المخرجات
# ═══════════════════════════════════════════════════════════════════════════

def run_full_analysis(
    model,
    test_dict: Dict,
    timeframes: List[str],
    target_specs: List['TargetSpec'],
    batch_size: int = 256,
    range_frac: float = 0.2,
    batch_analysis_size: int = 100,
    out_dir: str = DEFAULT_OUTPUT_DIR,
    make_plots: bool = True,
    run_integrity_check: bool = True,
    verbose: bool = True,
    n_display: int = 1,
    timestamp_col: Optional[int] = None,
    timestamp_key: Optional[str] = None,
    run_trade_selection: bool = True,
    trade_min_edge_ratio: float = 1.0,
    trade_sort_by: str = 'both',
    trade_top_n: Optional[int] = 20,
    trade_cost_pct: float = 0.08,
    pnl_market_neutral: bool = False,
    eval_all_rows: bool = True,
    val_dict: Optional[Dict] = None,
    calibrators: Optional[Dict] = None,
    calibration_method: str = 'isotonic',
) -> Dict:
    """
    خط الأنابيب الكامل: تنبؤ + فك تشفير مرن + تحقق + تقييم، ثم كل التقارير
    (ثقة/فهم، مخرجات موجّهة، اكتشاف أنماط، تشخيص نزاهة) + الرسوم البيانية،
    مع حفظ أي مخرج كبير في `out_dir` بدل إغراق الـ notebook.

    استخدام سريع:
        full = run_full_analysis(model, test_dict, ['1h','4h','1D'], 'close')   # أو ['high','low'] أو specs

    pnl_market_neutral: عائد كل صفقة في Tearsheet ناقص متوسط كل العملات في نفس الطابع الزمني (محفظة محايدة
        للسوق). لازم مع أهداف relative: متوسط العائد النسبي (ناقص **الوسيط**) موجب لأن العوائد منحرفة يميناً،
        فـ«شراء كل شيء» يبدو رابحاً بلا أي مهارة. بعد طرح المتوسط يربح «شراء كل شيء» صفراً قبل التكلفة.
    eval_all_rows: True (الافتراضي) = كل عملات وصفوف الاختبار تدخل المقاييس المجمّعة (لا تُتخطّى عملة قصيرة)؛ n_display يبقى عرضاً فقط.
    val_dict: بنية test_dict نفسها لقسم **val** (build_chicks_test_dict({'VAL': val}, ...)). إن مُرِّر تُلائَم معايرة الثقة post-hoc
        عليه وحده (method = calibration_method: 'isotonic' | 'platt') وتُطبَّق على test في تقرير الثقة (ECE قبل/بعد) — بلا إعادة تدريب.
    calibrators: معايِرات جاهزة (fit_confidence_calibrators) بدل val_dict.
    """
    ensure_output_dir(out_dir)

    test_results = test_all_assets_v4(
        model, test_dict, timeframes, target_specs,
        verbose=verbose, batch_size=batch_size, range_frac=range_frac,
        n_display=n_display, timestamp_col=timestamp_col, timestamp_key=timestamp_key,
        eval_all_rows=eval_all_rows,
    )
    target_specs = test_results.get('target_specs', target_specs)   # الأهداف بعد الحسم (نصوص → TargetSpec)
    per_asset = test_results['per_asset_results']
    if per_asset.empty:
        print("⚠️ لا توجد نتائج — تحقق من test_dict")
        return test_results

    flat_df = build_flat_dataframe(per_asset, target_specs)
    save_or_print(flat_df, 'flat_predictions_all', out_dir=out_dir, verbose=False)  # يُحفظ دوماً كأرشيف كامل

    # ── تقرير الثقة والفهم + مخطط المعايرة ──────────────────────────────────
    val_flat = None
    if calibrators is None and val_dict is not None:
        val_res = test_all_assets_v4(model, val_dict, timeframes, target_specs, verbose=False, batch_size=batch_size,
                                     range_frac=range_frac, run_verification=False, eval_all_rows=True)
        if not val_res['per_asset_results'].empty:
            val_flat = build_flat_dataframe(val_res['per_asset_results'], val_res.get('target_specs', target_specs))
            calibrators = fit_confidence_calibrators(val_flat, method=calibration_method, verbose=verbose)
        elif verbose:
            print("⚠️ val_dict بلا نتائج — تُتخطّى معايرة الثقة")
    trust = generate_trust_report(flat_df, group_by='target', verbose=verbose, calibrators=calibrators)
    if make_plots and not trust['reliability_curve'].empty:
        for tgt in flat_df['target'].unique():
            plot_calibration_curve(trust['reliability_curve'], group_val=tgt, save_dir=out_dir)
        plot_confidence_vs_error(flat_df, save_dir=out_dir)

    # ── تحليل موجَّه للمخرجات ────────────────────────────────────────────────
    continuous_df = flat_df[flat_df['kind'] == 'continuous']
    movement_analysis = batch_analysis = None
    if not continuous_df.empty:
        movement_analysis = analyze_by_predicted_movement(continuous_df, verbose=verbose)
        batch_analysis = analyze_by_batch(continuous_df, batch_size=batch_analysis_size, verbose=verbose)
        if make_plots:
            plot_error_distribution(continuous_df, save_dir=out_dir)
            if batch_analysis is not None and not batch_analysis.empty:
                plot_batch_performance(batch_analysis, save_dir=out_dir)

    confidence_analysis = analyze_by_confidence_bucket(flat_df, verbose=verbose)

    # ── اختيار الصفقات: تحرك متوقع > عدم يقين + مقارنة معدل النجاح ────────────
    trade_candidates = None   # جميع الصفقات التي اجتازت Edge Filter قبل Top-N
    trade_selection = None    # الصفوف المعروضة/المحفوظة بعد Top-N والترتيب
    if run_trade_selection and not continuous_df.empty:
        trade_candidates = select_and_rank_trades(
            continuous_df,
            min_edge_ratio=trade_min_edge_ratio,
            sort_by=trade_sort_by,
            top_n=None,
        )
        trade_selection = (
            trade_candidates.head(int(trade_top_n)).copy()
            if trade_top_n is not None else trade_candidates.copy()
        )
        if trade_selection is not None and not trade_selection.empty:
            save_or_print(trade_selection, 'trade_selection', out_dir=out_dir, verbose=False)
        if verbose:
            print_trade_selection(
                trade_selection,
                original=continuous_df,
                title=(
                    f"\n🎯 أفضل الصفقات — |التحرك المتوقع| > "
                    f"{float(trade_min_edge_ratio):.2f}× عدم اليقين، "
                    f"ترتيب: {trade_sort_by}"
                ),
                max_rows=trade_top_n,
            )

            ok_col = next((c for c in ('direction_correct', 'correct', 'ok')
                           if c in continuous_df.columns and c in trade_candidates.columns), None)
            if ok_col and not trade_candidates.empty:
                filtered_acc, filtered_n = _trade_accuracy_pct(trade_candidates[ok_col])
                base_acc, base_n = _trade_accuracy_pct(continuous_df[ok_col])
                if base_n and filtered_n:
                    print(
                        f"  📊 Win Rate بعد Edge Filter (كل المؤهلين): {filtered_acc:.1f}% "
                        f"(n={filtered_n}) | قبل الفلترة: {base_acc:.1f}% (n={base_n})"
                    )

    # ── اكتشاف الأنماط آلياً ─────────────────────────────────────────────────
    patterns = None
    try:
        patterns = detect_success_failure_patterns(flat_df, verbose=verbose)
    except ValueError as e:
        if verbose:
            print(f"\n⚠️ تخطي اكتشاف الأنماط: {e}")

    # ── تشخيص النزاهة (Naive Baseline × Lag × الانحياز) ─────────────────────
    integrity_df = None
    if run_integrity_check:
        integrity_df = run_integrity_diagnostics(per_asset, target_specs, out_dir=out_dir, verbose=verbose)

    # ── مقاييس أداء تداول إضافية (Sharpe/Sortino/Profit Factor/SQN) ─────────
    trading_metrics = None
    if 'close' in continuous_df['target'].unique() if not continuous_df.empty else False:
        close_df = continuous_df[continuous_df['target'] == 'close'].copy()
        # العائد الفعلي للاستراتيجية: إن كان التوقع "شراء" (pred>entry) نربح true_change،
        # وإن كان "بيع" (pred<entry) نربح عكس true_change
        true_change_pct = ((close_df['true'] - close_df['entry']) / (np.abs(close_df['entry']) + 1e-7)) * 100
        direction = np.sign(close_df['predicted_change'])
        ts = close_df['timestamp'] if 'timestamp' in close_df else pd.Series(np.nan, index=close_df.index)
        neutral = bool(pnl_market_neutral) and ts.notna().all()
        if neutral:
            true_change_pct = true_change_pct - true_change_pct.groupby(ts.values).transform('mean')
        strategy_returns = direction * true_change_pct - trade_cost_pct   # بعد تكلفة الذهاب والإياب
        if ts.notna().all() and ts.nunique() > 2:
            # ✅ محفظة متساوية الأوزان لكل فترة: صفقات كل العملات في نفس اللحظة تُنفَّذ معاً برأس مال
            # مقسوم بالتساوي، فعائد الفترة = متوسط عوائدها. سابقاً كانت كل صفقة تُركَّب على رأس المال
            # بعد سابقتها حتى لو كانت في نفس اليوم (ومرتَّبة حسب العملة لا الزمن) — منحنى مستحيل التنفيذ.
            period_returns = strategy_returns.groupby(ts.values).mean().sort_index()
            steps = np.diff(period_returns.index.values.astype(np.float64))
            step_ns = float(np.median(steps)) if len(steps) else 0.0
            # طوابع بالنانوثانية (last_candles). فاصل < دقيقة = ليست أوقاتاً حقيقية (مثلاً أرقام تسلسلية) — بلا تسنين.
            periods_per_year = (365.25 * 86400e9 / step_ns) if step_ns >= 60e9 else None
            # risk_per_trade=1.0: عائد الفترة هنا عائد محفظة كاملة (نفس risk المرسوم أدناه)، لا صفقة بحجم جزئي
            trading_metrics = compute_trading_performance_metrics(period_returns.values,
                                                                  periods_per_year=periods_per_year,
                                                                  risk_per_trade=1.0)
            plot_returns, risk = period_returns.values, 1.0
            title = f"Equity Curve — Close (equal-weight portfolio per period, cost {trade_cost_pct}%)"
            basis = f"محفظة متساوية الأوزان: {len(period_returns)} فترة، {len(close_df)} صفقة، بعد تكلفة {trade_cost_pct}%"
            if neutral:
                basis += " — محايدة للسوق: عائد كل صفقة ناقص متوسط كل العملات في نفس الفترة"
                title += " — market-neutral"
        else:
            trading_metrics = compute_trading_performance_metrics(strategy_returns.values)
            plot_returns, risk = strategy_returns.values, 0.1
            title = "Equity Curve — Close (per trade, NO timestamps: sequential, not executable)"
            basis = "⚠️ بلا طوابع زمنية: صفقات متتالية — ليس منحنى قابلاً للتنفيذ"

        if verbose and trading_metrics:
            print("\n" + "=" * 100)
            print("📈 مقاييس أداء تداول إضافية (Tearsheet-style) — هدف Close، كل العملات مجمّعة")
            print(f"   ({basis})")
            print("=" * 100)
            print_performance_verdict(trading_metrics)

        if trading_metrics and ts.notna().all() and ts.nunique() > 2:
            try:   # دلالة إحصائية + قواعد أساس بنفس التكلفة (لا تُسقط التحليل إن فشلت)
                trading_metrics['significance'] = signal_significance_report(
                    close_df, trade_cost_pct=trade_cost_pct, market_neutral=neutral,
                    periods_per_year=periods_per_year, verbose=verbose)
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ تعذّر تقرير الدلالة الإحصائية: {e}")

        if make_plots and trading_metrics:
            eq = simulate_equity_curve(plot_returns, risk_per_trade=risk)
            plot_equity_curve(eq['equity'], eq['drawdown_pct'], title=title, save_dir=out_dir)

    if verbose:
        print("\n" + "=" * 100)
        print("✅ اكتمل التحليل الشامل — الملخص التنفيذي")
        print("=" * 100)
        overall_trust = trust['overall'].get('trust_score_0_100')
        overall_acc = flat_df['correct'].mean() * 100
        print(f"   • إجمالي العينات المحللة: {len(flat_df):,}")
        _lo, _hi = wilson_ci(flat_df['correct'].sum(), len(flat_df))
        print(f"   • الدقة/نسبة النجاح العامة: {overall_acc:.2f}%  (Wilson 95%: {_lo * 100:.1f}–{_hi * 100:.1f}%)")
        _cov = test_results.get('coverage') or {}
        if _cov.get('test_rows'):
            print(f"   • التغطية: {_cov['evaluated_rows']:,} من {_cov['test_rows']:,} صف اختبار، "
                  f"{_cov.get('evaluated_assets', 0)} من {_cov.get('test_assets', 0)} عملة")
        if 'unc_clipped' in flat_df.columns and flat_df['unc_clipped'].any():
            print(f"   • عدم اليقين المقصوص بحدّ NIG_UNC_MAX={NIG_UNC_MAX}: {int(flat_df['unc_clipped'].sum()):,} صف "
                  f"({flat_df['unc_clipped'].mean() * 100:.2f}%) — عيّنات خارج التوزيع")
        _cal = trust.get('calibration')
        if _cal is not None and not _cal.empty:
            print("   • ECE قبل ← بعد المعايرة: " + " | ".join(
                f"{r['target']}: {r['ece_before']:.3f}←{r['ece_after']:.3f} ({r['signal']})" for _, r in _cal.iterrows()))
        if trade_selection is not None:
            print(f"   • الصفقات المؤهلة بعد Edge Filter: {len(trade_selection):,} صفقة")
        if overall_trust is not None:
            print(f"   • درجة الثقة والفهم (Trust Score): {overall_trust:.1f}/100")
        if not test_results['verification_summary'].empty:
            failed = (test_results['verification_summary']['status'] == 'FAILED').sum()
            print(f"   • حالة التحقق من فك التشفير: {'✅ سليم بالكامل' if failed == 0 else f'❌ {failed} فشل'}")
        if integrity_df is not None and not integrity_df.empty:
            healthy_pct = (integrity_df['mae_improvement_%'] > 10).mean() * 100
            print(f"   • نسبة التركيبات (عملة/هدف) التي تتفوق فعلياً على Naive Baseline: {healthy_pct:.0f}%")
        if trading_metrics:
            print(f"   • Sharpe Ratio (Close، {'سنوي' if trading_metrics.get('sharpe_basis') == 'annualized' else 'لكل صفقة'}): {trading_metrics['sharpe_ratio']:.2f} | "
                  f"Max Drawdown: {trading_metrics['max_drawdown_pct']:.1f}%")
        print(f"\n📁 كل الجداول/الرسوم الكبيرة محفوظة في: {os.path.abspath(out_dir)}")

    return {
        **test_results,
        'flat_df': flat_df,
        'trust_report': trust,
        'calibrators': calibrators,
        'movement_analysis': movement_analysis,
        'confidence_analysis': confidence_analysis,
        'trade_candidates': trade_candidates,
        'trade_selection': trade_selection,
        'batch_analysis': batch_analysis,
        'pattern_discovery': patterns,
        'integrity_diagnostics': integrity_df,
        'trading_metrics': trading_metrics,
    }


# ══════════════════════════════════════════════════════════════════════════
# 🚀 مثال استخدام
# ══════════════════════════════════════════════════════════════════════════
#
# specs = DEFAULT_PRICE_TARGETS  # أو + [make_categorical_spec(...)] لأهداف فئوية
# full_results = run_full_analysis(
#     model=model,
#     test_dict=test_dict,
#     timeframes=['1h', '4h', '1D'],
#     target_specs=specs,
#     out_dir='analysis_outputs',   # كل الجداول/الرسوم الكبيرة تُحفظ هنا
# )
#
# # الوصول للتقارير الفردية:
# full_results['trust_report']['by_group']         # جدول الثقة حسب كل هدف
# full_results['movement_analysis']                 # نجاح التوقع حسب نوع/حجم الحركة
# full_results['batch_analysis']                    # اكتشاف الانحدار عبر الدفعات
# full_results['integrity_diagnostics']              # Naive baseline / lag / bias
# full_results['trading_metrics']                    # Sharpe/Sortino/Profit Factor/SQN
# full_results['trade_selection']                    # الصفقات بعد Edge Filter + الترتيب
# full_results['pattern_discovery']['tree_rules']    # قواعد نجاح/فشل مقروءة
# full_results['pattern_discovery']['clusters']       # الأنماط السلوكية المكتشفة
