"""
PURPOSE:  Tearsheet-style trading metrics (Sharpe, Sortino, Omega, profit factor, SQN, drawdown), equity curve and signal significance report.
TAGS:     compute_trading_performance_metrics, simulate_equity_curve, signal_significance_report, Sharpe, drawdown, tearsheet, equity curve
PITFALLS: Two self-tests (_test_*) run at load, as the old notebook did. Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.

## 1️⃣3️⃣ 📈 مقاييس أداء تداول إضافية (Tearsheet-style Metrics)

Sharpe, Sortino, Omega, Profit Factor, SQN, Max Drawdown + محاكاة منحنى رأس المال — مبنية على ممارسات تقارير الـ backtest المالية القياسية (Sharpe/Drawdown/Profit Factor هي المقاييس الأساسية في أي "tearsheet" تداول احترافي).
"""
# ═══════════════════════════════════════════════════════════════════════════
# 📈 مقاييس أداء تداول إضافية (Tearsheet-style metrics)
# ═══════════════════════════════════════════════════════════════════════════
#
# مبنية على الممارسات القياسية في تقارير الـ backtest المالية (tearsheets):
# Sharpe, Sortino, Omega, Profit Factor, SQN, Max Drawdown — بالإضافة لمنحنى
# رأس المال (Equity Curve) نفسه لكل استراتيجية.
# ═══════════════════════════════════════════════════════════════════════════

def compute_trading_performance_metrics(returns_pct: np.ndarray, risk_free_pct: float = 0.0,
                                        risk_per_trade: float = 0.1,
                                        periods_per_year: Optional[float] = None) -> Dict[str, float]:
    """
    يحسب مجموعة مقاييس أداء قياسية من سلسلة عوائد الصفقات (بالنسبة المئوية).

    • Sharpe Ratio    : العائد المعدَّل بالمخاطرة (متوسط العائد / انحرافه المعياري)
    • Sortino Ratio    : مثل Sharpe لكن يُعاقب فقط التذبذب السلبي (الخسائر)
    • Max Drawdown %   : أكبر تراجع من قمة إلى قاع في منحنى رأس المال
    • Profit Factor    : إجمالي الأرباح / إجمالي الخسائر (>1 مربح، >2 قوي)
    • Win Rate %       : نسبة الصفقات الرابحة
    • Omega Ratio      : نسبة الأرباح المرجَّحة باحتمالها إلى الخسائر المرجَّحة (>1 جيد)
    • SQN              : System Quality Number لـ Van Tharp (>2 قابل للتداول، >3 ممتاز)

    periods_per_year: إن كانت returns_pct عوائد فترات متتالية (مثلاً عائد المحفظة اليومي) يُحسب
    Sharpe/Sortino سنوياً بـ √periods_per_year. بلاها يُعاد Sharpe **لكل صفقة** (متوسط/انحراف بلا
    تضخيم). ⚠️ سابقاً كان Sharpe يُضرب بـ √(عدد الصفقات) — أي إحصاء t لا نسبة Sharpe، يكبر مع عدد
    الصفقات بلا حدّ (39,545 صفقة بميزة ضئيلة → "Sharpe 4.4 ✅"). SQN بسقف N=100 (SQN100) لنفس السبب.
    """
    r = np.asarray(returns_pct, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return {}

    # ✅ قص عند -100% (لا يمكن خسارة أكثر من 100% من المركز المراهن عليه في
    # محاكاة بلا رافعة مالية). بلا هذا القص، صفقة واحدة بعائد أقل من -100%
    # (مثلاً: النموذج توقّع هبوطاً لأصل قفز أكثر من 100% في نفس الفترة — وارد
    # فعلياً في عملات ميمية متطرفة التقلّب) تقلب حاصل `equity` في `np.cumprod` إلى
    # قيمة سالبة، فيُفسد كل قيمة لاحقة (الضرب تراكمي) ويُنتج `max_drawdown_pct`
    # تتجاوز -100% رياضياً بلا معنى اقتصادي (لا يوجد "تراجع" أكثر من خسارة كل
    # رأس المال فعلياً).
    r = np.maximum(r, -100.0)

    excess = r - risk_free_pct
    ann = np.sqrt(periods_per_year) if periods_per_year else 1.0
    sharpe = float(np.mean(excess) / (np.std(excess) + 1e-9)) * ann

    downside = excess[excess < 0]
    sortino = float(np.mean(excess) / (np.std(downside) + 1e-9)) * ann if len(downside) > 0 else float('inf')

    # ✅ إصلاح (24 سبتمبر 2026): كانت `equity` هنا تُحسَب بتركيب كامل —
    # `100 * cumprod(1 + r/100)`، أي المراهنة بكل رأس المال في كل صفقة. هذا
    # يختلف كلياً عن `simulate_equity_curve` (المُستخدَمة فعلياً لرسم منحنى
    # رأس المال المعروض على المستخدم، بحجم مركز جزئي `risk_per_trade`)، وهو
    # مضمون الانهيار رياضياً مهما كانت جودة الإشارة: بمجرّد صفقة واحدة تُقصّ
    # عند -100% (واردة فعلياً، راجع الحاشية أعلاه) يصبح `cumprod` صفراً إلى
    # الأبد بعدها، فـ`total_return_pct`/`max_drawdown_pct` يخرجان -100.00%
    # بالضبط بصرف النظر عن أداء بقية الصفقات — بالضبط التناقض الذي ظهر عملياً:
    # رسم بياني ينمو إلى ملايين الدولارات، مقابل "إجمالي العائد: -100%" هنا.
    # الإصلاح: استخدام *نفس* معادلة `simulate_equity_curve` (حجم مركز جزئي)
    # لحساب `max_drawdown_pct`/`total_return_pct`، فتصف كلتا القيمتين نفس
    # الاستراتيجية الواقعية المعروضة للمستخدم، لا نموذجَين متناقضين.
    eq_curve = simulate_equity_curve(r, initial_capital=100.0, risk_per_trade=risk_per_trade)
    equity = eq_curve['equity']
    max_drawdown = float(eq_curve['drawdown_pct'].min())

    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    profit_factor = float(gains / losses) if losses > 0 else float('inf')

    win_rate = float(np.mean(r > 0)) * 100

    threshold = 0.0
    gains_omega = np.sum(np.maximum(r - threshold, 0))
    losses_omega = np.sum(np.maximum(threshold - r, 0))
    omega = float(gains_omega / losses_omega) if losses_omega > 0 else float('inf')

    sqn = float(np.mean(r) / (np.std(r) + 1e-9)) * np.sqrt(min(len(r), 100))

    return {
        'n_trades': len(r),
        'sharpe_basis': 'annualized' if periods_per_year else 'per_trade',
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown_pct': max_drawdown,
        'profit_factor': profit_factor,
        'win_rate_pct': win_rate,
        'omega_ratio': omega,
        'sqn': sqn,
        'total_return_pct': float(equity[-1] - 100.0),
        'avg_return_per_trade_pct': float(np.mean(r)),
        'risk_per_trade': risk_per_trade,
    }


def print_performance_verdict(metrics: Dict[str, float]) -> None:
    """طباعة حكم نوعي سريع على مقاييس الأداء (يشبه تفسير tearsheet مالي قياسي)."""
    if not metrics:
        print("⚠️ لا توجد بيانات كافية لحساب مقاييس الأداء")
        return
    if metrics.get('sharpe_basis') == 'annualized':
        print(f"   • Sharpe Ratio (سنوي): {metrics['sharpe_ratio']:.3f}  "
              f"({'✅ جيد' if metrics['sharpe_ratio'] > 1 else '⚠️ ضعيف' if metrics['sharpe_ratio'] > 0 else '❌ سلبي'})")
    else:
        print(f"   • Sharpe (لكل صفقة، غير سنوي): {metrics['sharpe_ratio']:.4f}")
    print(f"   • Sortino Ratio:      {metrics['sortino_ratio']:.3f}")
    print(f"   • Max Drawdown:       {metrics['max_drawdown_pct']:.2f}%  "
          f"({'✅ محتمل' if metrics['max_drawdown_pct'] > -20 else '⚠️ مرتفع'})")
    print(f"   • Profit Factor:      {metrics['profit_factor']:.3f}  "
          f"({'✅ قوي' if metrics['profit_factor'] > 2 else '⚠️ مقبول' if metrics['profit_factor'] > 1 else '❌ خاسر'})")
    print(f"   • Omega Ratio:        {metrics['omega_ratio']:.3f}")
    print(f"   • SQN (Van Tharp):    {metrics['sqn']:.3f}  "
          f"({'✅ ممتاز' if metrics['sqn'] > 3 else '⚠️ قابل للتداول' if metrics['sqn'] > 2 else '❌ ضعيف'})")
    unit = "فترة (محفظة)" if metrics.get('sharpe_basis') == 'annualized' else "صفقة"
    print(f"   • إجمالي العائد:      {metrics['total_return_pct']:+.2f}% على {metrics['n_trades']} {unit}")


def simulate_equity_curve(returns_pct: np.ndarray, initial_capital: float = 10000.0,
                           risk_per_trade: float = 0.1) -> Dict[str, np.ndarray]:
    """
    يبني منحنى رأس المال (equity curve) خطوة بخطوة من سلسلة عوائد الصفقات،
    مع منحنى الـ drawdown المقابل — أساس رسم `plot_equity_curve`.
    """
    r = np.asarray(returns_pct, dtype=np.float64)
    # ✅ نفس قص -100% المطبّق أعلاه (compute_trading_performance_metrics) — عائد واحد
    # لا يمكن أن يقل عن -100% اقتصادياً (السعر لا يهبط تحت الصفر)، فلا معنى
    # لإطلاق قيمة أقل تُمرّر دون قص إلى دالة المحاكاة (وإن كانت `risk_per_trade`
    # الافتراضي يجعلها مأمونة رياضياً هنا دائماً لوحدها إلا في حالات `risk_per_trade`
    # مرتفعة جداً).
    r = np.maximum(r, -100.0)
    equity = np.empty(len(r) + 1)
    equity[0] = initial_capital
    for i, ret in enumerate(r):
        trade_amount = equity[i] * risk_per_trade
        equity[i + 1] = equity[i] + trade_amount * (ret / 100.0)

    running_max = np.maximum.accumulate(equity)
    drawdown_pct = (equity - running_max) / running_max * 100
    return {'equity': equity, 'drawdown_pct': drawdown_pct, 'running_max': running_max}


def _test_drawdown_never_exceeds_100pct():
    # صفقة واحدة بعائد أقل من -100% (رهان خاطئ في الاتجاه على أصل تحرّك
    # أكبر من 100%) يجب ألاّ تنتج 'max_drawdown'/`equity` منطقيين أبداً —
    # قبل الإصلاح كان `compute_trading_performance_metrics` يُنتج -138% وequity
    # سالبة من صفقة واحدة فقط بهذا السيناريو.
    rng = np.random.default_rng(1)
    r = rng.normal(0, 5, 200)
    r[50] = -150.0
    metrics = compute_trading_performance_metrics(r)
    assert metrics['max_drawdown_pct'] >= -100.0, (
        f"max_drawdown_pct تجاوز -100% ({metrics['max_drawdown_pct']}) — صفقة <-100% لم تُقصَّ.")
    eq = simulate_equity_curve(r)
    assert eq['drawdown_pct'].min() >= -100.0, (
        f"drawdown_pct تجاوز -100% ({eq['drawdown_pct'].min()}) في simulate_equity_curve.")
    assert np.all(eq['equity'] > 0), "equity أصبحت غير موجبة بسبب صفقة أقل من -100% غير مقصوصة."
    print("✅ compute_trading_performance_metrics/simulate_equity_curve: drawdown/equity مُقيّدان بحدود منطقية (≥-100%) حتى مع صفقة بعائد أقل من -100%.")
    return True


def _test_tearsheet_matches_plotted_equity_curve():
    """اكتُشف عملياً (24 سبتمبر 2026، تشغيل حقيقي على 39,545 صفقة): الرسم
    البياني (`simulate_equity_curve`) نما إلى ملايين الدولارات، بينما
    `compute_trading_performance_metrics` أعطى `total_return_pct`/
    `max_drawdown_pct` = -100.00% بالضبط لنفس البيانات — تناقض صريح، سببه
    حساب `equity` بنموذجَين مختلفين تماماً لحجم المركز (تركيب كامل مقابل
    حجم جزئي). يتحقّق هذا الاختبار من أمرين: (أ) صفقة واحدة مقصوصة عند
    -100% وسط مئات الصفقات الرابحة **لا تفرض** -100% على الإجمالي بعد
    الإصلاح (حجم مركز جزئي يفقد `risk_per_trade` فقط من رأس المال، لا كلّه)،
    (ب) نفس `risk_per_trade` يُنتج نفس `max_drawdown_pct`/`equity[-1]`
    تماماً في الدالتين معاً — لا نموذجين متناقضين بعد الآن."""
    rng = np.random.default_rng(2)
    r = rng.normal(1.5, 3, 500)  # انحياز موجب — استراتيجية رابحة إجمالاً
    r[250] = -150.0  # صفقة كارثية واحدة وسط 500 صفقة رابحة غالباً

    metrics = compute_trading_performance_metrics(r, risk_per_trade=0.1)
    assert metrics['total_return_pct'] > -50.0, (
        f"صفقة كارثية واحدة وسط 500 صفقة رابحة غالباً لا يجب أن تفرض عائداً "
        f"إجمالياً كارثياً بحجم مركز جزئي (total_return_pct="
        f"{metrics['total_return_pct']:.2f}%).")

    eq_direct = simulate_equity_curve(r, initial_capital=100.0, risk_per_trade=0.1)
    assert abs(metrics['max_drawdown_pct'] - eq_direct['drawdown_pct'].min()) < 1e-9, (
        "max_drawdown_pct في compute_trading_performance_metrics يجب أن يطابق "
        "simulate_equity_curve تماماً بنفس risk_per_trade — لا نموذجين مختلفين.")
    assert abs(metrics['total_return_pct'] - (eq_direct['equity'][-1] - 100.0)) < 1e-9, (
        "total_return_pct يجب أن يطابق equity[-1] من simulate_equity_curve تماماً.")
    print("✅ _test_tearsheet_matches_plotted_equity_curve: total_return_pct/max_drawdown_pct "
          "الآن من نفس نموذج حجم المركز المعروض في الرسم البياني — لا تناقض.")
    return True


_test_drawdown_never_exceeds_100pct()
_test_tearsheet_matches_plotted_equity_curve()


# ═══════════════════════════════════════════════════════════════════════════
# 🧪 هل الربح إشارة أم حظ؟ — دلالة إحصائية + استراتيجيات أساس بنفس التكاليف
# ═══════════════════════════════════════════════════════════════════════════
# Sharpe مرتفع على عشرات الفترات قد يكون حظاً: الخطأ المعياري لـ Sharpe اليومي ≈ 1/√T. هذا التقرير يجيب
# عن ثلاثة أسئلة قبل الوثوق بأي منحنى رأس مال:
#   1) هل المتوسط دالّ؟ إحصاء t لعوائد الفترات (= SQN بلا سقف) + فترة ثقة bootstrap لـ Sharpe.
#   2) هل الإشارة أفضل من خلط عشوائي لنفس الإشارات؟ اختبار تبديل (permutation): تُخلط اتجاهات النموذج بين
#      العملات داخل كل فترة (نفس عدد الشراء/البيع كل يوم، نفس العوائد) — لا افتراض عن التوزيع.
#   3) هل النموذج يضيف شيئاً فوق قاعدة بسيطة؟ الزخم (اتجاه أمس) والارتداد (عكسه) بنفس التكلفة والحياد،
#      ونسبة اتفاق النموذج مع الارتداد (إن كانت ~100% فالنموذج «يعيد اكتشاف» الارتداد فقط).
# + تقسيم الفترة نصفين: ربح متركّز في البداية فقط علامة نظام انتهى أو حظ.

def _period_stats(p: np.ndarray, periods_per_year: Optional[float], n_boot: int = 2000,
                  rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    from scipy import stats as _st
    p = np.asarray(p, dtype=np.float64)
    p = p[np.isfinite(p)]
    T = len(p)
    if T < 3:
        return {'n_periods': T}
    sd = float(np.std(p, ddof=1))
    m = float(np.mean(p))
    t = m / (sd + 1e-12) * np.sqrt(T)
    ann = np.sqrt(periods_per_year) if periods_per_year else 1.0
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, T, size=(n_boot, T))
    bs = p[idx]
    boot_sharpe = bs.mean(1) / (bs.std(1, ddof=1) + 1e-12) * ann
    half = T // 2
    sh = lambda q: float(np.mean(q) / (np.std(q, ddof=1) + 1e-12) * ann) if len(q) > 2 else float('nan')
    return {'n_periods': T, 'mean_pct': m, 'sharpe': m / (sd + 1e-12) * ann, 't_stat': float(t),
            'p_value_one_sided': float(1 - _st.t.cdf(t, T - 1)),
            'sharpe_ci95': (float(np.percentile(boot_sharpe, 2.5)), float(np.percentile(boot_sharpe, 97.5))),
            'first_half_sharpe': sh(p[:half]), 'second_half_sharpe': sh(p[half:]),
            'first_half_sum_pct': float(np.sum(p[:half])), 'second_half_sum_pct': float(np.sum(p[half:]))}


def signal_significance_report(close_df: pd.DataFrame, trade_cost_pct: float = 0.08, market_neutral: bool = True,
                               periods_per_year: Optional[float] = None, n_perm: int = 1000, seed: int = 0,
                               verbose: bool = True) -> Dict[str, object]:
    """close_df: صفوف هدف close بأعمدة asset/entry/true/predicted_change/timestamp (flat_df في run_full_analysis)."""
    df = close_df[['asset', 'entry', 'true', 'predicted_change', 'timestamp']].dropna()
    df = df.sort_values(['timestamp', 'asset'], kind='mergesort').reset_index(drop=True)
    if df['timestamp'].nunique() < 3:
        return {}
    ret = ((df['true'] - df['entry']) / (df['entry'].abs() + 1e-7) * 100).values
    _, g = np.unique(df['timestamp'].values, return_inverse=True)
    if market_neutral:
        ret = ret - (np.bincount(g, ret) / np.bincount(g))[g]
    prev = df.groupby('asset')['entry'].shift(1).values
    past = (df['entry'].values - prev) / (np.abs(prev) + 1e-7)       # عائد أمس لنفس العملة (entry = إغلاق أمس)
    past_sign = np.where(np.isfinite(past), np.sign(past), 0.0)
    model_sign = np.sign(df['predicted_change'].values)

    def period_returns(sign):
        r = sign * ret - trade_cost_pct * (sign != 0)
        n = np.bincount(g, (sign != 0).astype(float))
        s = np.bincount(g, r)
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)

    rng = np.random.default_rng(seed)
    obs = period_returns(model_sign)
    out = {'model': _period_stats(obs, periods_per_year, rng=rng),
           'momentum': _period_stats(period_returns(past_sign), periods_per_year, rng=rng),
           'reversal': _period_stats(period_returns(-past_sign), periods_per_year, rng=rng)}
    # اختبار التبديل: الصفوف مرتّبة حسب الفترة، فـ lexsort(عشوائي، g) تبديل داخل كل فترة
    obs_mean = float(np.nanmean(obs))
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = np.lexsort((rng.random(len(g)), g))
        null[k] = np.nanmean(period_returns(model_sign[perm]))
    out['permutation'] = {'n_perm': n_perm, 'observed_mean_pct': obs_mean,
                          'null_mean_pct': float(null.mean()), 'null_p95_pct': float(np.percentile(null, 95)),
                          'p_value': float((1 + np.sum(null >= obs_mean)) / (1 + n_perm))}
    valid = (past_sign != 0) & (model_sign != 0)
    out['agreement_with_reversal_pct'] = float(np.mean(model_sign[valid] == -past_sign[valid]) * 100) if valid.any() else float('nan')

    if verbose:
        print("\n" + "=" * 100)
        print("🧪 هل الربح إشارة أم حظ؟ (عوائد الفترات بعد التكلفة" + (" ومحايدة للسوق" if market_neutral else "") + ")")
        print("=" * 100)
        print(f"{'الاستراتيجية':<12} {'Sharpe':>7} {'فترة ثقة 95%':>17} {'t':>6} {'p (t)':>7} {'نصف1 Σ%':>9} {'نصف2 Σ%':>9}")
        for name, lab in (('model', 'النموذج'), ('momentum', 'الزخم'), ('reversal', 'الارتداد')):
            s = out[name]
            if 'sharpe' not in s:
                continue
            lo, hi = s['sharpe_ci95']
            print(f"{lab:<12} {s['sharpe']:>7.2f} {f'[{lo:+.2f}, {hi:+.2f}]':>17} {s['t_stat']:>6.2f} "
                  f"{s['p_value_one_sided']:>7.3f} {s['first_half_sum_pct']:>+9.2f} {s['second_half_sum_pct']:>+9.2f}")
        pm = out['permutation']
        print(f"\n   • اختبار التبديل ({pm['n_perm']} خلطة لاتجاهات النموذج داخل كل فترة): متوسط الفترة "
              f"{pm['observed_mean_pct']:+.4f}% مقابل {pm['null_mean_pct']:+.4f}% للخلط (95%: {pm['null_p95_pct']:+.4f}%) "
              f"→ p = {pm['p_value']:.3f}")
        print(f"   • اتفاق النموذج مع قاعدة الارتداد: {out['agreement_with_reversal_pct']:.1f}% "
              "(50% = مستقل عنها، ~100% = يكرّرها)")
        m = out['model']
        if 'sharpe' in m:
            beats = all(m['sharpe'] > out[b].get('sharpe', -np.inf) for b in ('momentum', 'reversal'))
            # سؤالان مختلفان: المهارة (اتجاهات أفضل من خلطها عشوائياً — التكلفة متساوية في الطرفين) والربح بعد
            # التكلفة (t على العائد الصافي). مهارة بلا ربح دالّ = الإشارة حقيقية لكنها أصغر من التكاليف أو الفترة قصيرة.
            print("   • المهارة (أفضل من اتجاهات عشوائية): " + ("✅ نعم" if pm['p_value'] < 0.05 else "❌ غير مُثبتة")
                  + f" (p={pm['p_value']:.3f})")
            print("   • الربح بعد التكلفة: " + ("✅ دالّ" if m['t_stat'] >= 2 else "⚠️ غير دالّ بعد — قد يكون حظاً؛ يلزم فترة أطول")
                  + f" (t={m['t_stat']:.2f}؛ t≥2 يلزمه نحو {int(np.ceil(4 * m['n_periods'] / m['t_stat'] ** 2)) if m['t_stat'] > 0 else '∞'} فترة بنفس الأداء)")
            print("   • مقابل القواعد البسيطة: " + ("✅ يتفوّق على الزخم والارتداد" if beats else "⚠️ لا يتفوّق على كل قاعدة بسيطة"))
            if m['first_half_sum_pct'] > 0 >= m['second_half_sum_pct']:
                print("   • ⚠️ الربح كله في النصف الأول والنصف الثاني غير رابح — نظام انتهى أو حظ")
    return out


def _test_signal_significance_report():
    """إشارة حقيقية ⇒ p صغير وt>2؛ إشارة عشوائية ⇒ غير دالّة؛ الارتداد المزروع يُكتشف (اتفاق ~100%)."""
    rng = np.random.default_rng(3)
    T, A = 200, 40
    rows = []
    price = np.full(A, 100.0)
    for t in range(T):
        prev = price.copy()
        move = rng.normal(0, 0.02, A)
        price = price * (1 + move)
        for a in range(A):
            rows.append({'asset': f'A{a}', 'timestamp': float(t), 'entry': prev[a], 'true': price[a],
                         'truth_move': move[a]})
    df = pd.DataFrame(rows)
    df['predicted_change'] = np.where(rng.random(len(df)) < 0.60, np.sign(df['truth_move']),
                                      -np.sign(df['truth_move']))            # إصابة 60%
    good = signal_significance_report(df, trade_cost_pct=0.0, n_perm=200, verbose=False)
    assert good['model']['t_stat'] > 2 and good['permutation']['p_value'] < 0.05, good['model']
    df['predicted_change'] = rng.choice([-1.0, 1.0], len(df))
    rand = signal_significance_report(df, trade_cost_pct=0.0, n_perm=200, verbose=False)
    assert rand['permutation']['p_value'] > 0.05, rand['permutation']
    df = df.sort_values(['asset', 'timestamp'])
    df['predicted_change'] = -np.sign(df.groupby('asset')['entry'].diff().fillna(1.0))   # يكرّر الارتداد
    rev = signal_significance_report(df, trade_cost_pct=0.0, n_perm=50, verbose=False)
    assert rev['agreement_with_reversal_pct'] > 99, rev['agreement_with_reversal_pct']
    print("✅ signal_significance_report: الإشارة الحقيقية دالّة، العشوائية لا، والارتداد المكرَّر يُكتشف.")
    return True


_test_signal_significance_report()
