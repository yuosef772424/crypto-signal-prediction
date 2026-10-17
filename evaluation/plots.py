"""
PURPOSE:  Visual reports: calibration curve, confidence vs error, error distribution, actual vs predicted, lag correlation, confusion matrix, importances, clusters, equity curve, batch performance.
TAGS:     plot_calibration_curve, plot_equity_curve, plot_confusion_matrix_heatmap, matplotlib, figures, save_figure
PITFALLS: Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.

## 1️⃣4️⃣ 📊 الرسوم البيانية (Visual Reports)

كل دالة هنا تعرض الرسم inline **وتحفظه** تلقائياً كـ PNG في `analysis_outputs/figures/`.
"""
# ═══════════════════════════════════════════════════════════════════════════
# 📊 الرسوم البيانية (Visual Reports)
# ═══════════════════════════════════════════════════════════════════════════
#
# كل دالة رسم هنا: تعرض الرسم داخل الـ notebook مباشرة (plt.show) + تحفظه
# تلقائياً كملف PNG في analysis_outputs/figures/ (يفيد عند إنتاج عدد كبير
# من الرسوم دفعة واحدة عبر عدة عملات/أهداف، بدل تكديسها كلها في المخرجات).
#
# ملاحظة: عناوين/تسميات المحاور بالإنجليزية عمداً — matplotlib لا يدعم تشكيل
# الحروف العربية (RTL shaping) افتراضياً بدون مكتبات إضافية (arabic_reshaper).
# ═══════════════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_calibration_curve(reliability_df: pd.DataFrame, group_val=None, group_col: str = 'target',
                            title: Optional[str] = None, save_dir: str = DEFAULT_OUTPUT_DIR) -> Optional[str]:
    """مخطط المعايرة (Reliability Diagram): الثقة المتوقعة مقابل الدقة الفعلية."""
    df = reliability_df if group_val is None else reliability_df[reliability_df[group_col] == group_val]
    if df.empty:
        return None
    df = df.sort_values('avg_confidence')

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    ax.plot(df['avg_confidence'], df['avg_accuracy'], 'o-', color='steelblue', label='Model')
    for _, row in df.iterrows():
        ax.annotate(f"n={int(row['count'])}", (row['avg_confidence'], row['avg_accuracy']),
                    fontsize=7, alpha=0.6, xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel('Predicted Confidence')
    ax.set_ylabel('Actual Accuracy')
    ax.set_title(title or f"Calibration Curve{f' — {group_val}' if group_val else ''}")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    path = save_figure(fig, f"calibration_curve_{group_val or 'overall'}", save_dir)
    plt.show()
    plt.close(fig)
    return path


def plot_confidence_vs_error(df: pd.DataFrame, target: Optional[str] = None,
                              error_col: str = 'abs_error', save_dir: str = DEFAULT_OUTPUT_DIR) -> Optional[str]:
    """علاقة الثقة بالخطأ — يجب أن تظهر علاقة عكسية إن كان النموذج معايَراً جيداً."""
    work = df if target is None else df[df['target'] == target]
    work = work.dropna(subset=['confidence', error_col])
    if work.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    correct_mask = work['correct'].astype(bool) if 'correct' in work.columns else pd.Series(True, index=work.index)
    ax.scatter(work.loc[correct_mask, 'confidence'], work.loc[correct_mask, error_col],
               s=10, alpha=0.4, color='seagreen', label='Correct')
    ax.scatter(work.loc[~correct_mask, 'confidence'], work.loc[~correct_mask, error_col],
               s=10, alpha=0.4, color='crimson', label='Wrong')
    if work['confidence'].std() > 1e-9:
        z = np.polyfit(work['confidence'], work[error_col], 1)
        xs = np.linspace(work['confidence'].min(), work['confidence'].max(), 50)
        ax.plot(xs, np.polyval(z, xs), 'k--', alpha=0.7, label='Trend')
    ax.set_xlabel('Confidence')
    ax.set_ylabel(error_col)
    ax.set_title(f"Confidence vs Error{f' — {target}' if target else ''}")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    path = save_figure(fig, f"confidence_vs_error_{target or 'all'}", save_dir)
    plt.show()
    plt.close(fig)
    return path


def plot_error_distribution(df: pd.DataFrame, target: Optional[str] = None,
                             error_col: str = 'pct_error', save_dir: str = DEFAULT_OUTPUT_DIR) -> Optional[str]:
    """توزيع الأخطاء — يكشف عن ذيول ثقيلة (Fat tails) أو انحياز في التوزيع."""
    work = df if target is None else df[df['target'] == target]
    vals = work[error_col].dropna()
    if vals.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(vals, bins=40, color='steelblue', alpha=0.8, edgecolor='white')
    ax.axvline(vals.mean(), color='crimson', ls='--', label=f'Mean={vals.mean():.3f}')
    ax.axvline(vals.median(), color='orange', ls='--', label=f'Median={vals.median():.3f}')
    ax.set_xlabel(error_col)
    ax.set_ylabel('Count')
    ax.set_title(f"Error Distribution{f' — {target}' if target else ''}")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    path = save_figure(fig, f"error_distribution_{target or 'all'}", save_dir)
    plt.show()
    plt.close(fig)
    return path


def plot_actual_vs_predicted(y_actual: np.ndarray, y_pred: np.ndarray, y_naive: Optional[np.ndarray] = None,
                              title: str = "Actual vs Predicted", save_dir: str = DEFAULT_OUTPUT_DIR,
                              filename: Optional[str] = None) -> str:
    """منحنى القيم الفعلية مقابل المتوقعة عبر الزمن (+ naive baseline اختياري)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_actual, c="limegreen", lw=2.2, alpha=0.85, label="Actual", zorder=5)
    ax.plot(y_pred, c="darkorange", lw=1.8, label="Predicted", alpha=0.9)
    if y_naive is not None:
        ax.plot(y_naive, c="gray", lw=1.3, ls="--", label="Naive Baseline (t-1)", alpha=0.7)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Value")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    path = save_figure(fig, filename or f"actual_vs_predicted_{title}".replace(' ', '_'), save_dir)
    plt.show()
    plt.close(fig)
    return path


def plot_lag_correlation(lags: List[int], corrs: List[float], best_lag: int,
                          title: str = "Cross-Correlation vs Lag", save_dir: str = DEFAULT_OUTPUT_DIR,
                          filename: Optional[str] = None) -> str:
    """أعلى ارتباط عند أي إزاحة زمنية؟ (0 = صحي، غير 0 = نسخ متأخر مشبوه)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["crimson" if l == best_lag else "steelblue" for l in lags]
    ax.bar(lags, corrs, color=colors)
    ax.axvline(0, color="black", lw=1, ls=":")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Lag (steps)")
    ax.set_ylabel("Correlation")
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    path = save_figure(fig, filename or f"lag_correlation_{title}".replace(' ', '_'), save_dir)
    plt.show()
    plt.close(fig)
    return path


def plot_confusion_matrix_heatmap(cm: np.ndarray, class_names: List[str], title: str = "Confusion Matrix",
                                   save_dir: str = DEFAULT_OUTPUT_DIR, filename: Optional[str] = None) -> str:
    """خريطة حرارية لمصفوفة الارتباك — للأهداف الفئوية."""
    fig, ax = plt.subplots(figsize=(1.2 * len(class_names) + 2, 1.2 * len(class_names) + 2))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title, fontweight="bold")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = save_figure(fig, filename or f"confusion_matrix_{title}".replace(' ', '_'), save_dir)
    plt.show()
    plt.close(fig)
    return path


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15,
                             save_dir: str = DEFAULT_OUTPUT_DIR, filename: str = "feature_importance") -> str:
    """أهمية الخصائص في التمييز بين النجاح والفشل (من detect_success_failure_patterns)."""
    top = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(top))))
    ax.barh(top['الخاصية'], top['الأهمية'], color='teal')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance (Random Forest)', fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    plt.tight_layout()
    path = save_figure(fig, filename, save_dir)
    plt.show()
    plt.close(fig)
    return path


def plot_clusters_2d(clustered_df: pd.DataFrame, feature_cols: List[str],
                      save_dir: str = DEFAULT_OUTPUT_DIR, filename: str = "clusters_pca") -> str:
    """يُسقط الأنماط السلوكية المكتشفة (K-Means) على مستوى ثنائي الأبعاد عبر PCA."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(clustered_df[feature_cols])
    coords = PCA(n_components=2, random_state=42).fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=clustered_df['cluster'],
                          cmap='tab10', s=15, alpha=0.6)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('Behavioral Clusters (PCA Projection)', fontweight="bold")
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best", fontsize=8)
    ax.add_artist(legend1)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    path = save_figure(fig, filename, save_dir)
    plt.show()
    plt.close(fig)
    return path


def plot_equity_curve(equity: np.ndarray, drawdown_pct: np.ndarray, title: str = "Equity Curve",
                       save_dir: str = DEFAULT_OUTPUT_DIR, filename: Optional[str] = None) -> str:
    """منحنى رأس المال + منحنى الـ Drawdown المقابل (نمط tearsheet قياسي)."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    axes[0].plot(equity, color='steelblue', lw=1.8)
    axes[0].set_title(title, fontweight="bold")
    axes[0].set_ylabel('Equity ($)')
    axes[0].grid(True, alpha=0.3, linestyle='--')

    axes[1].fill_between(range(len(drawdown_pct)), drawdown_pct, 0, color='crimson', alpha=0.5)
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].set_xlabel('Trade #')
    axes[1].grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    path = save_figure(fig, filename or f"equity_curve_{title}".replace(' ', '_'), save_dir)
    plt.show()
    plt.close(fig)
    return path


def plot_batch_performance(batch_stats: pd.DataFrame, z_threshold: float = 1.5,
                            title: str = "Performance Across Batches", save_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """نسبة النجاح عبر الدفعات المتتالية مع تمييز الدفعات الشاذة (اكتشاف الانحدار)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = np.where(batch_stats['z_score'] <= -z_threshold, 'crimson',
              np.where(batch_stats['z_score'] >= z_threshold, 'seagreen', 'steelblue'))
    ax.bar(batch_stats['batch_id'], batch_stats['نسبة_نجاح'], color=colors)
    ax.axhline(batch_stats['نسبة_نجاح'].mean(), color='black', ls='--', alpha=0.6, label='Mean')
    ax.set_xlabel('Batch #')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    path = save_figure(fig, "batch_performance", save_dir)
    plt.show()
    plt.close(fig)
    return path
