"""
PURPOSE:  save_or_print / save_figure: large tables and figures go to analysis_outputs/ with a short preview instead of flooding the notebook.
TAGS:     save_or_print, save_figure, analysis_outputs, DEFAULT_OUTPUT_DIR, DEFAULT_MAX_ROWS, output management
PITFALLS: Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.

## 2️⃣ 💾 إدارة المخرجات: `save_or_print` و `save_figure`

ضع هذه الخلية مبكراً — تعتمد عليها كل التقارير والرسوم اللاحقة لتفادي إغراق الـ notebook بمخرجات ضخمة (كل شيء كبير يُحفظ كاملاً في `analysis_outputs/` مع معاينة مختصرة هنا).
"""
# ═══════════════════════════════════════════════════════════════════════════
# 💾 إدارة المخرجات: طباعة مختصرة + حفظ كامل في ملف عند كِبر الحجم
# ═══════════════════════════════════════════════════════════════════════════
#
# المشكلة: بعض التقارير (مثل تحليل كل عملة × كل دفعة × كل عيّنة شاذة) قد
# تنتج مئات/آلاف الأسطر، مما "يُغرق" الـ notebook ويصعّب القراءة. الحل هنا:
# أي DataFrame/نص يتجاوز حداً معيّناً يُحفظ كاملاً في ملف (CSV أو TXT) داخل
# مجلد analysis_outputs/، وتُطبع فقط معاينة مختصرة + مسار الملف.
# ═══════════════════════════════════════════════════════════════════════════

import os
from datetime import datetime

DEFAULT_OUTPUT_DIR = "analysis_outputs"
DEFAULT_MAX_ROWS = 40
DEFAULT_MAX_CHARS = 4000


def ensure_output_dir(out_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_or_print(
    data: Union[pd.DataFrame, str],
    name: str,
    out_dir: str = DEFAULT_OUTPUT_DIR,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_chars: int = DEFAULT_MAX_CHARS,
    verbose: bool = True,
    timestamp: bool = False,
) -> Optional[str]:
    """
    يطبع البيانات مباشرة إن كانت صغيرة، أو يحفظها كاملة في ملف ويطبع معاينة
    مختصرة + مسار الملف إن كانت كبيرة. يقلّل التشتت في مخرجات الـ notebook
    دون فقدان أي تفصيل (كل شيء محفوظ في analysis_outputs/).

    Args:
        data: DataFrame أو نص طويل
        name: اسم وصفي يُستخدم كاسم ملف (بدون امتداد)
        out_dir: المجلد الذي تُحفظ فيه الملفات الكبيرة
        max_rows: أقصى عدد صفوف يُطبع مباشرة (لـ DataFrame)
        max_chars: أقصى عدد حروف يُطبع مباشرة (لنص)
        timestamp: أضف طابعاً زمنياً لاسم الملف (لتفادي الكتابة فوق ملف سابق)

    Returns:
        مسار الملف المحفوظ، أو None إن طُبعت البيانات مباشرة بدون حفظ
    """
    ensure_output_dir(out_dir)
    suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if timestamp else ""

    if isinstance(data, pd.DataFrame):
        if len(data) > max_rows:
            path = os.path.join(out_dir, f"{name}{suffix}.csv")
            data.to_csv(path, index=False, encoding="utf-8-sig")
            if verbose:
                print(f"📄 النتيجة كبيرة ({len(data):,} صف) → حُفظت كاملة في: {path}")
                print(f"   معاينة أول {min(max_rows, len(data))} صف:")
                print(data.head(max_rows).to_string(index=False))
                if len(data) > max_rows:
                    print(f"   ... ({len(data) - max_rows:,} صف إضافي في الملف)")
            return path
        else:
            if verbose:
                print(data.to_string(index=False))
            return None

    elif isinstance(data, str):
        if len(data) > max_chars:
            path = os.path.join(out_dir, f"{name}{suffix}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
            if verbose:
                print(f"📄 النص طويل ({len(data):,} حرف) → حُفظ كاملاً في: {path}")
                print(data[:max_chars])
                print(f"   ... [تم الاقتصاص هنا — راجع الملف للنص الكامل]")
            return path
        else:
            if verbose:
                print(data)
            return None

    else:
        raise TypeError("save_or_print يدعم فقط pd.DataFrame أو str")


def save_figure(fig, name: str, out_dir: str = DEFAULT_OUTPUT_DIR, dpi: int = 130) -> str:
    """يحفظ أي matplotlib figure في analysis_outputs/figures/ ويُرجع المسار."""
    fig_dir = ensure_output_dir(os.path.join(out_dir, "figures"))
    path = os.path.join(fig_dir, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
