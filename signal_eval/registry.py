"""
PURPOSE:  Experiment registry: one permanent JSON record per hypothesis (id, claim, source track, status, full axis results incl. per_window): register_hypothesis, list_registry, get_hypothesis.
TAGS:     register_hypothesis, list_registry, get_hypothesis, experiment registry, DISCOVERY_TRACKS, REGISTRY_STATUSES, registry.json, hypothesis status
PITFALLS: source/status outside DISCOVERY_TRACKS/REGISTRY_STATUSES raise. Registering an existing id replaces the record entirely. Path is on Drive when mounted, else ./experiment_registry/ (git-ignored). Executed into the one shared signal_eval namespace by signal_eval/_loader.py (never imported on its own): names from other modules resolve at call time.

سجلّ التجارب — محفوظ كملف JSON (لا قاعدة بيانات) بنفس روح checkpoint
العمل الحالي في دفتر التحضير: بصيغة قابلة للقراءة والتحكّم اليدوي، وتُكتب
على القرص بعد كل تسجيل فوراً (لا في الذاكرة فقط) فتنجو من فقدان جلسة Colab.


## ٩) سجلّ التجارب (Experiment Registry)

المكوّن الثالث من "خطة بناء النظام" في خطة المشروع — لم يكن مبنياً بعد. كل
فرضية تُختبَر عبر محور هذا الدفتر تحصل على سجلّ واحد دائم: معرّف، صياغة
بسطر واحد، المصدر (أيّ من مسارات الاكتشاف الأربعة)، الحالة، ونتائج المحور
**كاملة** — لا رقم ملخّص وحده، بل جدول كل نافذة أيضاً، لأن رقماً واحداً
("IC=-0.08") لا يكفي للحكم لاحقاً على نافذة شاذّة أو نمط زمني فيها.

القاعدة الحاكمة (من الخطة): لا حالة وسطى دائمة. "قيد الاختبار" عابرة فقط
حتى ينتهي التشغيل الفعلي؛ بعدها الفرضية إمّا **مقبولة** أو **مرفوضة**،
موثَّقة في الحالتين.
"""
# -*- coding: utf-8 -*-
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


#: المسارات الأربعة من قسم "منهجية الاكتشاف" في خطة المشروع — قيمة `source`
#: موحّدة، لا نصاً حراً، ليبقى الحقل قابلاً للتصفية لاحقاً.
DISCOVERY_TRACKS = (
    "hypothesis_driven",   # أ) مبنيّ على الفرضية
    "data_driven",         # ب) مبنيّ على البيانات
    "literature_mining",   # ج) مبنيّ على الأدبيات
    "genetic_search",      # د) البحث الموجَّه
)

#: لا حالة وسطى دائمة (راجع خطة المشروع، قسم "لماذا مشروع منفصل").
REGISTRY_STATUSES = ("قيد الاختبار", "مقبولة", "مرفوضة")


def _default_registry_path(config: Optional[dict] = None) -> Path:
    """داخل مجلد المشروع على Drive إن كان مركَّباً (نفس drive_mount_point في
    دفتر التحضير)، وإلا محلياً بجانب الدفتر — يعمل بلا Drive للاختبارات."""
    config = config or {}
    base = config.get("drive_mount_point")
    d = (Path(base) / "MyDrive" / config.get("project_name", "crypto_model")
         / "experiment_registry") if base and Path(base).exists() else Path("experiment_registry")
    d.mkdir(parents=True, exist_ok=True)
    return d / "registry.json"


def _summarize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """يستخلص من مخرجات evaluate_windows/evaluate_hypothesis_over_rolling_windows
    كل ما يلزم لحكم مستقل لاحقاً — per_window كاملة، لا الملخّص وحده."""
    per_window = report.get("per_window")
    if per_window is not None and hasattr(per_window, "to_dict"):
        per_window = per_window.to_dict(orient="records")
    return {"mean_ic": report.get("mean_ic"), "std_ic": report.get("std_ic"),
            "frac_significant": report.get("frac_significant"),
            "consistent_sign": report.get("consistent_sign"),
            "n_ok": report.get("n_ok"), "per_window": per_window}


def _load_entries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_entries(path: Path, entries: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def register_hypothesis(hyp_id: str, hypothesis: str, source: str, status: str,
                        report: Optional[Dict[str, Any]] = None, notes: str = "",
                        registry_path: Optional[Path] = None,
                        config: Optional[dict] = None) -> Dict[str, Any]:
    """يُضيف سجلّاً جديداً، أو يستبدل سجلّاً بنفس ``hyp_id`` بالكامل إن وُجد
    (لا دمج جزئي — يمنع حقولاً قديمة متبقية من نتيجة سابقة).

    ``source``/``status`` تُرفَض بخطأ صريح إن خرجتا عن :data:`DISCOVERY_TRACKS`/
    :data:`REGISTRY_STATUSES` — قيمة حرّة هنا تكسر أي تصفية لاحقة على السجلّ.
    """
    if source not in DISCOVERY_TRACKS:
        raise ValueError(f"source يجب أن يكون أحد {DISCOVERY_TRACKS} (وصل {source!r}).")
    if status not in REGISTRY_STATUSES:
        raise ValueError(f"status يجب أن يكون أحد {REGISTRY_STATUSES} (وصل {status!r}).")

    path = registry_path or _default_registry_path(config)
    entries = _load_entries(path)
    entry = {"id": hyp_id, "hypothesis": hypothesis, "source": source, "status": status,
             "notes": notes, "axis_results": _summarize_report(report) if report is not None else None,
             "registered_at": datetime.now(timezone.utc).isoformat()}
    entries = [e for e in entries if e["id"] != hyp_id] + [entry]
    _save_entries(path, entries)
    print(f"\u2705 \u0633\u064f\u062c\u0651\u0644\u062a '{hyp_id}' \u0628\u062d\u0627\u0644\u0629 '{status}' \u0641\u064a {path}")
    return entry


def list_registry(registry_path: Optional[Path] = None, config: Optional[dict] = None):
    """جدول موجز — عمود واحد لكل حقل أساسي، بلا per_window (طويلة، عرضها
    هنا سيغرق البقية). راجعها عبر ``get_hypothesis(id)['axis_results']['per_window']``."""
    import pandas as pd
    path = registry_path or _default_registry_path(config)
    entries = _load_entries(path)
    cols = ["id", "hypothesis", "source", "status", "mean_ic", "frac_significant", "consistent_sign"]
    if not entries:
        print(f"\u0627\u0644\u0633\u062c\u0644\u0651 \u0641\u0627\u0631\u063a \u0628\u0639\u062f ({path}).")
        return pd.DataFrame(columns=cols)
    rows = [{"id": e["id"], "hypothesis": e["hypothesis"], "source": e["source"], "status": e["status"],
             "mean_ic": (e.get("axis_results") or {}).get("mean_ic"),
             "frac_significant": (e.get("axis_results") or {}).get("frac_significant"),
             "consistent_sign": (e.get("axis_results") or {}).get("consistent_sign")} for e in entries]
    return pd.DataFrame(rows)[cols]


def get_hypothesis(hyp_id: str, registry_path: Optional[Path] = None,
                   config: Optional[dict] = None) -> Dict[str, Any]:
    path = registry_path or _default_registry_path(config)
    for e in _load_entries(path):
        if e["id"] == hyp_id:
            return e
    raise KeyError(f"\u0644\u0627 \u0633\u062c\u0644\u0651 \u0628\u0645\u0639\u0631\u0651\u0641 '{hyp_id}' \u0641\u064a {path}.")
