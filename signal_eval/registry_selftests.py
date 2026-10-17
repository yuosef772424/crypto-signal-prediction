"""
PURPOSE:  run_registry_selftests: registry self-tests on a temp path, no Drive. Defines only; the runner notebook calls it.
TAGS:     run_registry_selftests, registry selftests, tempfile
PITFALLS: The old notebook called run_registry_selftests() at load; that call now lives in the runner notebook. Executed into the one shared signal_eval namespace by signal_eval/_loader.py (never imported on its own): names from other modules resolve at call time.

### اختبارات ذاتية — سجلّ التجارب
"""
# @title
def run_registry_selftests() -> None:
    """اختبارات سجلّ التجارب — تعمل بلا Drive، على مسار مؤقّت."""
    import tempfile, shutil

    tmp = Path(tempfile.mkdtemp())
    reg = tmp / "registry.json"
    tests = 0
    failed = []

    def check(name, cond):
        nonlocal tests
        tests += 1
        if not cond:
            failed.append(name)

    try:
        # 1) تسجيل بسيط ثم قراءته
        register_hypothesis("H_test", "فرضية تجريبية", "hypothesis_driven",
                            "قيد الاختبار", registry_path=reg)
        e = get_hypothesis("H_test", registry_path=reg)
        check("تسجيل بسيط", e["hypothesis"] == "فرضية تجريبية" and e["status"] == "قيد الاختبار")

        # 2) استبدال كامل عند نفس المعرّف (لا دمج جزئي)
        register_hypothesis("H_test", "فرضية تجريبية", "hypothesis_driven",
                            "مقبولة", notes="بعد التحديث", registry_path=reg)
        e2 = get_hypothesis("H_test", registry_path=reg)
        check("استبدال كامل عند تكرار المعرّف",
              e2["status"] == "مقبولة" and e2["notes"] == "بعد التحديث")
        check("لا تكرار للسجلّ", len(_load_entries(reg)) == 1)

        # 3) source/status غير صالحين يُرفضان بخطأ صريح
        try:
            register_hypothesis("H_bad", "x", "not_a_track", "مقبولة", registry_path=reg)
            check("رفض source غير صالح", False)
        except ValueError:
            check("رفض source غير صالح", True)

        try:
            register_hypothesis("H_bad", "x", "hypothesis_driven", "not_a_status", registry_path=reg)
            check("رفض status غير صالح", False)
        except ValueError:
            check("رفض status غير صالح", True)

        # 4) report كامل (تقليد شكل evaluate_windows) يُختزَل ويُحفَظ بصورة قابلة لـJSON
        import pandas as pd
        fake_report = {"mean_ic": -0.05, "std_ic": 0.02, "frac_significant": 0.6,
                       "consistent_sign": False, "n_ok": 3,
                       "per_window": pd.DataFrame([{"window": "نافذة 1", "ic": -0.05}])}
        register_hypothesis("H_report", "فرضية بنتائج", "literature_mining",
                            "مقبولة", report=fake_report, registry_path=reg)
        e3 = get_hypothesis("H_report", registry_path=reg)
        check("per_window يُحفَظ كقائمة قواميس لا DataFrame",
              isinstance(e3["axis_results"]["per_window"], list))
        check("mean_ic محفوظ بصحّة", e3["axis_results"]["mean_ic"] == -0.05)

        # 5) list_registry يرجع جدولاً بعدد الصفوف الصحيح
        table = list_registry(registry_path=reg)
        check("list_registry بعدد الصفوف الصحيح", len(table) == 2)

        # 6) معرّف غير موجود يرفع KeyError، لا يرجع None بصمت
        try:
            get_hypothesis("لا_يوجد", registry_path=reg)
            check("get_hypothesis يرفع KeyError لمعرّف غائب", False)
        except KeyError:
            check("get_hypothesis يرفع KeyError لمعرّف غائب", True)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if failed:
        print(f"\u274c {len(failed)}/{tests} \u0641\u0634\u0644\u062a: {failed}")
    else:
        print(f"\u2705 \u0643\u0644 \u0627\u0644\u0627\u062e\u062a\u0628\u0627\u0631\u0627\u062a \u0646\u062c\u062d\u062a ({tests}/{tests}) \u2014 \u0633\u062c\u0644\u0651 \u0627\u0644\u062a\u062c\u0627\u0631\u0628.")
