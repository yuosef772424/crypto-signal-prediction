# سكربتات إعادة إنتاج تدقيق التطبيع

سكربتات مستقلّة تُعيد إنتاج أرقام
[`normalization_audit_and_corrected_retest.md`](../normalization_audit_and_corrected_retest.md)
حرفياً (تحمّل خطّ الأنابيب ومحور التقييم واللاب من دفاتر المستودع نفسها).

| السكربت | ما يُنتجه |
|---|---|
| `corrected_retest.py absolute\|relative` | جداول §٣–§٤ (+ `corrected_retest_diag.pkl` لارتباطات الاستقلال) |
| `causal_decomposition.py` | جدول §٥ (بواقي رتب داخل كل يوم) |
| `holdout_momentum_orth_natr.py` | جدول §٦ (خارج العيّنة، مسجَّل مسبقاً) |

متغيّرات البيئة: `DATA_DIR` (مجلد فيه `csv/` و`csv_expanded/` بملفّات OHLCV
يومية لكل عملة، الافتراضي `/tmp/realdata`)، `OUT_DIR` (مكان حفظ النتائج،
الافتراضي المجلد الحالي)، `REPO_DIR` (جذر المستودع، يُستنتج تلقائياً).
`corrected_retest.py` يقرأ أيضاً أرشيفات الميزات الساعية
(`hourly_efficiency_ratio.pkl`، `hourly_vwap_deviation.pkl`،
`hourly_volume_concentration.pkl`) من `OUT_DIR`.
