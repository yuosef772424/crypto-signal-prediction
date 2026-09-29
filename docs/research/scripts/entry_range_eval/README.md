# entry_range_eval — تقييم نموذج entry_range المُدرَّب (`crypto_model_v1_entry_range_s100_am`)

التقرير الكامل والنتائج: [`docs/research/entry_range_eval.md`](../../entry_range_eval.md). ترتيب التشغيل:

```
python predict.py        --data preprocessing_output_1h_w32_s8_h1_latest.pkl --weights best.weights.h5 --out outputs.npz   # يحذف الـholdout قبل أي استدعاء للنموذج
python discover_val.py   --npz outputs.npz --out results/frozen.json        # VAL فقط: اختيار القواعد والعتبات وتجميدها
python eval_test.py      --npz outputs.npz --frozen results/frozen.json --split test --out results/test_results.json   # مرة واحدة
python incremental.py    --npz outputs.npz --frozen results/frozen.json --out results/incremental.json                # تحليل لاحق وصفي
```

`outputs.npz` (40 MB) والبيانات والأوزان لا تُودَع في المستودع؛ `results/` يحمل مخرجات التشغيل المرجعي.
`eval_test.py --split val` تشغيل تجريبي لنفس المسار على VAL (لصيد الأخطاء) وليس نتيجة.
