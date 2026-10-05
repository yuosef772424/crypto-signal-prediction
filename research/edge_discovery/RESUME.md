# RESUME — نقطة الاستئناف (اقرأ هذا أولاً عند أي توقف/انقطاع)

> هذا الملف هو "ذاكرة" مشروع البحث. أي جلسة جديدة (بعد حدّ الاستخدام أو انقطاع الحاوية)
> تبدأ من هنا. حدّثه في نهاية كل مرحلة ثم commit + push.

## الحالة الحالية (آخر تحديث: 2026-10-01)

**المرحلة:** مكتمل — التقرير النهائي في `REPORT.md`.
- كل الفرضيات H01–H12 مُختبرة؛ HOLDOUT استُخدم مرة واحدة فقط لـ H06/H07 (`14_holdout.py`، مواصفات Addendum B).
- H12 رُفضت على كامل البيانات (413 إدراجاً) — HOLDOUT لم يُستهلك لها.
- الناجي الوحيد: H07 (TSMOM30 long-only BTC+ETH) كفلتر خفض مخاطر — **تأكّد على 2018–2023 غير المستخدمة (Addendum C، سكربتات 16/17)، ثقة متوسطة**.
- H13 (إشارة PR #7 كمحفظة) مرفوضة بعد التكاليف (سكربت 18). بيانات CoinMetrics: curl من raw.githubusercontent.com/coinmetrics/data/master/csv/<asset>.csv إلى /home/user/research/coinmetrics/.
- **HOLDOUT 2026 أصبح "مستهلكاً"**: أي فرضية جديدة تحتاج بيانات جديدة (بعد 2026-09) أو تاريخاً أقدم (2020–2023).

## الخطوة التالية (إن استُؤنف العمل)
1. أفضل خطوة: السماح بـ `data.binance.vision` في إعدادات شبكة البيئة، ثم تنزيل 2020–2023 واختبار H07
   (ومجموعة الاتجاه كاملة) على فترة لم تُلمس — هذا هو الـ out-of-sample الحقيقي المتبقي.
2. تنزيل الملفات من Drive: `data_tools/folder_index.py` يبني فهرس `title → fileId` من نتائج
   `search_files` بـ `parentId = '<folder>'` (صفحات 100)، ويكتب `missing_ids.txt` بالمعرّفات الناقصة فقط؛
   ثم `download_file_content` لكل معرّف (تُحفظ تلقائياً على القرص دون قراءتها) و `sweep.py` لفك الترميز.

## إعادة بناء البيانات (إذا ضاعت الحاوية)
- المصدر: Google Drive، مجلد `futures_metrics` (id: `1LmmrTpBo-62nao_0FdqkZrohlMgfGTQT`)،
  ملف لكل عملة `<SYMBOL>.csv.gz` (~0.9MB، ساعي 2024-01 → 2026-09).
- الطريقة الوحيدة المتاحة في هذه البيئة: أداة Drive `download_file_content` لكل ملف (تحفظ النتيجة
  في tool-results)، ثم `python3 data_tools/sweep.py` لفك الترميز إلى `/home/user/research/metrics/`،
  ثم `python3 data_tools/build_panel.py` لبناء `panel_*.parquet`.
  - ابحث عن المعرّفات: `search_files` بـ `parentId = '<folder>' and (title = 'X.csv.gz' or ...)` (≤48 اسماً لكل استعلام).
  - قائمة العملات الأساسية (195): `data_tools/metrics_downloaded.txt`.
- **قيود معروفة:** موصل Drive يفشل مع الملفات > ~2MB (يقطع الجلسة)، لذلك بيانات OHLCV 15m
  (2020→2026، 6–11MB/ملف) **غير قابلة للتنزيل** حالياً. شبكة البيئة تحجب Binance
  (`data.binance.vision`, `fapi.binance.com`, `api.binance.com`).
  **الحل الأفضل:** السماح بهذه النطاقات في إعدادات الشبكة للبيئة، ثم استخدام
  `tools/fetch_history_csv_concurrent.py` مباشرة — يفتح اختبار 6+ سنوات بدل 2.75.

## ملفات المشروع البحثي
| ملف | الدور |
|---|---|
| `00_PREREGISTRATION.md` | التقسيم الزمني ومعايير القبول (ثابتة، مسجّلة قبل الاختبار) |
| `hypothesis_log.csv` | سجل كل فرضية: البيانات، الطريقة، النتيجة، الحالة، الخطوة التالية |
| `lib.py`, `features.py`, `events.py`, `tsbt.py` | أدوات مشتركة |
| `01..11_*.py` | الشاشات المنفّذة (كل واحدة قابلة لإعادة التشغيل) |
| `*.csv` | مخرجات الشاشات |
| `data_tools/` | فك ترميز تنزيلات Drive وبناء اللوحة |


## تشغيل نموذج الصور (H19) — اكتمل (2026-10-05)
> **اكتمل:** الإعدادات 1h C1–C3 شُغّلت في جلسة لاحقة (بيانات 5m من `tools/fetch_crypto_dataset.py`، مقصوصة عند 2026-09-30)؛
> كل الإعدادات الثمانية مرفوضة — انظر REPORT.md وسجلّ الفشل F-0063. التعليمات أدناه للتاريخ فقط.

### (تاريخي) توقّف هنا عمداً بطلب المالك
- اكتمل: 4h (C0..C3) و 1h C0 — النتائج في `image_cnn_results.csv` (السكربت يتخطّى أي تجربة موجودة فيه ويكمل الباقي).
- المتبقي: 1h C1, C2, C3.
- الخطوات على Colab (يفضّل GPU):
  1. `pip install numba torch scikit-learn pandas pyarrow`
  2. ضع ملفات `ohlc_<SYMBOL>_5m.parquet` للعملات العشر في مجلد (مثلاً Drive) — تُبنى من github Speirsy11/crypto-dataset
     (5m parquet عبر media.githubusercontent.com، ثم الدمج كما في هذه الجلسة).
  3. `DATA_DIR=/content/drive/MyDrive/ohlc OUT_CSV=image_cnn_results.csv python3 25_image_cnn.py`
- لاستخدام GPU: أضف `.to('cuda')` للنموذج والدفعات — غير مطلوب للنتائج، فقط للسرعة.
