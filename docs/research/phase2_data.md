# المرحلة ٢ — بيانات Binance الكاملة (15m + تمويل + OI + metrics) في خط الأنابيب

خط الأنابيب `crypto_data_pipeline_v6.ipynb` صار يقرأ الأرشيف الذي نزّله `tools/fetch_history_vision_colab.py` على Drive، ويحوّله إلى ميزات يومية **بلا تسرّب**، تدخل نفس `preprocessing_output*.pkl.gz` الذي يقرؤه `main.ipynb` و`cross_asset/data.py`. الأهداف (`high/low/close`، ووضع `relative`) لم تتغيّر، والتطبيع هو نفسه (تطبيع دلالي داخل كل نافذة، ولا يستخدم إحصاءات من خارج النافذة). الحساب كله في وحدة مستقلة: `tools/intraday_features.py`، ولها اختبارات في `tests/test_intraday_features.py`.

## صيغ الملفات (تحقّقت منها على عيّنات حقيقية من Drive ومن الأرشيف الخام)

| المجلد (تحت `MyDrive`) | الأعمدة | الطابع الزمني |
|---|---|---|
| `history_15m/<S>.csv.gz` | `timestamp, datetime_utc, open, high, low, close, volume, quote_volume, trades, taker_buy_volume, taker_buy_quote_volume` | `timestamp` بالمللي ثانية = وقت **فتح** الشمعة بتوقيت UTC. تحقّقت من أن الأرشيف الخام لعقود USDT ما زال بالمللي ثانية حتى 2026-09. |
| `funding_rate/<S>.csv.gz` | `timestamp, funding_rate` | نص `YYYY-mm-dd HH:MM:SS+00:00` = وقت التسوية (كل 8 ساعات، وكل 4 ساعات أو ساعة لبعض العملات). |
| `open_interest/<S>.csv.gz` | `timestamp, open_interest` | نص ISO = **بداية** الساعة. |
| `futures_metrics/<S>.csv.gz` | `timestamp, sum_open_interest, sum_open_interest_value, count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio, count_long_short_ratio, sum_taker_long_short_vol_ratio` | نص ISO = بداية الساعة، ويبدأ من 2024-01-01. |
| `crypto_data/` | `asset_registry.csv`، `gaps_report.csv`، `funding_oi_report.csv` | — |

ملاحظات ظهرت من العيّنات، وعالجها المُحمِّل:
- **صفّ OI أو metrics المؤرَّخ بالساعة h يحمل آخر لقطة 5 دقائق داخل الساعة [h, h+1h)** (مثلاً 00:55)، فهو لا يُعرف إلا عند h+1h. لذلك وقت التوفّر = الطابع + الفترة، والفترة تُستنتج من الملف (1h افتراضياً، أو 15m إن نُزّل بـ `OI_PERIOD="15m"`).
- **العملات المشطوبة** (مثل `BTCSTUSDT`) فيها صفوف بعد الشطب قيمتها `sum_open_interest = 0E-16` والنسب فارغة. القيم ≤ 0 تُعامَل كمفقودة.
- **الثغرات** (`gaps_report.csv`): أيام كاملة مفقودة لعشرات العملات (مثل 2022-02-26 إلى 02-28 و2022-04-01 إلى 04-02) وأجزاء من أيام. كل يوم يغطّي أقل من 75% من شموع 15m يُعطى `NaN` ويُعلَّم `ITD_available = 0`، فلا يُحسب من بيانات ناقصة.
- الطوابع الرقمية تُكشف وحدتها لكل قيمة على حدة (ثوانٍ، مللي، مايكرو، نانو). الحساب بأعداد صحيحة لا بـ float: وقت التمويل الخام `1785542400001` (أي +1ms) كان يفسد مع float64. التكرارات تُحذف ويُبقى آخرها، ثم تُرتَّب الصفوف.
- ملفات الشموع القديمة ذات الأعمدة السبعة تُقبل: أعمدة taker وtrades تصير `NaN`، ويُحسب VWAP بالسعر النموذجي بدلاً منها.

## مبدأ عدم التسرّب

فهرس الدفتر هو وقت **فتح** الشمعة. الشمعة اليومية D تُغلق عند D+1 00:00، وكل قيمة تُنسب إلى الشمعة فقط إن كان وقت توفّرها ≤ إغلاقها:
- ميزات اليوم D من شموع 15m التي تفتح داخل [D, D+1)، وتُعتبر متاحة عند D+1 00:00. على فريمات أصغر من اليوم تأخذ كل شمعة **آخر يوم مكتمل** قبل إغلاقها.
- التمويل: التسوية ذات الطابع τ متاحة بعد τ مباشرة. تسوية D+1 00:00 تُنسب إذن إلى D+1 لا إلى D، وهذا تحفّظ مقصود.
- OI وmetrics: متاحة عند الطابع + الفترة.
- القيمة التي يزيد عمرها عند الإغلاق على `max_age` (يوم واحد افتراضياً) تُعامَل كمفقودة ولا تُمدَّد. هذا يغطّي ما بعد الشطب وما قبل بداية الأرشيف.
- الاختبار `test_no_lookahead` يغيّر كل البيانات بعد إغلاق D (الأسعار والأحجام والتمويل والـ metrics)، ويحذف بعض الشموع، ثم يتحقّق أن ميزات D وما قبله لم تتغيّر.

## الميزات

القيم المفقودة تصير 0 مع علم توفّر (0/1)، بنفس أسلوب `FUND_available` القائم، لأن النوافذ لا تقبل `NaN`. الوحدة نفسها تُرجع `NaN` مع القناع.

| العمود | التعريف | التطبيع (`FEATURE_KINDS`) |
|---|---|---|
| `EFF_RATIO_24H` *(فتحة قائمة)* | \|log إغلاق اليوم − log فتحه\| ÷ مجموع \|Δlog\| للإغلاقات الساعية داخل اليوم | unit_0_1 |
| `VWAP_DEVIATION` *(فتحة قائمة)* | (إغلاق آخر شمعة − VWAP) ÷ VWAP، حيث VWAP = Σquote_volume ÷ Σvolume | sign_robust |
| `VOL_CONC_HHI` *(فتحة قائمة)* | مؤشر هيرفندال لحصص الحجم على 24 ساعة | unit_0_1 |
| `ITD_TAKER_BUY_RATIO` | Σtaker_buy_volume ÷ Σvolume | unit_0_1 |
| `ITD_TRADES_LOG` | log1p(عدد صفقات اليوم) | log_centered (ناقص وسيط النافذة، بلا قسمة: قيمة يومية داخل نافذة ساعات دالّة درجية، وتقييسها بـIQR النافذة كان يمحو حجم التغيّر — [r2_04](audit/r2_04_step_feature_saturation.py)) |
| `ITD_TRADES_Z` | درجة معيارية لـ `ITD_TRADES_LOG` مقابل الثلاثين يوماً **السابقة** للعملة نفسها | zscore |
| `ITD_RVOL` | 100·√Σr² لعوائد 15m اللوغاريتمية داخل اليوم (تقلّب محقَّق، بالنسبة المئوية) | percent |
| `ITD_VOL_TOPK` | حصة أعلى 4 شموع 15m من حجم اليوم | unit_0_1 |
| `ITD_available` | 1 إن غطّت شموع اليوم 75% أو أكثر | binary_flag |
| `EFF_RATIO_available`, `VWAP_DEVIATION_available`, `VOL_CONC_HHI_available` | 1 حيث للفتحة قيمة من 15m | binary_flag (كل `*_available`؛ كانت unit_0_1 فتُصفَّر حين تثبت — [r2_03](audit/r2_03_dead_availability_flags.py)) |
| `FUND_rate`, `FUND_rate_z` | آخر تسوية قبل الإغلاق، ودرجتها المعيارية على آخر 90 تسوية | funding_rate / zscore |
| `FUND_sum_1d`, `FUND_sum_3d` | مجموع التسويات في آخر يوم وفي آخر 3 أيام | funding_rate |
| `OI_chg_1` | log OI(إغلاق t) − log OI(إغلاق t−1)، أي تغيّر يومي على الفريم اليومي | sign_robust |
| `LSR_TOP_ACCT`, `LSR_TOP_POS`, `LSR_GLOBAL` | log نسبة long/short: كبار المتداولين بالحسابات، ثم كبارهم بالمراكز، ثم كل الحسابات | unit_sym |
| `LSR_GLOBAL_chg_1` | التغيّر اليومي لـ `LSR_GLOBAL` | sign_robust |
| `TAKER_LSR_1d` | متوسط log(taker buy/sell vol ratio) على لقطات آخر يوم | unit_sym |
| `MET_available` | 1 إن توفّرت metrics عند الإغلاق. تكون 0 قبل 2024-01-01 وبعد الشطب | binary_flag |

عند تشغيل المفتاحين معاً يصير عدد الميزات 43، ومن دونهما يبقى 23 ميزة (القائمة الحالية بلا أي تغيير). تنبيه: الفتحات الثلاث القائمة (`EFF_RATIO_24H`، `VWAP_DEVIATION`، `VOL_CONC_HHI`) رُفضت سابقاً بحثياً على الهدف المطلق (انظر [kaufman_efficiency_ratio.md](kaufman_efficiency_ratio.md) و[vwap_deviation_and_volume_concentration.md](vwap_deviation_and_volume_concentration.md)). أُعيد وصلها هنا لأنها صارت تُحسب من بيانات حقيقية كاملة لكل العملات بدلاً من أرشيف 50 عملة. بقاؤها في المدخلات قرار يُتخذ بعد اختبارها على اللوحة. لاستبعاد أي عمود: `exclude_features([...])`.

## المفاتيح

في `CONFIG['phase2_data']`، داخل خلية `DEFAULT_CONFIG` (القسم 4):

| المفتاح | الافتراضي | المعنى |
|---|---|---|
| `use_intraday_15m` (USE_INTRADAY_15M) | `"auto"` | `True` إن وُجد `history_15m/` غير فارغ تحت الجذر ووُجدت `tools/intraday_features.py` |
| `use_futures_metrics` (USE_FUTURES_METRICS) | `"auto"` | `True` إن وُجد أيٌّ من `funding_rate/` أو `open_interest/` أو `futures_metrics/` |
| `data_root` | `None` | `None` يعني `/content/drive/MyDrive`، ويمكن وضع أي مسار يحوي المجلدات |
| `min_coverage`, `trades_z_window`, `topk_bars`, `max_age` | 0.75, 30, 4, `"1D"` | عتبة تغطية اليوم، ونافذة `ITD_TRADES_Z`، وk لـ `ITD_VOL_TOPK`، وأقصى عمر مقبول للقيمة |
| `module_dirs` | `/content/crypto-signal-prediction`, `/content/drive/MyDrive/crypto` | أين يُبحث عن `tools/intraday_features.py`، بعد جذر المستودع الحاوي لحزمة `data/` (أولاً، مهما كان مجلد العمل) ثم مجلد العمل الحالي |

القيمة `"auto"` تُثبَّت إلى True أو False في بداية `build_dataset*` عبر `resolve_phase2_toggles`، فتتطابق `feature_order` مع ما تضيفه الخطافات فعلاً. من دون المجلدات أو الوحدة، يبقى المسار اليومي كما كان حرفياً: نفس الميزات ونفس القيم، والاختبار الذاتي `t_phase2_toggles_off_leave_pipeline_unchanged` يتحقّق من ذلك. بصمة نقاط الاستئناف تتغيّر فقط عند تفعيل أحد المفتاحين.

## التشغيل على Colab

1. شغّل `main.ipynb` من خليته الأولى كالمعتاد. الخلية 2 تركّب Drive وتسحب المستودع، والخلية 3 تنفّذ `%run "crypto_data_pipeline_v6.ipynb"`. بذلك يصير مجلد العمل هو المستودع (والمُشغِّل يضعه على `sys.path` ليستورد الحزمة `data/`)، فتُوجد `tools/intraday_features.py` تلقائياً (تُبحث أولاً تحت جذر المستودع الحاوي لـ`data/`). يجب أن ترى في خرج الاختبارات الذاتية: `✅ t_phase2_hooks_from_drive_files`.
   إن فتحت دفتر الأنابيب وحده، فاستنسخ المستودع (يحتاج مجلد `data/` أيضاً بجانب الدفتر) إلى `/content/crypto-signal-prediction` (أو ضعه في `MyDrive/crypto`)، أو أضف مساره إلى `phase2_data['module_dirs']`.
2. لا يلزم أي إعداد آخر: المجلدات موجودة في `MyDrive`، فيتحوّل `"auto"` إلى True. للتحكّم الصريح نفّذ قبل البناء:
   ```python
   update_config(phase2_data={'use_intraday_15m': True, 'use_futures_metrics': True})   # أو False
   ```
3. ابنِ البيانات كما في القسم 20 (الخلية 56): `mount_drive()`، ثم `registry = load_asset_registry()` و`configs = filter_desired_coins(...)`، ثم `dataset = build_dataset(configs, load_asset_fn=load_asset, resample_fn=make_resample_fn(CONFIG), config=CONFIG, checkpoint_dir=...)`، ثم `save_data_to_drive(dataset)`. عند البداية سيُطبع سطر `🧩 المرحلة ٢: USE_INTRADAY_15M=True | USE_FUTURES_METRICS=True`.
4. **الزمن والذاكرة:** قراءة ملف 15m لست سنوات (نحو 13MB مضغوطاً، 210 آلاف شمعة) وحساب ميزاته يستغرقان نحو ثانيتين محلياً، وذروة الذاكرة نحو 100MB لكل خيط، ثم تُخزَّن النتيجة مؤقتاً لآخر 32 عملة. على Colab، ومع القراءة من Drive، التقدير نحو 3 إلى 6 ثوانٍ لكل عملة، أي زيادة تقارب 10 إلى 30 دقيقة لنحو 600 عملة مع الخيوط الافتراضية. ملفات التمويل والـ metrics صغيرة (نحو 1MB). الرام الإضافية لكل خيط نحو 150MB، و`default_workers` يحدّ عدد الخيوط بالرام أصلاً.
5. بعد الحفظ، يقرأ `main.ipynb` (الخلية 7) الملف `preprocessing_output_latest.pkl.gz` الجديد كما هو. عدد الميزات تغيّر، فيلزم تدريب جديد، ولا تُستخدم نماذج قديمة.
