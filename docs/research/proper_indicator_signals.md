# الاستخدام "الصحيح" للمؤشرات المعروفة (لا القيمة الخام) — مرفوضة (36 من 36)

**الحالة: مرفوضة (36 من 36)،** لكن بنتائج فرعية لافتة تستحق التوثيق. رابعة فئات الأبحاث بعد استنفاد `pandas_ta` (بعد [مقدّرات التقلّب](new_volatility_liquidity_estimators.md)، [هيرست/Variance Ratio](hurst_variance_ratio_features.md)، [نسبة القفزات](jump_ratio_feature.md)) — لكن بمنهجية مختلفة جوهرياً: بدل ميزات **جديدة كلياً**، هذا بحث في **كيفية الاستخدام الصحيح لمؤشرات معروفة موجودة أصلاً** (`RSI`, `MACD`, `ADX`/`DMI`, `Bollinger Bands`, `SuperTrend`) — بناءً على طلب صريح لفهم كيف تُستخرَج الفائدة من كل مؤشر ولماذا يُستخدم، لا مجرّد اختبار قيمته الخام إحصائياً.

## الخلفية والدافع

الدفعات السابقة (١-٥ من `SURVEY_CANDIDATES_ROBUST`) استبعدت `RSI`/`MACD`/`ADX`/`DM±`/`BBANDS` من الاختبار الصريح لأنها **موجودة أصلاً في `feature_order`** (يراها النموذج المُدرَّب مباشرة). لكن هذا يعني أن **قيمتها الخام** لم تُختبَر أبداً عبر محور IC الصارم — والأهمّ: **لا واحد منها اختصاراً استُخدم بصيغته "الصحيحة"** كما توصي بها مصادره الأصلية (كتاب وايلدر، موقع بولنجر، StockCharts) بدل قيمته الرقمية الخام مباشرة. `SuperTrend` غائب كلياً عن مجموعة الميزات (اختُبِر مرّة واحدة فقط بصيغته الخام في [الدفعة الخامسة](pandas_ta_batch4_5_final.md)، مرفوض).

## المنهجية: بحث ويب لكل مؤشر، ثمّ صيغتان (خام مقابل صحيحة)

لكل مؤشر: بحث في مصادره الأصلية/الموثوقة عن **كيف يُستخدم فعلياً في التداول ولماذا** — ثمّ اختبار الصيغة الخام (كما لو أُخذت القيمة الرقمية مباشرة) **مقابل** الصيغة الموصى بها، على نفس مقياس H003 الكامل (50 أصلاً، 30 نافذة)، كل مؤشر بصفّه الخاص في الجدول (لا دفعة عمياء واحدة).

### ١) RSI — التباعد (Divergence) لا عتبتَي 70/30

وايلدر نفسه (كتابه الأصلي *New Concepts in Technical Trading Systems*) يصف **التباعد بين RSI والسعر** بأنه **الميزة الأقوى** في RSI — أقوى من عتبتَي ذروة الشراء (>70) والبيع (<30) التقليديتين. تباعد هابط = السعر يصنع قمّة أعلى بينما RSI يصنع قمّة أدنى (زخم يضعف رغم ارتفاع السعر، إنذار انعكاس). [Wikipedia](https://en.wikipedia.org/wiki/Relative_strength_index)، [Plisio](https://plisio.net/education/rsi-divergence-bullish-bearish)، [Whale Story](https://goraestory.com/en/academy/rsi-divergence/).

**التنفيذ**: `RSI_DIVERGENCE_{L}` — تقريب مبسَّط (لا مطابقة قمم/قيعان مؤكَّدة حرفياً): فرق الدرجة المعيارية بين تغيّر السعر وتغيّر RSI على نفس الأفق `L`. أُضيف في `crypto_data_pipeline_v6.ipynb` (`custom_settings["rsi_divergence_windows"]`)، اختبار ذاتي يتحقّق من الإشارة الصحيحة (موجب لتباعد هابط صناعي، سالب لتباعد صاعد) على سيناريوهات صناعية صريحة تطابق تعريف وايلدر حرفياً.

### ٢) MACD — إشارة تقاطع الهستوغرام لا قيمته الخام

تقاطع خط MACD مع خط الإشارة هو الإشارة التقليدية؛ الهستوغرام **نظام إنذار مبكّر** لهذا التقاطع (يتغيّر اتجاهه قبل التقاطع الفعلي بفترة). [StockCharts](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-histogram)، [Macroption](https://www.macroption.com/macd-histogram-crossover/). **ملاحظة مهمّة من المصادر**: قوّة الإشارة تعتمد على موقعها من خط الصفر — تقاطعات بعيدة عن الصفر أقوى، القريبة منه في سوق عرضي غالباً "ضوضاء".

**التنفيذ**: `MACD_HIST_SIGN` = إشارة (`sign`) آخر قيمة لـ`MACDh_12_26_9` (الموجودة أصلاً في `feature_order`) — لا حاجة لميزة جديدة، فقط تحويل في نصّ الاختبار.

### ٣) ADX/DMI — قوّة الاتجاه × اتجاه DI+/DI−، لا ADX بمفرده

النظام الأصلي لوايلدر (ADX جزء من نظام Directional Movement الكامل): **ADX يقيس قوّة الاتجاه بلا اتجاه** (رقم غير موجَّه)؛ **الاتجاه نفسه** يأتي من تقاطع `+DI`/`−DI`؛ الإشارة الحقيقية تُؤكَّد فقط حين يكون ADX مرتفعاً (عادة >25) مصاحباً لتقاطع DI. [ChartSchool](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx)، [TrendSpider](https://help.trendspider.com/kb/indicators/average-directional-index-adx)، [FinanceStrategists](https://www.financestrategists.com/wealth-management/fundamental-vs-technical-analysis/wilders-dmi-adx/).

**التنفيذ**: `ADX_DI_SIGNAL` = `sign(DMP_14 − DMN_14) × (ADX_14/100)` — اتجاه DI مرجَّح بقوّة الاتجاه (بدل عتبة ثنائية 25 عشوائية)، محسوبة مباشرة من الأعمدة الموجودة أصلاً (`ADX_14`, `DMP_14`, `DMN_14`).

### ٤) Bollinger Bands — %B (الموضع داخل النطاق)، لا السعر الخام

`%B` (`BBP` في `pandas_ta`) يقيس موضع الإغلاق **نسبياً** داخل النطاقين: `>1` فوق النطاق العلوي (اختراق/تشبّع شرائي)، `<0` تحت السفلي. هذه **هي** الصيغة الصحيحة أصلاً — [TrendSpider](https://trendspider.com/learning-center/bollinger-band-width-and-b-an-overview/)، [IG](https://www.ig.com/en/trading-strategies/what-are-bollinger-bands-and-how-do-you-use-them-in-trading--200129). ملاحظة إضافية من بولينجر نفسه: عرض النطاق (`BandWidth`/`BBB`) يُستخدم للتنبؤ **بتوسّع التقلّب المستقبلي** (Squeeze)، لا باتجاه العائد — سؤال مختلف عن IC الاتجاهي المُختبَر هنا، فلم يُختبَر ضمن هذه الدفعة (أولوية منخفضة، يحتاج هدف "تقلّب مستقبلي" لا "عائد مستقبلي").

**التنفيذ**: `BBP_20_2.0` — عمود موجود أصلاً في `feature_order` (`bbands: [20]` في `indicator_settings` الافتراضية)، لم يُختبَر عبر IC صراحة من قبل.

### ٥) SuperTrend — اتجاه الخط (فوق/تحت السعر)، لا قيمته السعرية

مؤشّر اتّجاهي استمراري شائع جداً عملياً (وقف متحرّك مبني على ATR)، غائب كلياً عن مجموعة الميزات الافتراضية لهذا المشروع (خلافاً لـRSI/MACD/ADX/BBANDS). الإشارة الصحيحة هي **الاتجاه**: أخضر (فوق السعر أدناه) = صاعد، أحمر (تحته) = هابط؛ الانقلاب يحدث حين يعبر السعر الخطّ. [LuxAlgo](https://www.luxalgo.com/blog/how-to-use-the-supertrend-indicator-effectively/)، [Mudrex](https://mudrex.com/learn/supertrend-indicator/)، [forex.com](https://www.forex.com/en/news-and-analysis/how-to-use-the-supertrend-indicator/). **ملاحظة من المصادر**: مؤشّر متأخّر (lagging) بطبيعته، يُنتج إشارات كاذبة في الأسواق العرضية — يُوصى باستخدامه مع مؤشّرات أخرى للتأكيد (ADX، RSI).

**التنفيذ**: `SUPERT_DIR_{w}` (اتجاه خام من `pandas_ta`، ±1) و`SUPERT_STRETCH_{w}` (مسافة الإغلاق عن الخطّ، نسبة مئوية — "مدى امتداد الاتجاه"). طول 10 و20 (الإعداد الكلاسيكي 10/مضاعف 3.0 المذكور في المصادر). أُضيفا في `crypto_data_pipeline_v6.ipynb` (`custom_settings["supertrend_windows"]`)، اختبار ذاتي يطابق `SUPERTd` الخام من `pandas_ta` تماماً.

بما أن SuperTrend استمراري (trend-following) بطبيعته، فرضيته **معاكسة تماماً** لفرضية H003 الانعكاسية: اتجاه صاعد ← يُتوقَّع استمرار موجب، لا ارتداد.

## النتيجة الفعلية عبر المحور الصارم (50 أصلاً، 30 نافذة — نفس مقياس H003)

| المؤشّر | الصيغة | الهدف | mean_ic | frac_significant | consistent_sign |
|---|---|---|---|---|---|
| RSI | خام (`RSI_14`) | close | -0.125 | **77%** | ❌ False |
| RSI | خام (`RSI_14`) | high | -0.024 | 40% | ❌ False |
| RSI | خام (`RSI_14`) | low | -0.092 | 70% | ❌ False |
| RSI | تباعد (`RSI_DIVERGENCE_14`) | close | -0.075 | 53% | ❌ False |
| RSI | تباعد (`RSI_DIVERGENCE_14`) | high | -0.038 | 40% | ❌ False |
| RSI | تباعد (`RSI_DIVERGENCE_14`) | low | -0.075 | 57% | ❌ False |
| RSI | تباعد (`RSI_DIVERGENCE_28`) | close | -0.022 | 27% | ❌ False |
| RSI | تباعد (`RSI_DIVERGENCE_28`) | high | +0.001 | 23% | ❌ False |
| RSI | تباعد (`RSI_DIVERGENCE_28`) | low | -0.049 | 37% | ❌ False |
| MACD | خام (`MACDh`) | close | -0.053 | 43% | ❌ False |
| MACD | خام (`MACDh`) | high | +0.006 | 37% | ❌ False |
| MACD | خام (`MACDh`) | low | -0.029 | 37% | ❌ False |
| MACD | إشارة الهستوغرام | close | -0.037 | 37% | ❌ False |
| MACD | إشارة الهستوغرام | high | +0.007 | 30% | ❌ False |
| MACD | إشارة الهستوغرام | low | -0.013 | 37% | ❌ False |
| ADX | خام (`ADX_14`) | close | +0.016 | 40% | ❌ False |
| ADX | خام (`ADX_14`) | high | +0.038 | 40% | ❌ False |
| ADX | خام (`ADX_14`) | low | -0.022 | 50% | ❌ False |
| ADX | اتجاه DI×قوّة | close | -0.010 | 27% | ❌ False |
| ADX | اتجاه DI×قوّة | high | -0.044 | 33% | ❌ False |
| ADX | اتجاه DI×قوّة | low | -0.009 | 37% | ❌ False |
| Bollinger | %B (`BBP_20`) | close | -0.090 | **67%** | ❌ False |
| Bollinger | %B (`BBP_20`) | high | -0.001 | 33% | ❌ False |
| Bollinger | %B (`BBP_20`) | low | -0.054 | 50% | ❌ False |
| SuperTrend | اتجاه (10) | close/high/low | -0.000/+0.020/-0.019 | 7%/33%/27% | ❌ False |
| SuperTrend | اتجاه (20) | close/high/low | +0.005/+0.021/-0.008 | 17%/30%/23% | ❌ False |
| SuperTrend | امتداد (10) | close/high/low | -0.029/+0.057/-0.080 | 20%/37%/**67%** | ❌ False |
| SuperTrend | امتداد (20) | close/high/low | -0.028/+0.046/-0.072 | 17%/30%/**63%** | ❌ False |

**36 من 36 مرفوضة.** لكن ثلاث ملاحظات فرعية لافتة تستحق تسجيلاً صريحاً:

1. **`RSI_14` الخام أقوى بكثير من أي مرشّح آخر في هذه الدفعة** (`frac_significant=77%` على close، ثاني أعلى رقم شُوهد في هذا المشروع بعد `NATR_14` نفسه 93-97%) — لكن `consistent_sign=False` يمنع القبول رغم القوّة الظاهرية: الإشارة قوية إحصائياً في أغلب النوافذ لكن تتقلّب اتجاهها بين نافذة وأخرى، خلافاً لـ`NATR_14` الذي يحافظ على نفس الاتجاه في كل نافذة تقريباً.
2. **مفاجأة منهجية**: صيغة "التباعد الصحيحة" (`RSI_DIVERGENCE`) كانت **أضعف** من `RSI` الخام (`frac_significant` 53% مقابل 77% على close) — عكس ما تتوقّعه توصية وايلدر بأن التباعد "أقوى ميزة". تفسيران محتملان غير متنافيين: (أ) التقريب المبسَّط المُستخدَم هنا (فرق درجة معيارية) لا يلتقط جوهر التباعد الحقيقي الذي يحتاج مطابقة قمم/قيعان مؤكَّدة حرفياً؛ (ب) ذروة الشراء/البيع الخام تحدث باستمرار فتُعطي عيّنة أكبر وأثبت إحصائياً عبر 30 نافذة، بينما التباعد نادر الحدوث فيصعب قياسه بثبات على هذا المقياس الزمني.
3. **`SUPERT_STRETCH` على `low`** (كلا النافذتين) و**`BB_PCTB_20` على `close`** يقتربان من نفس نمط `RSI_14`: `frac_significant` مرتفع نسبياً (63-67%) لكن بلا اتساق اتجاه.

## تشخيص إضافي: لماذا يتذبذب اتجاه RSI الخام تحديداً؟

فحص IC لكل نافذة على حدة لـ`RSI_14`/close (لا الملخّص المُجمَّع فقط) يكشف نمطاً غير عشوائي: **27 من 30 نافذة سالبة الاتجاه** (منها 25 معنوية إحصائياً بـ`p<0.05`، بعضها بقوّة استثنائية مثل `IC=-0.31`)، لكن **نافذتان فقط موجبتان بقوّة معنوية** (`نافذة 6: IC=+0.122, p=0.000` و`نافذة 24: IC=+0.148, p=0.000`) — وهما وحدهما كافيتان لإسقاط `consistent_sign`. مطابقة تواريخ هاتين النافذتين بسجلّ بناء البيانات:

- **النافذة 6** (فترة اختبار ≈ 2021-05-28 إلى 2021-07-30): مباشرة بعد **انهيار مايو 2021** (BTC من ~58 ألف إلى ~30 ألف) — فترة ارتداد حادّ من القاع.
- **النافذة 24** (فترة اختبار ≈ 2022-11-19 إلى 2023-01-21): مباشرة بعد **انهيار FTX** (نوفمبر 2022) — فترة ارتداد حادّ مماثلة من قاع صادم.

هذا يطابق **بالضبط** التحذير الوارد في بحث RSI أعلاه: تشبّع الشراء (RSI مرتفع) "يحدث في كل اتجاه صاعد قويّ" ويكون **مؤكِّداً للاستمرار لا مُنذِراً بالانعكاس** في مثل هذه الفترات — عكس افتراض الانعكاس الذي يفترضه اختبار IC الخطّي المفرد ضمنياً. بعبارة أخرى: RSI الخام **ليس عشوائي الفشل** بل يفشل في نظام سوقي محدَّد بوضوح (الارتداد الحادّ بعد ذعر بيعي)، وينجح بقوّة في كل الأنظمة الأخرى تقريباً. هذا **لا يغيّر الحكم النهائي** (لا حالة وسطى: `consistent_sign=False` يبقى رفضاً حسب معيار هذا المشروع)، لكنه يفتح سؤالاً بحثياً واضحاً لم يُختبَر بعد: هل ميزة "نظام السوق" (مثلاً: هل آخر 30-60 يوماً شهدت هبوطاً حاداً >30%) كتصفية/تفاعل مع RSI تحلّ هذا التناقض؟ هذا يطابق تماماً اقتراح [القسم ٧ من مراجعة الأدبيات](external_literature_review.md#٧-اكتشاف-نظام-التقلّب-volatility-regime-detection-فكرة-تحسين-لـh003-نفسها-لا-ميزة-مستقلّة-جديدة) — لم يُختبَر صراحة بعد، أولوية متوسطة للمتابعة.

## الخلاصة

**الاستخدام "الصحيح" لهذه المؤشرات (تقاطعات، تباعد، دمج DI/ADX، %B) لم يتفوّق على قيمها الخام — وفي حالة RSI، كان أضعف منها فعلياً.** هذا لا يعني أن الصيغة الخام "أفضل" بمعنى عملي (كلتاهما مرفوضتان)، لكنه يوضّح أن **مشكلة هذه المؤشرات هنا ليست "طريقة الاستخدام"** — المشكلة الجوهرية نفسها التي رصدتها الدفعات السابقة: **لا اتساق اتجاه عبر نوافذ زمنية منفصلة (`consistent_sign=False`)**، بصرف النظر عن كيفية تحويل القيمة الخام. النمط اللافت (RSI/BBP/SUPERT_STRETCH بـ`frac_significant` مرتفع نسبياً لكن اتجاه متذبذب) يدعم فرضية متكرّرة في هذا المشروع: **الأصول المختلفة/الحقب الزمنية المختلفة تستجيب لهذه المؤشرات بإشارة معاكسة أحياناً** — تماماً كما ظهر مع فيشر واختبار التبديل المشترك سابقاً.

## أسئلة مفتوحة

- تقريب `RSI_DIVERGENCE` الحالي (فرق درجة معيارية) قد لا يعكس التعريف الحرفي (مطابقة قمم/قيعان `FRACTAL` مؤكَّدة) — تجربة نسخة أكثر حرفية باستخدام ميزة `FRACTAL_high`/`FRACTAL_low` الموجودة أصلاً ممكنة لاحقاً، لكن أولوية منخفضة بالنظر لضعف النتيجة الحالية أصلاً.
- `Bollinger BandWidth` (Squeeze) لم يُختبَر — يحتاج هدف "تقلّب مستقبلي" لا "عائد مستقبلي" (سؤال مختلف جوهرياً عن IC الاتجاهي)، خارج نطاق محور H003 الحالي.
- تفعيل `funding_rate`/`open_interest` الحقيقيين يبقى الأولوية الأولى الفعلية من [مراجعة الأدبيات](external_literature_review.md) — لم يتأثّر هذا الاستنتاج بنتيجة هذا الاختبار.

## المصادر

- [How to Use the Supertrend Indicator Effectively (LuxAlgo)](https://www.luxalgo.com/blog/how-to-use-the-supertrend-indicator-effectively/)
- [How to use the SuperTrend indicator (forex.com)](https://www.forex.com/en/news-and-analysis/how-to-use-the-supertrend-indicator/)
- [Supertrend Indicator: Formula, Best Settings, Signals & Strategies (Mudrex)](https://mudrex.com/learn/supertrend-indicator/)
- [Bollinger Band Squeeze (StockCharts ChartSchool)](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/bollinger-band-squeeze)
- [Bollinger Band Width and %B: An Overview (TrendSpider)](https://trendspider.com/learning-center/bollinger-band-width-and-b-an-overview/)
- [Bollinger Bands: The Complete Guide by John Bollinger (IG)](https://www.ig.com/en/trading-strategies/what-are-bollinger-bands-and-how-do-you-use-them-in-trading--200129)
- [MACD-Histogram (StockCharts ChartSchool)](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-histogram)
- [Using MACD Histogram to Trade the Traditional MACD Crossover (Macroption)](https://www.macroption.com/macd-histogram-crossover/)
- [Average Directional Index (ADX) (StockCharts ChartSchool)](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx)
- [Average Directional Index (ADX) (TrendSpider)](https://help.trendspider.com/kb/indicators/average-directional-index-adx)
- [Wilder's DMI (ADX) | Definition, Components, and How to Use (FinanceStrategists)](https://www.financestrategists.com/wealth-management/fundamental-vs-technical-analysis/wilders-dmi-adx/)
- [Relative strength index (Wikipedia)](https://en.wikipedia.org/wiki/Relative_strength_index)
- [RSI Divergence: 4 Types, Crypto Examples (Plisio)](https://plisio.net/education/rsi-divergence-bullish-bearish)
- [How to Read RSI Divergence (Whale Story)](https://goraestory.com/en/academy/rsi-divergence/)
