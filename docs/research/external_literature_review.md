# مراجعة أدبيات خارجية — مصادر ميزات جديدة بعد استنفاد pandas_ta

**الحالة: بحث/توثيق (لا تنفيذ كود)،** استكمالاً لإغلاق مسار `SURVEY_CANDIDATES_ROBUST` في [الدفعات 1-5](pandas_ta_batch4_5_final.md). الهدف: بعد اختبار 154 مرشّحاً من مكتبة pandas_ta بلا نتيجة جديدة، ما الذي تقوله الأدبيات الأكاديمية والمصادر المتخصّصة عن ميزات فعّالة فعلاً في توقّع عوائد العملات الرقمية قصيرة المدى؟ ولأيّ منها بيانات متاحة لهذا المشروع بالفعل؟

## المنهجية

بحث ويب مُوجَّه (لا تصفّح مصادر فردية بعمق) عبر 8 استعلامات تغطّي: (أ) أيّ فئات ميزات تنجح تجريبياً في الأدبيات الحديثة (2024-2026)، (ب) عدم توازن دفتر الأوامر (order book imbalance)، (ج) مقاييس على السلسلة (on-chain: MVRV، تدفّق البورصات)، (د) أثر التتابع بين BTC والعملات البديلة (lead-lag)، (هـ) معدّل التمويل والمراكز المفتوحة (funding rate/open interest)، (و) نسبة حجم الشراء/البيع الفوري (taker buy/sell ratio)، (ز) انعكاسات تصفية المراكز (liquidation cascades)، (ح) اكتشاف نظام التقلّب (volatility regime detection). كل نتيجة أدناه تُقاس بمعيارين: هل بيانات هذا المشروع تدعمها فعلاً الآن؟ وهل اختُبِرت مسبقاً في هذا المشروع بصورة أو بأخرى؟

## النتائج بحسب الفئة

### ١) الميزات البسيطة (سعر/حجم/تقلّب/زخم) تتفوّق على الهندسة المعقّدة — يتّسق مع تجربة المشروع

أبحاث 2024-2025 (مثل *Predicting cryptocurrency returns with machine learning*، ScienceDirect) تُظهر أن **مجموعة صغيرة مركَّزة من مؤشّرات الحجم/التقلّب/الزخم كانت كافية للتنبؤ قصير المدى بدقّة أعلى من نماذج غنيّة بعشرات المؤشّرات الهندسية** — بل إن استخدام حدّ أدنى من ميزات السعر/الحجم الخام غالباً ما تفوّق على نماذج مُثقَلة بمؤشّرات معقّدة. هذا **يتّسق تماماً** مع نتيجة هذا المشروع بعد 154 مرشّحاً: الإشارة الوحيدة الحقيقية مصدرها التقلّب (`NATR_14`/H003)، لا عشرات مذبذبات الزخم المُختبَرة.

**الخلاصة**: لا حاجة للبحث عن "مؤشّر سحري" إضافي من نفس عائلة pandas_ta — الأدبيات تؤكّد أن هذا الاتجاه مُستنفَد فعلاً بشكل عام، لا في هذا المشروع فقط.

### ٢) عدم توازن دفتر الأوامر (Order Book Imbalance) — **غير قابل للتنفيذ الآن**

أقوى إشارة قِصَر أفق في الأدبيات: علاقة شبه خطّية مباشرة بين عدم توازن التدفّق (order flow imbalance) وتغيّر السعر خلال ثوانٍ إلى عشرات الثواني، تتّسق عبر عملات بفارق رتبة سوقية كاملة. **لكنها تحتاج بيانات المستوى الثاني لدفتر الأوامر (L2 order book) بدقّة زمنية عالية (ثوانٍ)** — هذا المشروع يعمل على شموع يومية/ساعية من REST API عادي (OHLCV فقط)، لا بيانات دفتر أوامر مطلقاً. **غير قابل للتنفيذ بدون مصدر بيانات جديد كلياً (WebSocket L2 stream)** — خارج نطاق هذا المشروع حالياً.

### ٣) مقاييس على السلسلة (On-chain: MVRV، تدفّق البورصات) — **أولوية منخفضة لهذا المشروع تحديداً**

مقاييس مفيدة فعلاً (MVRV، تدفّق البورصات، العناوين النشطة) — لكنها **مصمَّمة أساساً لبيتكوين تحديداً وعلى أفق دوري/شهري (تموضع السوق)**، لا 50 عملة بديلة متنوّعة على أفق يومي قصير كما في هذا المشروع. بيانات on-chain موثوقة لأغلب الـ50 عملة (خاصةً الأصغر رسملة) غير متاحة بسهولة أو غير متجانسة عبر السلاسل المختلفة. **أولوية منخفضة** ما لم يتحوّل هدف المشروع لأفق أطول أو التركيز على BTC/ETH فقط.

### ٤) أثر التتابع بين BTC والعملات البديلة (Lead-Lag) — **اختُبِر مرّتين مسبقاً في هذا المشروع، رُفِض كلتا المرّتين**

الأدبيات (*Cross-cryptocurrency Return Predictability*، وغيرها) تُظهر أن **عوائد BTC المتأخّرة زمنياً تتنبّأ بعوائد عملات أخرى نتيجة بطء انتشار المعلومات** (BTC يتفاعل فوراً مع الصدمات، العملات الصغيرة تتأخّر). لكن هذا المشروع **اختبر نسخاً من هذه الفكرة مرّتين مسبقاً بصورة خطّية بسيطة، ورُفِضت كلتاهما**:

- [معمارية مشتركة بين الأصول](cross_asset_architecture.md): متوسط عائد الأصول الأربعة الأخرى في نفس اللحظة (تعاصري لا متأخّر) — 6 تجارب، كلّها مرفوضة.
- `MKT_beta` (في PR الحالي، على 50 أصلاً/30 نافذة): ميزة `MKT_ret_1` (عائد BTC الفعلي، `market_context.enabled=True`) — مرفوضة (`mean_ic` ≤ -0.07، `consistent_sign=False`).

**الفارق المهمّ**: الأبحاث الإيجابية غالباً تستخدم **بناء محفظة طويلة/قصيرة (long-short portfolio sort)** — ترتيب الأصول حسب حساسيتها لعائد BTC المتأخّر ثم قياس عائد فارق أعلى/أدنى مجموعة — لا ارتباط IC مباشر مجمَّع كما في محور هذا المشروع. **هذا لا يعني أن الفكرة صحيحة بالضرورة لو أُعيد صياغتها**، لكنه يعني أن رفض هذا المشروع لها بصيغتها الخطّية البسيطة **لا يناقض** الأدبيات — لم يُختبَر بعد بمنهجية "الترتيب النسبي" الفعلية. أولوية منخفضة نظراً لكلفة إعادة الصياغة مقابل الأدلة الحالية الضعيفة.

### ٥) معدّل التمويل والمراكز المفتوحة (Funding Rate / Open Interest) — **⭐ الأولوية الأولى الموصى بها**

هذه الفئة الأقوى دعماً في البحث لهذا المشروع تحديداً، لثلاثة أسباب مجتمعة:

1. **دعم أدبي قوي ومحدَّد**: معدّل تمويل متطرّف (>0.05%-0.1% كل 8 ساعات) يسبق تاريخياً تصحيحات السوق — إشارة **انعكاس تعاكسي (contrarian mean-reversion)** موثَّقة جيداً، مرتبطة بتصفية مراكز الرافعة المالية المزدحمة.
2. **البنية التحتية جاهزة فعلاً في هذا المشروع**: `crypto_data_pipeline_v6.ipynb` يملك بالفعل `update_config({'funding_rate': {'enabled': ...}, 'open_interest': {'enabled': ...}})` ودالّتَي `load_funding_open_interest`/`_funding_oi_drive_path` — **مُعطَّلة افتراضياً فقط لعدم توفّر بيانات Drive حقيقية في هذه الجلسة المعزولة**، لا لعيب في الكود. أُشير إليها صراحةً في الخطوات التالية المؤجَّلة بخطة المشروع منذ البداية ("تفعيل funding rate/open interest في هذا المسار البحثي") لكن لم تُختبَر فعلياً بعد.
3. **يتّسق مع الإشارة الوحيدة المؤكَّدة في المشروع (H003)**: H003 نفسها إشارة **انعكاس مدفوع بالتقلّب** (`NATR_14`). معدّل التمويل المتطرّف هو **نفس فكرة الانعكاس بالضبط، من زاوية مختلفة** (ازدحام مراكز بدل تقلّب سعري) — امتداد طبيعي مدعوم بالأدبيات لنفس الموضوع الناجح الوحيد في تاريخ هذا المشروع، لا اتجاه عشوائي جديد.

**التوصية الملموسة**: في تشغيل Colab القادم (حيث Drive متاح)، فعِّل `funding_rate.enabled=True`/`open_interest.enabled=True` على نفس عيّنة الـ50 أصلاً، واختبر مرشّحين بسيطين مباشرين عبر نفس محور `signal_evaluation_axis`: (أ) `funding_rate` الخام أو تغيّره الأخير كمتنبّئ (فرضية: تمويل موجب متطرّف ↔ عائد مستقبلي سالب، والعكس)، (ب) تفاعل `funding_rate × NATR_14` (هل التمويل المتطرّف يُضخّم انعكاس التقلّب الموجود أصلاً؟) — هذا الأخير تحديداً مثير لأنه يختبر **تفاعلاً غير خطّي بين ميزتين موجودتين فعلاً**، بالضبط نوع النمط الذي ناقشناه سابقاً كقادر نموذج غير خطّي كامل على استغلاله حيث يفشل IC الخطّي المفرد لكلّ ميزة بمفردها.

### ٦) نسبة حجم الشراء/البيع الفوري (Taker Buy/Sell Ratio) — **أولوية ثانية، بيانات قريبة المنال**

مؤشّر ضغط شراء/بيع فعلي (لا مجرّد اتجاه سعر) — نسبة حجم "Taker Buy" إلى الإجمالي. **مهمّ**: هذا الحقل (`taker_buy_base_asset_volume`) هو **عمود قياسي في استجابة kline العادية من Binance** (بجانب `open/high/low/close/volume`) — **لا يحتاج نقطة API منفصلة كما funding rate/OI**، فقط الاحتفاظ بعمودين إضافيين عند سحب البيانات الخام مستقبلاً. تحقّقت محلياً: ملفّات CSV المخزَّنة في هذه الجلسة (`/tmp/realdata/csv*/*.csv`) **لا تحوي هذا العمود حالياً** — يحتاج إعادة سحب من نفس المصدر مع طلب الحقل الإضافي. أرخص من funding rate/OI لأنه من نفس نقطة البيانات الحالية، لكن يحتاج تعديل خط سحب البيانات الخام (خارج نطاق هذه الجلسة المعزولة).

### ٧) اكتشاف نظام التقلّب (Volatility Regime Detection) — **فكرة تحسين لـH003 نفسها، لا ميزة مستقلّة جديدة**

أبحاث 2025 تؤكّد أن **اكتشاف النظام (توسّع/انكماش/محايد) هو العامل الأهمّ في نجاح أو فشل استراتيجيات الانعكاس** (معامل ربح 1.62 في نظام متذبذب مقابل -0.74 في نظام مُتّجِه، بنفس الاستراتيجية بالضبط). هذا يقترح تحسيناً محدَّداً على H003 بدل ميزة جديدة: **ميزة تصنّف نظام التقلّب الحالي** (مثلاً: نسبة تقلّب قصير المدى إلى طويل المدى، أو موضع `NATR_14` الحالي ضمن توزيعه التاريخي) **قد تُحسّن أداء H003 نفسها إن أُضيفت كميزة تفاعل**، بدل أن تكون مرشّحاً IC مستقلاً يُختبَر بمعزل (فشل هذا النهج مع كل شيء آخر في هذا المشروع).

## التوصيات المرتَّبة (بانتظار قرار صريح من صاحب المشروع)

1. **⭐ تفعيل `funding_rate`/`open_interest` الحقيقيين واختبارهما عبر المحور** (بما في ذلك تفاعلهما مع `NATR_14`) — أعلى أولوية، بنية تحتية جاهزة، دعم أدبي مباشر، امتداد طبيعي لـH003.
2. إضافة عمود `taker_buy_base_asset_volume` عند أي إعادة سحب مستقبلية لبيانات الشموع الخام، لاختبار نسبة الشراء/البيع الفوري لاحقاً.
3. اختبار ميزة "نظام التقلّب" (نسبة تقلّب قصير/طويل المدى) كتفاعل مع `NATR_14` داخل H003 نفسها، لا كمرشّح IC مستقلّ.
4. إعادة صياغة فرضية "تتابع BTC/العملات البديلة" كترتيب محفظة نسبي (long-short sort) إن رغب صاحب المشروع — أولوية منخفضة نظراً لرفضها مرّتين بصيغة أبسط.
5. عدم توازن دفتر الأوامر ومقاييس on-chain: مؤجَّلة فعلياً، تحتاج مصادر بيانات جديدة كلياً خارج نطاق هذا المشروع الحالي.

## المصادر

- [Predicting cryptocurrency returns with machine learning (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0927538X25003701)
- [CryptoPulse: Short-Term Cryptocurrency Forecasting (arXiv)](https://arxiv.org/pdf/2502.19349)
- [Price Impact of Order Book Imbalance in Cryptocurrency Markets](https://towardsdatascience.com/price-impact-of-order-book-imbalance-in-cryptocurrency-markets-bf39695246f6/)
- [Exploring Microstructural Dynamics in Cryptocurrency Limit Order Books (arXiv)](https://arxiv.org/html/2506.05764v2)
- [Explainable Patterns in Cryptocurrency Microstructure (arXiv)](https://arxiv.org/pdf/2602.00776)
- [Onchain Metrics: Key Indicators for Cryptocurrency Price Prediction (Nansen)](https://nansen.ai/post/onchain-metrics-key-indicators-for-cryptocurrency-price-prediction)
- [Bitcoin price direction prediction using on-chain data and feature selection (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S266682702500057X)
- [Cross-cryptocurrency Return Predictability (ResearchGate)](https://www.researchgate.net/publication/380164431_Cross-cryptocurrency_Return_Predictability)
- [A seesaw effect in the cryptocurrency market (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0927539823000956)
- [Price Transmission from Bitcoin to Altcoins (Springer)](https://link.springer.com/article/10.1007/s10690-026-09589-z)
- [How do crypto derivatives market signals predict price movements (Gate.com)](https://www.gate.com/crypto-wiki/article/how-do-crypto-derivatives-market-signals-predict-price-movements-futures-open-interest-funding-rates-liquidation-data-long-short-ratio-and-options-explained-20260129)
- [Funding Rates in Crypto: The Hidden Cost, Sentiment Signal, and Strategy Trigger](https://quantjourney.substack.com/p/funding-rates-in-crypto-the-hidden)
- [Who sets the range? Funding mechanics and 4h context in crypto markets (arXiv)](https://arxiv.org/pdf/2601.06084)
- [Taker BUY/SELL Ratio Indicator (CryptoQuant)](https://cryptoquant.com/insights/quicktake/621a91edd962520d5c511338-Taker-BUYSELL-Ratio-Indicator)
- [Taker Buy Sell Volume/Ratio (CryptoQuant User Guide)](https://userguide.cryptoquant.com/cryptoquant-metrics/market/taker-buy-sell-volume-ratio)
- [Detecting Volatility Regimes in Crypto Markets (SSRN)](https://papers.ssrn.com/sol3/Delivery.cfm/5920642.pdf?abstractid=5920642&mirid=1)
- [Building a Mean-Reversion Strategy in Cryptocurrency Markets: Evidence from 78 Backtests (Coinquant)](https://www.coinquant.ai/blog/building-a-mean-reversion-strategy-in-cryptocurrency-markets-evidence-from-78-backtests)
- [Multivariate forecasting of bitcoin volatility with gradient boosting (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0957417425040199)
