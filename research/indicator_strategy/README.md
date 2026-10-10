# indicator_strategy — دراسة المؤشرات والشروط على الفريمات 4h و1d

## السؤال
أي إشارات من المؤشرات الفنية تستحق المتابعة بعد التكلفة، وما الشروط التي تحسّن النتيجة أو تضعفها؟

## المصادر من البحث وحالة التحقق منها
| المصدر | ما ادّعاه | الحالة |
|---|---|---|
| [Zarattini, Pagani, Barbon — Catching Crypto Trends (SFI 25-80)](https://concretumgroup.com/catching-crypto-trends-a-tactical-approach-for-bitcoin-and-altcoins/) | Donchian متعدد الأطوال مع تحجيم بالتقلب، Sharpe > 1.5 صافٍ | **لم أتحقق من الورقة نفسها** (الوصول إلى المصدر محجوب). المؤلفون مرتبطون بجهة تسويق، والنتائج غير مستقلة |
| [13 famous trading strategies graded on 6 years of crypto (dev.to)](https://dev.to/tessen/we-graded-13-famous-trading-strategies-on-6-years-of-data-all-13-failed-26o4) | ADX +10.7 bp، RSI عكسي −19.8 bp، 0 من 13 اجتازت البوابات الخمس | **غير متحقق**: المصدر محجوب بسياسة الشبكة (403). معلومات من مقتطف البحث فقط |
| [ADX: trend vs range (MQL5)](https://www.mql5.com/en/blogs/post/774925) وسكربتات TradingView المذكورة | ADX < 25 نطاق، > 25 اتجاه؛ RSI تحت 30 للارتداد | مدوّنة ونصوص مجتمعية، **ليست بحثاً محكّماً** |
| [Gainium squeeze bot community post](https://community.gainium.io/t/strategy-final-squeeze-bot-long-short/4693) | Bollinger/Keltner squeeze على 1h | **غير متحقق** من الأرقام، وأدائه يتغير بالنظام |

**الخلاصة من المصادر:** الأدلة ضعيفة، وأغلبها مدوّنات واختبارات مجتمعية. الادعاء الأقوى (Zarattini) غير مستقل ولم أتمكن من فحصه. لذلك لم أعتمد على أي رقم منها كدليل؛ اختبرت كل إشارة على بياناتنا.

## الإشارات المختبرة (E-ind-001)
| الإشارة | الشرط | التفسير والاستخدام الصحيح |
|---|---|---|
| `rsi_mr` | RSI يعبر 30 صعوداً / 70 هبوطاً | ارتداد؛ يصلح فقط في سوق عرضي (ADX منخفض). فشل هنا |
| `macd_cross` | عبور هيستوغرام MACD للصفر | زخم متأخر؛ يحتاج تأكيداً من الاتجاه. فشل هنا |
| `bb_break` | إغلاق يخترق نطاق Bollinger العلوي/السفلي | اختراق؛ يفشل عند التقلب المفرط. يعمل على 1d |
| `donchian` | إغلاق يخترق أعلى/أدنى 20 شمعة سابقة | اختراق الاتجاه (أساس ورقة Zarattini). يعمل على 1d |
| `adx_trend` | ADX يعبر 25 مع اتجاه DI | بداية اتجاه؛ عينة صغيرة على 1d، ضعيف في VAL |
| `ema_pullback` | EMA20>EMA50 وتراجع إلى EMA20 ثم استعادة | ارتداد داخل الاتجاه؛ ضعيف على الفريمين |

## التكلفة بالـR (مقيسة)
وسط التكلفة على كل العملات: **0.027R على 4h و0.010R على 1d** عند وقف 2×ATR. هذا يحقق الشرط R1 لإعادة فتح الإخفاق F-0054، لكن فقط على الفريمين 4h و1d. على 15m و1h لم يُعَد الفتح.

## التقسيم
- DEV: 2019–2021 (الاختيار). VAL: 2022–2023 (فحص الاتساق، وقد رُئي قبل التعديلات). TEST: 2024–2026-09 (استُخدم مرة واحدة في E-ind-003).
- TEST ليست عينة جديدة: تتداخل مع فترات استُخدمت في H06 وH07 وH18. أي ادعاء بحافة يحتاج بيانات بعد 2026-09-30.

## البطاقات والنتائج
| البطاقة | الهدف | الحكم |
|---|---|---|
| `cards/E-ind-001.md` | شاشة 24 تكويناً على DEV وVAL | 4 تكوينات مرشحة (كلها 1d)؛ RSI وMACD وبعض 4h مرفوضة |
| `cards/E-ind-002.md` | حذف شروط: البيع، وADX > 40 | الشراء فقط على Donchian مرشح استكشافي؛ ADX > 40 يفشل على VAL |
| `cards/E-ind-003.md` | اختبار واحد على TEST | **مقبول كمرشح للتأكيد** خارج الفترة: +0.29R، pf 1.66، t 3.47، 6/7 عملات |

## أين تنجح وأين تفشل (E-ind-003 على TEST)
- الانخفاض عبر السنوات: 2024 +0.41R، 2025 +0.26R، 2026 +0.10R.
- ETH سالبة (−0.05R)؛ DOGE وXRP الأعلى.
- ADX فوق 40 سالب (−0.16R)؛ السوق الهادئ أفضل بكثير (+0.67R).
- منحنى الربح مسطّح تقريباً بين 2022 و2023.

الصور في `figures/`: منحنى الربح التراكمي، ومقارنة المناطق، وعينات شموع لصفقات رابحة وخاسرة.

## الحدود
- المرشح لم يُثبت كحافة؛ اختياره من 24 تكويناً ثم 3 تعديلات يرفع احتمال الصدفة.
- عينة TEST صغيرة (205 صفقة) ومستهلكة الآن. التعديلات المقترحة من TEST (ADX>40، استبعاد ETH) لا يمكن اختبارها عليها مرة أخرى.

## الملفات غير المتتبَّعة
`E-ind-001_trades.csv` (نحو 6MB، كل الصفقات) حُذف من المستودع لأن الحد الأقصى لأي ملف متتبَّع 1MB، وهو قابل لإعادة الإنتاج من `screen.py` (الأمر أدناه).

## إعادة التشغيل
```
python research/indicator_strategy/screen.py   --data /home/user/research/ohlc --out research/indicator_strategy
python research/indicator_strategy/variants.py --data /home/user/research/ohlc --out research/indicator_strategy
python research/indicator_strategy/test_final.py --data /home/user/research/ohlc --out research/indicator_strategy   # مرة واحدة فقط
python research/indicator_strategy/charts.py   --data /home/user/research/ohlc --out research/indicator_strategy/figures
```
