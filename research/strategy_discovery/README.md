# strategy_discovery — البحث عن استراتيجية تداول قابلة للتحقق

> Pre-registration caveat: every card of this PR was committed together with its results, so all results are exploration, not confirmation. See `research/strategy_discovery/PREREGISTRATION_NOTE.md`.

## الهدف والقيد الأساسي
البحث عن إشارة أو مجموعة شروط تحقق ربحاً صافياً بعد التكلفة، مع **عدم الادعاء بربح** قبل أن تجتاز الإشارة الاختبارات الثلاثة:
متانة على العملات الأصلية، ثم عملات لم تُستخدم، ثم بيانات طازجة. لا نتيجة هنا استراتيجية مثبتة ولا مرشحة للتداول: كل ما في هذه الدراسة استكشاف (انظر `PREREGISTRATION_NOTE.md`).

## الفرع والبيانات
- الفرع: `claude/strategy-discovery`، مبني على آخر commit للمشروع (`68d2cef`).
- البيانات: Speirsy11/crypto-dataset، شموع 1h spot، 10 عملات، تنتهي في **2026-10-08 23:00 UTC** (المسار `/home/user/research/ohlc_full`).
- التقسيم:
  - DEV 2019–2021 و VAL 2022–2023: الاختيار على العملات الأصلية السبع.
  - TEST 2024–2026-09: مُستهلك جزئياً من تجارب سابقة.
  - العملات غير المستخدمة BCH وTRX وZEC: لم تُحمَّل في أي شاشة، واستُخدمت مرة واحدة في E-disc-003 و006 (على فترة مُستهلكة).
  - الطازج 2026-10-01 → 2026-10-08: نافذة قصيرة (حوالي 8 شموع يومية)، عينة فحص فقط.

## الأدوات
- `tools/trade_engine.py`: محرك المحاكاة المشترك (أُخرج من `research/indicator_strategy` لأن الدراسات لا تستورد بعضها). مختبر في `tests/test_trade_engine.py`.
- `grid.py`: شاشة E-disc-001 (7 عائلات × 3 بوابات × فريمين).
- `grid2.py`: شاشة E-disc-005 (Supertrend، Keltner squeeze، Donchian بتأكيد الحجم)، مع BH على الحملة كلها (60 تكويناً).
- `robustness.py`: متانة E-disc-002 (تكلفة مضاعفة، تأخير شمعة، شبكة 36 خروجاً، فحص السنوات).
- `oos.py`: E-disc-003 (عملات غير مستخدمة وعينة طازجة).
- `survivor_checks.py`: E-disc-006 (متانة + عملات غير مستخدمة + طازج لتكوين واحد).
- `significance.py`: مراجعة لاحقة (post-hoc) لـE-disc-003 و006: يعيد توليد سجلات صفقات العملات غير المستخدمة ويحسب t الساذج وt المجمَّع باليوم والتكلفة المضاعفة وحذف أفضل 5%. يتحقق أولاً من تطابق n والمتوسط مع الـCSV المحفوظ.
- `intraday_macd.py`: E-srch-002 (MACD على 15m و1h). `seasonality15m.py`: E-srch-003 (موسمية ساعة اليوم على 15m). `forward.py`: E-fwd-001 (اختبار أمامي لفرضية واحدة مجمَّدة).
- `rotation.py`: E-disc-004 (تدوير الزخم المقطعي كمحفظة، تكلفة على الدوران).

## البطاقات والنتائج
| البطاقة | ما اختبرته | النتيجة |
|---|---|---|
| E-disc-001 | 42 تكويناً (7 عائلات، 3 بوابات، 4h و1d) على DEV/VAL | 3 تكوينات اجتازت شاشة المؤلف، كلها اختراقات يومية (انتقاء، لا تأكيد) |
| E-disc-002 | متانة المرشحين الثلاثة | الثلاثة متينة؛ S3 حساسة لتأخير الدخول (من +0.196 إلى +0.068) |
| E-disc-003 | S1 وS2 وS3 على BCH وTRX وZEC و2026-10 | S1 (n=98، +0.116R) وS3 (n=158، +0.135R) اجتازتا قاعدة المتوسط وpf وعملتين من ثلاث، لكن **بلا دلالة إحصائية**: t الساذج 0.96 و1.42، وt المجمَّع باليوم 0.93 و1.43؛ BCH سالبة؛ بعد حذف أفضل 5% من الصفقات +0.042 و+0.062R. **لا حافة معنوية على أصول غير مرئية** (F-0076، F-0077). S2 تفشل (−0.118R). الطازج غير كافٍ (صفقة واحدة) |
| E-disc-004 | تدوير الزخم المقطعي (18 إعداداً) | **مرفوض**: q≥0.47، وتراجع فوق 35% في كل الحالات |
| E-disc-005 | 3 عائلات جديدة (18 تكويناً) مع BH على 60 تكويناً | **1d volume_donchian بدون بوابة** اجتاز شاشة المؤلف (q=0.070، VAL +0.220R)؛ النتيجة الكاملة في `cards/E-disc-005_result.md` |
| E-disc-006 | الجديد على العملات غير المستخدمة | متين على الأصلية، **يفشل** على غير المستخدمة (ZEC فقط موجبة؛ t الساذج 0.91)؛ F-0072 |
| E-srch-002 | MACD على 15m و1h (6 تكوينات) | **مرفوض**: لا تكوين يجتاز؛ العائد الإجمالي أصغر من التكلفة؛ F-0073 |
| E-srch-003 | موسمية ساعة اليوم على 15m (144 تكويناً) | **مرفوض**: 0 من 144 موجبة على VAL، أعلى عائد إجمالي 8.4bp أقل من تكلفة 12bp؛ F-0075 |
| E-fwd-001 | اختبار أمامي لفرضية واحدة مجمَّدة (Donchian 20 يومي بلا بوابة) | **قيد التشغيل**: 0 صفقات مغلقة حتى آخر شمعة 2026-10-08، لا نتيجة ولا قرار |

## الخلاصة الحالية
- **لا حافة معنوية على أصول غير مرئية.** اختراق Donchian (20 شمعة) اليومي، مع أو بدون فلتر ADX، يعطي متوسطاً موجباً صغيراً على BCH وTRX وZEC (S1 +0.116R، S3 +0.135R)، لكن t < 2 (الساذج 0.96 و1.42، المجمَّع باليوم 0.93 و1.43). بعد حذف أفضل 5% من الصفقات يهبط المتوسط إلى +0.042 و+0.062R. الجدول الكامل في `results/E-disc-003_significance.md`.
- **التركّز في عملات بعينها:** النتائج الموجبة تأتي من ZEC وTRX، وBCH سالبة في كل التكوينات. العائلة لم تُثبت أنها تعمّم عبر العملات.
- **البيانات الطازجة لم تكفِ بعد:** نافذة 8 أيام تعطي صفقة واحدة لكل تكوين. E-fwd-001 هو الاختبار الأمامي المجمَّد؛ لا نتيجة له بعد.
- **حالة السجل:** S1 وS3 مغلقتان بحالة `closed-unverified-cause` (F-0076، F-0077). شرط إعادة الفتح: تأكيد مجمَّد على 30 عملة مختومة أو أكثر وبـ375 صفقة أو أكثر (حاجز H4، `docs/research/plan/ROADMAP.md` §R5).
- **تنبيه عام:** كل بطاقات هذا الفرع أُضيفت مع نتائجها في commit واحد، فكل ما هنا استكشاف لا تأكيد.

## الإخفاقات المسجلة في هذه الدراسة
| المعرّف | الموضوع |
|---|---|
| F-0070 | تدوير الزخم المقطعي (E-disc-004) |
| F-0071 | Bollinger breakout مع ADX على العملات غير المستخدمة (S2) |
| F-0072 | Donchian بتأكيد الحجم على العملات غير المستخدمة (E-disc-006) |
| F-0073 | MACD على 15m و1h (E-srch-002) |
| F-0074 | غير الناجين على 4h في E-disc-001 وE-disc-005 |
| F-0075 | موسمية ساعة اليوم على 15m (E-srch-003) |
| F-0076 | S1 (Donchian 20 مع ADX25) على BCH وTRX وZEC: بلا دلالة، t < 1 (E-disc-003) |
| F-0077 | S3 (Donchian 20 بلا بوابة) على BCH وTRX وZEC: بلا دلالة، t < 2 (E-disc-003) |

## ما لم يُختبر بعد
- Donchian 20 على عدد أكبر من العملات (تتوفر في المستودع 10 فقط، ولا توجد عملات أخرى حالياً).
- تحجيم المراكز حسب التقلب والتراجع على المحفظة الكاملة (هنا النتيجة لكل صفقة بوحدات R فقط).
- تأخير التنفيذ على الفريمات الأصغر، وتأثير الانزلاق الحقيقي على العملات الصغيرة.

## إعادة التشغيل (من جذر المستودع)
```
python research/strategy_discovery/grid.py --data /home/user/research/ohlc --out research/strategy_discovery/results
python research/strategy_discovery/robustness.py --data /home/user/research/ohlc --out research/strategy_discovery/results
python research/strategy_discovery/oos.py --data /home/user/research/ohlc_full --out research/strategy_discovery/results
python research/strategy_discovery/rotation.py --data /home/user/research/ohlc --out research/strategy_discovery/results
python research/strategy_discovery/grid2.py --data /home/user/research/ohlc --out research/strategy_discovery/results
python research/strategy_discovery/survivor_checks.py --data /home/user/research/ohlc_full --family volume_donchian --gate none
```
ملفات الصفقات الكاملة تُكتب خارج المستودع (`/home/user/research/strategy_discovery/`) لأن حد CI هو 1MB للملف. سجلات صفقات اختبار الأصول غير المرئية (S1 وS2 وS3 وvolume_donchian) المضغوطة محفوظة في `results/*_trades_*.csv` وتُعاد من `significance.py`.

```
python research/strategy_discovery/significance.py --data /home/user/research/ohlc_full --out research/strategy_discovery/results
```
