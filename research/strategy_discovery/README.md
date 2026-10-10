# strategy_discovery — البحث عن استراتيجية تداول قابلة للتحقق

## الهدف والقيد الأساسي
البحث عن إشارة أو مجموعة شروط تحقق ربحاً صافياً بعد التكلفة، مع **عدم الادعاء بربح** قبل أن تجتاز الإشارة الاختبارات الثلاثة:
متانة على العملات الأصلية، ثم عملات لم تُستخدم، ثم بيانات طازجة. أي نتيجة هنا مرشح، لا استراتيجية مثبتة.

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
- `rotation.py`: E-disc-004 (تدوير الزخم المقطعي كمحفظة، تكلفة على الدوران).

## البطاقات والنتائج
| البطاقة | ما اختبرته | النتيجة |
|---|---|---|
| E-disc-001 | 42 تكويناً (7 عائلات، 3 بوابات، 4h و1d) على DEV/VAL | 3 مرشحين، كلها اختراقات يومية |
| E-disc-002 | متانة المرشحين الثلاثة | الثلاثة متينة؛ S3 حساسة لتأخير الدخول (من +0.196 إلى +0.068) |
| E-disc-003 | المرشحون على BCH وTRX وZEC و2026-10 | S1 وS3 **تجتاز** (+0.116 و+0.135R)، S2 **تفشل** (−0.118R)، الطازج غير كافٍ |
| E-disc-004 | تدوير الزخم المقطعي (18 إعداداً) | **مرفوض**: q≥0.47، وتراجع فوق 35% في كل الحالات |
| E-disc-005 | 3 عائلات جديدة (18 تكويناً) مع BH على 60 تكويناً | **1d volume_donchian بدون بوابة** مرشح جديد (q=0.070، VAL +0.220R) |
| E-disc-006 | الجديد على العملات غير المستخدمة | متين على الأصلية، **يفشل** على غير المستخدمة (ZEC فقط موجبة)؛ F-0072 |

## الخلاصة الحالية
- **العائلة الأكثر ثباتاً:** اختراق Donchian (20 شمعة) اليومي، مع أو بدون فلتر ADX. تجتاز الاختبار على العملات غير المستخدمة بعملتين من ثلاث.
- **النقطة الحاسمة:** النتائج الموجبة تتركز في عملة واحدة أو اثنتين (ZEC وTRX)، وBCH سالبة في كل المرشحين. هذا يعني أن العائلة **لم تُثبت أنها تعمّم عبر العملات**.
- **البيانات الطازجة لم تكفِ بعد:** نافذة 8 أيام تعطي صفقة واحدة لكل تكوين. أي حكم على الحافة يحتاج أسابيع إلى أشهر من بيانات بعد 2026-10-08.

## الإخفاقات المسجلة في هذه الدراسة
| المعرّف | الموضوع |
|---|---|
| F-0070 | تدوير الزخم المقطعي (E-disc-004) |
| F-0071 | Bollinger breakout مع ADX على العملات غير المستخدمة (S2) |
| F-0072 | Donchian بتأكيد الحجم على العملات غير المستخدمة (E-disc-006) |

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
ملفات الصفقات الكاملة تُكتب خارج المستودع (`/home/user/research/strategy_discovery/`) لأن حد CI هو 1MB للملف.
