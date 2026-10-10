# hedge_recovery — دراسة الصفقات المتحوّطة (تجارب E-hedge)

## الفكرة
استراتيجيات تتعامل مع الصفقة الخاسرة بفتح صفقة معاكسة (تحوّط) بدل إغلاقها، ثم تغلق المراكز عند أهداف ربح أو عند إشارة معاكسة.

## الملفات
| الملف | الدور |
|---|---|
| `cards/E-hedge-001.md` | الصفقة الخاسرة تُتحوّط عند خسارة ≥ 1×NATR، إغلاق بعد إشارة معاكسة أو خروج عكس اتجاه NATR. **مرفوضة** (F-0064) |
| `cards/E-hedge-002.md` | تجاهل الصفقة المعاكسة الخاسرة عند الانعكاس، إغلاقها عند ربح ≥ 1×NATR، موازنة عدد الشراء مع البيع، حجم 0.5% لكل صفقة. **مرفوضة** (F-0065) |
| `sim.py` | المحاكي الأول (B / H / C) وتحميل البيانات والمؤشرات |
| `sim2.py` | المحاكي الثاني (B / E2 / C) بحجم 0.5% من رأس المال |
| `E-hedge-001_results.csv`, `E-hedge-002_results.csv` | المخرجات الخام لكل عملة وكل فترة وكل ذراع |

## البيانات والتقسيم
- Speirsy11/crypto-dataset، شموع 1h spot، BTC ETH SOL BNB XRP ADA DOGE، حتى 2026-09-30.
- DEV 2019–2021 (لا تُستخدم في قبول)، VAL 2022–2023، TEST 2024–2026-09.
- TEST ليست عينة جديدة: تتداخل مع فترات استُخدمت في H06 وH07 وH18. أي نجاح يجب إعادة اختباره على شموع بعد 2026-09-30.

## القيود المعروفة
- بيانات الشموع فيها فجوات نادرة (ساعتان أو ثلاث).
- الدخول مبني على إشارة الاتجاه (EMA20/50) واللمس والاستعادة عند إغلاق الشمعة، وينفَّذ عند افتتاح الشمعة التالية، بلا انزلاق مخصص.
- القيم الافتراضية (الحد الأقصى 6 صفقات مفتوحة لكل عملة، مدة 24 شمعة، ثابت التكلفة 0.12%) قرارات تصميم من البطاقة، لا قيم مُحسَّنة.

## إعادة التشغيل
```
python tools/fetch_crypto_dataset.py --out /home/user/research/ohlc --interval 1h \
  --coins BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT ADAUSDT DOGEUSDT --end "2026-09-30 23:00"
python research/hedge_recovery/sim.py  --data /home/user/research/ohlc --out research/hedge_recovery/E-hedge-001_results.csv
python research/hedge_recovery/sim2.py --data /home/user/research/ohlc --out research/hedge_recovery/E-hedge-002_results.csv
```
