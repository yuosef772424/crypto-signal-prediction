# H07 — تداول ورقي على بيانات لم يرها أحد (من 2026-10-01)

القاعدة **مجمّدة** كما في `research/edge_discovery/16_h07_oos.py` (وأُعيد تأكيدها على أسعار Binance في `research/h07_volsizing`):
شراء BTC وETH فقط حين يكون عائد آخر 30 يوماً موجباً، والوزن = 0.02 ÷ الانحراف المعياري لآخر 30 يوماً (حدّ أقصى 3)، مقسوماً
على العملتين؛ تكلفة 0.06% لكل جانب. أول قرار عند إغلاق 2026-09-30، فالربح الورقي يبدأ من 2026-10-01.

- التحديث: `python tools/h07_forward.py` — يجلب الإغلاقات اليومية من GitHub (تُحدَّث يومياً)، ويضيف الأيام الجديدة إلى
  [`ledger.csv`](ledger.csv) ويرفض إعادة كتابة أي يوم سابق.
- **لا يُعدَّل شيء في القاعدة بناءً على هذا السجلّ.** هو الاختبار الحقيقي الوحيد المتبقي لـ H07.
- معيار الحكم (مسجّل مسبقاً): بعد 12 شهراً، Sharpe المرشّح > Sharpe الشراء والاحتفاظ 50/50، وأقصى تراجع أصغر.

## صيغ الظل (من `research/h07_combos`، سجلّ الفشل F-0079؛ F-0065 على PR #13)
تُتابَع بجانب H07 دون أن تحلّ محلّه: `ledger_v3_10coins.csv` (نفس القاعدة على 10 عملات) و`ledger_v4_ens_10coins.csv`
(متوسط إشارات 10/20/30/60 يوماً على 10 عملات). كانت أفضل من H07 على val وtest لكن ضمن هامش الضجيج. **الحكم بعد 12 شهراً**
بنفس اختبار الفرق المزدوج (bootstrap) على val + الفترة الأمامية مجتمعتين؛ لا تُعتمد قبل ذلك. التحديث: `python tools/h07_forward.py --variant all`.

## Time seal and access log (added 2026-10-13)
Under `docs/research/plan` (time seal: bars with open time >= 2026-10-01 are forward data for every coin), this ledger IS
the R4 forward access for the frozen H07 rule: `tools/h07_forward.py` reads daily bars from 2026-10-01 (decision days
from 2026-09-30, whose P&L starts on 2026-10-01), so those reads are its intended use, not a leak of sealed data. They
are logged here: every row of `ledger.csv`, `ledger_v3_10coins.csv` and `ledger_v4_ens_10coins.csv` is one day of forward
bars that has been read, and the files are append-only. The judging rules are in `cards/E-h07fwd-1.md`; the shadow
variants are judged on these ledgers only (not pooled with VAL/TEST), which supersedes the "val + forward" wording above.
