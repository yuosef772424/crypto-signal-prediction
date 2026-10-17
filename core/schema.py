"""
PURPOSE:  The data schema names shared by data, workflow, discovery, cross_asset and tools: the last_candles column order (LAST_COLUMNS, TS_COL, LAST_COLUMN_INDEX), its dtype, the price target columns and the target-mode names.
TAGS:     LAST_COLUMNS, TS_COL, LAST_COLUMN_INDEX, LAST_DTYPE, TARGET_COLUMNS, TARGET_MODES, NO_RELATIVE_BASES, ENTRY_CLOSE_REGS, last_candles, schema, column order
PITFALLS: LAST_COLUMNS is the physical column order of every saved dataset's last_candles array: reordering or inserting breaks existing datasets, checkpoints and every positional index, so only APPEND (and then update the "(N, 7)" docs). It is a tuple (immutable); `.index(name)` is the way to get a position. LAST_COLUMN_INDEX is a read-only name -> index mapping (cross_asset's old LC dict). TARGET_MODES lists the base modes of workflow.retarget_splits; "+relative" may be appended to every base not in NO_RELATIVE_BASES, and the alias "relative" means "return+relative".
"""
from types import MappingProxyType

#: أعمدة مصفوفة ``last_candles`` (N, 7) بترتيبها الفعلي في كل مجموعة بيانات محفوظة.
LAST_COLUMNS = ('last_high', 'last_low', 'last_close', 'timestamp',
                'future_close', 'future_low_min', 'future_high_max')

#: اسم العمود -> فهرسه داخل ``last_candles`` (للقراءة فقط).
LAST_COLUMN_INDEX = MappingProxyType({name: i for i, name in enumerate(LAST_COLUMNS)})

#: فهرس عمود الطابع الزمني داخل ``last_candles``.
TS_COL = LAST_COLUMNS.index('timestamp')

#: ✅ ``last_candles`` تُخزَّن ``float64`` لا ``float32``: الطابع الزمني بالنانوثانية
#: يبلغ ~1.7e18، وهو يفقد ~12 ثانية من الدقة في ``float32``. هذا لا يضرّ على فريم
#: 4h لكنه يكسر أي تقسيم أو ربط زمني دقيق (وفريمات الدقائق تماماً). التكلفة
#: مهملة: المصفوفة (N, 7) مقابل مصفوفات X بحجم (N, T, F).
LAST_DTYPE = 'float64'

#: أعمدة الأسعار اللازمة لحساب الأهداف — تُبقى في البيانات دائماً حتى لو
#: استُبعدت من مدخلات النموذج عبر ``exclude_from_features``.
TARGET_COLUMNS = ('high', 'low', 'close')

#: أوضاع الهدف الأساسية في ``retarget_splits`` (+ اختيارياً "+relative"؛ و"relative" وحدها = "return+relative").
TARGET_MODES = ("return", "return_close", "scaled", "magnitude", "volnorm", "entry_range")

#: أوضاع لا تقبل لاحقة "+relative" (الوسيط المقطعي لا يُطبَّق على موقع الإغلاق).
NO_RELATIVE_BASES = ("entry_range",)

#: تعريفات انحدار close في وضع entry_range: "abs_return" = |عائد الإغلاق من P|، "range_pos" = موقع الإغلاق في المدى [0,1].
ENTRY_CLOSE_REGS = ("abs_return", "range_pos")
