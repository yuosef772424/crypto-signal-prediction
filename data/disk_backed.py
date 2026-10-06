"""
PURPOSE:  Disk-backed build (CONFIG['disk_backed']): per-coin spill to local .npy, ordered merge into open_memmap, RAM/scratch helpers.
TAGS:     disk backed, memmap, spill, scratch, SMALL_ARRAY_BYTES, _COPY_CHUNK_BYTES, RAM, _merge_spilled
PITFALLS: The merged output must stay byte-identical to the old np.concatenate (tests/test_disk_backed.py patches SMALL_ARRAY_BYTES/_COPY_CHUNK_BYTES in the namespace). Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.
"""
# ══════════════════════════════════════════════════════════════════════════
# بناء مدعوم بالقرص (CONFIG['disk_backed']) — رام محدودة مهما كثرت العملات
# ══════════════════════════════════════════════════════════════════════════
# نقطة الاستئناف أعلاه (npz على Drive) **متينة** لكنها تُحمَّل كاملة إلى الرام، فلا تحلّ مشكلة الرام. التسريب هنا مؤقّت
# ومحلي (/content): كل عملة تُكتب بعد معالجتها ملفات .npy غير مضغوطة (مصفوفة لكل ملف، بنوع التخزين نفسه — float16 حيث
# يطلبه الإعداد)، ثم تُحرَّر من الرام، فلا يبقى فيها إلا عملة واحدة (لكل خيط عامل). بعد آخر عملة: حساب N الكلي لكل مصفوفة،
# وحجز open_memmap، ونسخ أجزاء العملات بالترتيب نفسه على دفعات (وحذف كل جزء بعد نسخه ⇒ ذروة القرص ≈ 1× لا 2×).
# الناتج مطابق بايتاً لـ np.concatenate القديم؛ المصفوفات الكبيرة memmap للقراءة فقط، والصغيرة (<SMALL_ARRAY_BYTES) في الرام.
# نقطة استئناف Drive تعمل كما كانت: حمولة مُحمَّلة منها تمرّ عبر التسريب نفسه.
import re
import shutil
import zlib

#: مصفوفات أصغر من هذا الحدّ (بايت) تُحمَّل إلى الرام بعد الدمج/التحميل (y وbase_params وlast_candles)؛ الأكبر memmap.
SMALL_ARRAY_BYTES = 64 * 1024 * 1024
#: حجم دفعة النسخ عند الدمج والحفظ — يحدّ الرام المؤقتة بغضّ النظر عن حجم الجزء.
_COPY_CHUNK_BYTES = 32 * 1024 * 1024


def _rss_gb() -> Optional[float]:
    """الرام المقيمة الحالية لهذه العملية بالجيجابايت (لينكس: /proc/self/statm)؛ None إن تعذّر."""
    try:
        with open('/proc/self/statm') as f:
            return int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 1e9
    except Exception:                                          # noqa: BLE001
        return None


def _trim_memory() -> None:
    """gc ثم إعادة الذاكرة المحرَّرة إلى النظام (glibc malloc_trim) — بدونها يبقى RSS مرتفعاً بعد del بسبب تجزّؤ الكومة."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:                                          # noqa: BLE001
        pass


def _default_scratch_root() -> Path:
    if os.path.isdir('/content'):
        return Path('/content/pipeline_scratch')
    import tempfile
    return Path(tempfile.gettempdir()) / 'pipeline_scratch'


def _scratch_run_dir(config: dict, run_fp: str) -> Path:
    """مجلد هذا البناء: ``<scratch_dir>/<بصمة الإعدادات>/`` (``CONFIG['scratch_dir']`` أو /content/pipeline_scratch).
    يُفرَّغ عند البدء: لا ملفات قديمة تختلط بالبناء الجاري."""
    d = Path(config.get('scratch_dir') or _default_scratch_root()) / run_fp
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    return d


class _SpilledArray:
    """مرجع خفيف لمصفوفة كُتبت على القرص (.npy): المسار والشكل والنوع فقط — لا بيانات في الرام."""
    __slots__ = ('path', 'shape', 'dtype')

    def __init__(self, path, shape, dtype):
        self.path, self.shape, self.dtype = Path(path), tuple(shape), np.dtype(dtype)

    def __len__(self) -> int:
        return self.shape[0]


def _spill_payload(spill_dir, name: str, payload):
    """يكتب حمولة عملة ``(X_tf, y_t, bases, last)`` ملفات .npy ويُرجع الحمولة نفسها بمراجع :class:`_SpilledArray`."""
    X_tf, y_t, bases, last = payload
    safe = re.sub(r'[^A-Za-z0-9._-]', '_', str(name)) + f"_{zlib.crc32(str(name).encode('utf-8')):08x}"
    d = Path(spill_dir) / 'parts' / safe
    d.mkdir(parents=True, exist_ok=True)

    def _w(fname, arr):
        arr = np.ascontiguousarray(arr)
        np.save(d / fname, arr, allow_pickle=False)
        return _SpilledArray(d / fname, arr.shape, arr.dtype)

    return ({tf: _w(f'X__{tf}.npy', a) for tf, a in X_tf.items()},
            {h: _w(f'y__{h}.npy', a) for h, a in y_t.items()},
            _w('bases.npy', bases), _w('last.npy', last))


def _merge_spilled(parts: List['_SpilledArray'], out_path, ram_limit: Optional[int] = None) -> np.ndarray:
    """يدمج أجزاء العملات (بالترتيب) في ملف .npy واحد على القرص، دفعات بحجم ثابت، ويحذف كل جزء بعد نسخه.
    يُرجع memmap للقراءة فقط (أو مصفوفة في الرام إن كانت أصغر من ``ram_limit``)."""
    n = sum(p.shape[0] for p in parts)
    tail = parts[0].shape[1:]
    bad = [p.path.name for p in parts if p.shape[1:] != tail]
    if bad:
        raise ValueError(f"أشكال غير متطابقة عند الدمج ({out_path}): {bad[:3]} ≠ {tail}")
    dtype = np.result_type(*[p.dtype for p in parts])
    out = np.lib.format.open_memmap(out_path, mode='w+', dtype=dtype, shape=(n,) + tail)
    row_bytes = max(1, dtype.itemsize * int(np.prod(tail, dtype=np.int64)))
    step = max(1, _COPY_CHUNK_BYTES // row_bytes)
    pos = 0
    for p in parts:
        src = np.load(p.path, mmap_mode='r')
        for s in range(0, src.shape[0], step):
            e = min(s + step, src.shape[0])
            out[pos + s:pos + e] = src[s:e]
        pos += src.shape[0]
        out.flush()
        del src
        p.path.unlink()                      # حرّر القرص فوراً: الذروة ≈ 1× لا 2×
    assert pos == n, (pos, n)
    del out
    final = np.load(out_path, mmap_mode='r')
    if final.nbytes <= (SMALL_ARRAY_BYTES if ram_limit is None else ram_limit):
        arr = np.array(final)
        del final
        Path(out_path).unlink()
        return arr
    return final


def _count_nonfinite(X: np.ndarray) -> Tuple[int, int]:
    """(عدد NaN، عدد Inf) بدفعات — بلا مصفوفة منطقية مؤقتة بحجم X كاملة (مهمّ مع memmap بحجم جيجابايتات)."""
    if X.size == 0:
        return 0, 0
    row_bytes = max(1, X.dtype.itemsize * int(np.prod(X.shape[1:], dtype=np.int64)))
    step = max(1, _COPY_CHUNK_BYTES // row_bytes)
    nan = inf = 0
    for s in range(0, X.shape[0], step):
        blk = np.asarray(X[s:s + step])
        nan += int(np.isnan(blk).sum())
        inf += int(np.isinf(blk).sum())
    return nan, inf
