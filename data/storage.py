"""
PURPOSE:  Save/load the prepared dataset to/from mounted Drive: pkl.gz or npy_dir (memmap) formats, latest + timestamped copies.
TAGS:     storage, save_data_to_drive, load_data_from_drive, save_dataset_dir, load_dataset_dir, npy_dir, pkl.gz, memmap, Drive
PITFALLS: Tests replace mount_drive in the namespace (outside Colab the path is relative ./<project>/preprocessed_data). Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.

تخزين وتحميل البيانات
"""
# @title
"""
حفظ/تحميل مجموعة البيانات المُجهَّزة من/إلى Google Drive.

النسخة السابقة كانت تحمل دالتين غير متوافقتين فعلياً:
  * ``save_data_to_drive`` تحفظ عبر Drive المُركَّب (``drive.mount``)، بمسار
    مُوسَّم بالوقت فقط — بلا نسخة بمسار ثابت (سطر ``save_latest`` كان
    مُعلَّقاً/معطَّلاً)، فلا توجد طريقة آلية لمعرفة أحدث ملف لاحقاً.
  * ``load_preprocessed_data_from_drive`` تُحمِّل عبر ``gdown`` + رابط
    مشاركة عام (``file_id``) — مسار مختلف تماماً عن Drive المُركَّب،
    ولا يستطيع قراءة ما تكتبه ``save_data_to_drive`` إطلاقاً.

**صيغتان للحفظ** (``CONFIG['save_format']`` أو معامل ``fmt``):
  * ``pkl.gz`` — الصيغة القديمة: ملف pickle مضغوط؛ القراءة تحمّل كل المصفوفات إلى الرام (لمجموعات صغيرة/يومية).
  * ``npy_dir`` — مجلد ``<اسم>_latest.dataset`` (مصفوفة ``.npy`` غير مضغوطة لكل مفتاح + ``meta.pkl`` للبيانات الوصفية +
    ``manifest.json``)؛ يُكتب دفعات (بلا نسخة رام)، ويُقرأ بـ ``mmap_mode='r'`` فيبقى التدريب يقرأ الدفعات كسولاً من القرص
    (:func:`load_dataset_dir`). هو ما يختاره ``'auto'`` لمجموعة بيانات ``disk_backed`` (memmap)؛ وإلا تبقى ``pkl.gz``.
  ``load_data_from_drive`` يكتشف الصيغة تلقائياً (الأحدث إن وُجدتا معاً).

هنا: الحفظ يكتب نسخة مؤرَّخة **ونسخة `_latest` فعلية** بنفس المسار
المُركَّب (``mount_drive()``، الدالة الموحّدة المستخدمة أصلاً لأرشيف
funding/OI في هذا الدفتر) — والتحميل يقرأ من نفس المسار المُركَّب مباشرة،
لا عبر رابط عام. ``load_preprocessed_data_from_drive`` (gdown/file_id)
أُبقيت للتوافق الخلفي فقط مع من لا يزال يستخدم ملفاً عاماً مُشارَكاً —
لا تستخدمها لقراءة ما يكتبه ``save_data_to_drive``.
"""
import gzip
import pickle
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

#: لاحقة مجلد مجموعة البيانات بصيغة npy_dir.
DATASET_DIR_SUFFIX = ".dataset"


def _write_npy_chunked(path, arr: np.ndarray) -> None:
    """يكتب ``arr`` بصيغة .npy دفعات متتالية (تسلسلياً، بلا نسخة رام كاملة) — يصلح لـ memmap بحجم جيجابايتات ولـ Drive."""
    arr = np.asarray(arr)
    if arr.ndim == 0 or arr.size == 0:
        np.save(path, arr, allow_pickle=False)
        return
    with open(path, 'wb') as f:
        np.lib.format.write_array_header_1_0(f, {'descr': np.lib.format.dtype_to_descr(arr.dtype),
                                                 'fortran_order': False, 'shape': arr.shape})
        row_bytes = max(1, arr.dtype.itemsize * int(np.prod(arr.shape[1:], dtype=np.int64)))
        step = max(1, _COPY_CHUNK_BYTES // row_bytes)
        for s in range(0, arr.shape[0], step):
            f.write(np.ascontiguousarray(arr[s:s + step]).tobytes())


def save_dataset_dir(data: dict, dest, verbose: bool = True) -> Path:
    """يحفظ قاموس مجموعة بيانات كمجلد: ``<key>.npy`` لكل مصفوفة numpy + ``meta.pkl`` لباقي القيم + ``manifest.json``
    (يُكتب أخيراً = علامة الاكتمال). الكتابة في مجلد مؤقّت ثم إعادة تسمية فلا يُخلَّف مجلد نصفي. يعمل مع memmap بلا رام."""
    import uuid
    dest = Path(dest)
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray) and not v.dtype.hasobject}
    meta = {k: v for k, v in data.items() if k not in arrays}
    for k, v in arrays.items():
        _write_npy_chunked(tmp / f"{k}.npy", v)
    with open(tmp / "meta.pkl", "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {"format": "npy_dir", "version": 1, "id": uuid.uuid4().hex, "created": datetime.now().isoformat(),
                "key_order": list(data), "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                                                    for k, v in arrays.items()}}
    (tmp / "manifest.json").write_text(json.dumps(manifest))
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    if verbose:
        size = sum(p.stat().st_size for p in dest.iterdir()) / 1e6
        print(f"💾 مجلد مجموعة البيانات: {dest.name} ({len(arrays)} مصفوفة، {size:,.0f} MB)")
    return dest


def load_dataset_dir(path, mmap: bool = True, local_dir=None, small_bytes: Optional[int] = None,
                     verbose: bool = True) -> dict:
    """يحمّل مجلداً كتبته :func:`save_dataset_dir` إلى قاموس مجموعة بيانات بنفس مفاتيح ملف pkl.gz.

    ``mmap=True`` (الافتراضي): المصفوفات الكبيرة ``np.load(..., mmap_mode='r')`` — لا تدخل الرام إلا الصفحات المقروءة، وbatch
    الفهارس المرتّبة (``make_shuffled_dataset``) تقرأ ما تحتاجه فقط. المصفوفات الأصغر من ``small_bytes`` (افتراضياً
    ``SMALL_ARRAY_BYTES``: y وbase_params وlast_candles) تُحمَّل إلى الرام. ``mmap=False`` يحمّل كل شيء إلى الرام.

    ``local_dir``: قراءة memmap مباشرة من Drive المُركَّب (FUSE) بطيئة وغير موثوقة، فإن كان المجلد تحت ``/content/drive`` يُنسَخ أولاً
    إلى القرص المحلي (``/content/pipeline_data/<اسم>``، تسلسلياً بلا رام؛ نسخة محلية مطابقة للـmanifest تُعاد استخدامها) ثم يُحمَّل
    منه. ``local_dir=False`` = لا نسخ أبداً؛ مسار صريح = انسخ إليه. يحتاج القرص المحلي بحجم المجلد."""
    src = Path(path)
    small = SMALL_ARRAY_BYTES if small_bytes is None else small_bytes
    if not (src / "manifest.json").exists():
        raise FileNotFoundError(f"مجلد مجموعة بيانات غير مكتمل أو غير موجود: {src}")
    if local_dir is None:
        local_dir = (Path("/content/pipeline_data") / src.name) if str(src).startswith("/content/drive") else False
    if local_dir:
        dst = Path(local_dir)
        if dst.resolve() != src.resolve():
            same = ((dst / "manifest.json").exists()
                    and (dst / "manifest.json").read_bytes() == (src / "manifest.json").read_bytes()
                    and all((dst / p.name).exists() and (dst / p.name).stat().st_size == p.stat().st_size
                            for p in src.iterdir() if p.is_file()))
            if not same:
                shutil.rmtree(dst, ignore_errors=True)
                dst.mkdir(parents=True)
                for p in sorted((q for q in src.iterdir() if q.is_file()),
                                key=lambda q: q.name == "manifest.json"):   # manifest أخيراً
                    shutil.copyfile(p, dst / p.name)
                if verbose:
                    print(f"📥 نُسخ {src.name} إلى القرص المحلي {dst}")
            src = dst
    manifest = json.loads((src / "manifest.json").read_text())
    with open(src / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    loaded = {}
    for k in manifest["arrays"]:
        a = np.load(src / f"{k}.npy", mmap_mode='r' if mmap else None)
        if mmap and a.nbytes <= small:
            a = np.array(a)
        loaded[k] = a
    out = {k: (loaded[k] if k in loaded else meta[k]) for k in manifest["key_order"]}
    if verbose:
        lazy = sum(1 for v in out.values() if isinstance(v, np.memmap))
        print(f"✅ تم التحميل من مجلد {src.name}: {len(out)} مفتاحاً ({lazy} مصفوفة memmap على القرص)")
    return out


def _is_disk_backed(data: Any) -> bool:
    return isinstance(data, dict) and any(isinstance(v, np.memmap) for v in data.values())


def _cleanup_scratch_of(data: dict) -> None:
    """يحذف مجلد scratch الذي تسكنه مصفوفات memmap للبناء (``<root>/<بصمة>/merged/*.npy``) — حارس صارم: لا يحذف إلا مجلداً
    اسم أبي الملف فيه ``merged``. تبقى المصفوفات المفتوحة قابلة للقراءة في هذه العملية (لينكس)."""
    dirs = {Path(v.filename).parent.parent for v in data.values()
            if isinstance(v, np.memmap) and getattr(v, 'filename', None) and Path(v.filename).parent.name == 'merged'}
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
        print(f"🧹 حُذف مجلد scratch: {d} (المصفوفات المفتوحة تبقى قابلة للقراءة في هذه الجلسة؛ أعد التحميل من Drive بعد إعادة التشغيل)")


def save_data_to_drive(
    data: Any,
    project_name: Optional[str] = None,
    data_type: str = "preprocessed_data",
    filename_base: str = "preprocessing_output",
    save_latest: bool = True,
    config: Optional[dict] = None,
    fmt: Optional[str] = None,
    single_copy: Optional[bool] = None,
    compresslevel: Optional[int] = None,
) -> Path:
    """يحفظ ``data`` إلى Drive المُركَّب: نسخة مؤرَّخة دائماً، ونسخة
    ``{filename_base}_latest.pkl.gz`` بمسار ثابت إن ``save_latest=True``
    (هذا ما يقرؤه ``load_data_from_drive`` افتراضياً).

    ``fmt``: ``'pkl.gz'`` | ``'npy_dir'`` | ``'auto'`` — ``None`` يقرأ ``config['save_format']`` (افتراضياً ``'auto'``:
    مجلد ``.dataset`` لمجموعة بيانات memmap، وإلا pkl.gz). صيغة المجلد تُكتب دفعات بلا رام؛ ``pkl.gz`` القديمة تبقى تعمل
    (وتُسلسِل memmap تدفقياً، لكن قراءتها لاحقاً تحتاج كل المصفوفات في الرام). ``config['scratch_cleanup_after_save']=True``
    يحذف مجلد scratch بعد نجاح حفظ صيغة المجلد.

    ``single_copy``: ``True`` = نسخة ``_latest`` وحدها (بلا المؤرَّخة) — نصف المساحة ووقت الرفع؛ ``None`` يقرأ
    ``config['save_single_copy']`` (افتراضياً ``False``: نسختان كالسابق). بلا أثر مع ``save_latest=False`` (نسخة مؤرَّخة واحدة أصلاً).

    ``compresslevel``: مستوى gzip لصيغة pkl.gz (1–9؛ 1 أسرع بكثير)؛ ``None`` يقرأ ``config['pkl_compresslevel']`` (افتراضياً 9 =
    افتراضي gzip السابق). بلا أثر على صيغة المجلد (غير مضغوطة).

    ``project_name=None`` يستخدم ``config['project_name']`` (افتراضياً
    ``CONFIG``) بدل اسم عام ثابت — بحيث يُحفَظ فعلياً تحت مشروعك، لا تحت
    ``trading_project`` بغضّ النظر عن إعداداتك.
    """
    config = CONFIG if config is None else config
    project_name = project_name or config.get("project_name", "crypto_model")

    root = mount_drive(config=config)
    base_dir = (root / project_name / data_type) if root is not None else Path(f"./{project_name}/{data_type}")
    base_dir.mkdir(parents=True, exist_ok=True)

    fmt = fmt or config.get("save_format", "auto")
    if fmt == "auto":
        fmt = "npy_dir" if _is_disk_backed(data) else "pkl.gz"
    if fmt not in ("pkl.gz", "npy_dir"):
        raise ValueError(f"fmt غير معروف: {fmt!r} — المتاح: 'pkl.gz' | 'npy_dir' | 'auto'")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    single_copy = bool(config.get("save_single_copy", False) if single_copy is None else single_copy)
    only_latest = single_copy and save_latest          # نسخة latest وحدها؛ save_latest=False يكتب المؤرَّخة وحدها أصلاً
    level = int(config.get("pkl_compresslevel", 9) if compresslevel is None else compresslevel)
    if not 0 <= level <= 9:
        raise ValueError(f"compresslevel يجب أن يكون بين 0 و9، لا {level!r}")

    if fmt == "npy_dir":
        if not isinstance(data, dict):
            raise TypeError("صيغة npy_dir تتطلّب قاموس مجموعة بيانات")
        if only_latest:
            versioned_path = save_dataset_dir(data, base_dir / f"{filename_base}_latest{DATASET_DIR_SUFFIX}")
            print(f"✅ تم الحفظ (مجلد، نسخة واحدة): {versioned_path.name}")
        else:
            versioned_path = save_dataset_dir(data, base_dir / f"{filename_base}_{timestamp}{DATASET_DIR_SUFFIX}")
            if save_latest:
                save_dataset_dir(data, base_dir / f"{filename_base}_latest{DATASET_DIR_SUFFIX}")
            print(f"✅ تم الحفظ (مجلد): {versioned_path.name}" + (" — ونسخة latest بنفس المجلد" if save_latest else ""))
        if config.get("scratch_cleanup_after_save"):
            _cleanup_scratch_of(data)
        return versioned_path

    # pkl.gz: memmap تُسلسَل كمصفوفات عادية (عرض بلا نسخة) — التدفّق إلى gzip لا يحتاج نسخة رام كاملة.
    if isinstance(data, dict):
        data = {k: (np.asarray(v) if isinstance(v, np.memmap) else v) for k, v in data.items()}
    if only_latest:
        versioned_path = base_dir / f"{filename_base}_latest.pkl.gz"
        with gzip.open(versioned_path, "wb", compresslevel=level) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✅ تم الحفظ (نسخة واحدة): {versioned_path.name} ({versioned_path.stat().st_size / (1024 * 1024):.2f} MB)")
        return versioned_path

    versioned_path = base_dir / f"{filename_base}_{timestamp}.pkl.gz"
    with gzip.open(versioned_path, "wb", compresslevel=level) as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    if save_latest:
        latest_path = base_dir / f"{filename_base}_latest.pkl.gz"
        with gzip.open(latest_path, "wb", compresslevel=level) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    v_size = versioned_path.stat().st_size / (1024 * 1024)
    print(f"✅ تم الحفظ: {versioned_path.name} ({v_size:.2f} MB)"
          + (f" — ونسخة latest بنفس المجلد" if save_latest else ""))
    return versioned_path


def load_data_from_drive(
    project_name: Optional[str] = None,
    data_type: str = "preprocessed_data",
    filename_base: str = "preprocessing_output",
    filename: Optional[str] = None,
    config: Optional[dict] = None,
    fmt: str = "auto",
    mmap: bool = True,
    local_dir=None,
) -> Any:
    """يقرأ ما كتبته ``save_data_to_drive`` من Drive المُركَّب — بلا
    ``gdown`` ولا رابط عام، بنفس مسار الحفظ بالضبط.

    ``filename=None`` (الافتراضي) يقرأ ``{filename_base}_latest`` بإحدى الصيغتين: ``.dataset`` (مجلد، :func:`load_dataset_dir`:
    memmap كسول — معاملا ``mmap``/``local_dir``) أو ``.pkl.gz`` (القديمة، كل شيء في الرام). ``fmt='auto'`` يأخذ الأحدث إن
    وُجدتا، و``'pkl.gz'``/``'npy_dir'`` يفرض إحداهما.
    مرّر ``filename`` صراحةً (مثلاً اسم نسخة مؤرَّخة بعينها من مخرَج
    ``save_data_to_drive`` السابق؛ مجلداً أو ملفاً) لتحميل نسخة محدَّدة بدل الأحدث.
    """
    config = CONFIG if config is None else config
    project_name = project_name or config.get("project_name", "crypto_model")

    root = mount_drive(config=config)
    base_dir = (root / project_name / data_type) if root is not None else Path(f"./{project_name}/{data_type}")
    if filename:
        path = base_dir / filename
    else:
        pkl = base_dir / f"{filename_base}_latest.pkl.gz"
        dpath = base_dir / f"{filename_base}_latest{DATASET_DIR_SUFFIX}"
        d_ok = (dpath / "manifest.json").exists() and fmt in ("auto", "npy_dir")
        p_ok = pkl.exists() and fmt in ("auto", "pkl.gz")
        if d_ok and p_ok:
            d_ok = (dpath / "manifest.json").stat().st_mtime >= pkl.stat().st_mtime
        path = dpath if d_ok else pkl
    if path.is_dir():
        return load_dataset_dir(path, mmap=mmap, local_dir=local_dir)

    if not path.exists():
        raise FileNotFoundError(
            f"❌ لا يوجد ملف محفوظ في {path} — شغّل save_data_to_drive أولاً، "
            f"أو مرّر filename لاسم نسخة مؤرَّخة موجودة فعلاً في {base_dir}."
        )

    with gzip.open(path, "rb") as f:
        data = pickle.load(f)

    print(f"✅ تم التحميل من {path.name} — نوع: {type(data)}"
          + (f"، المفاتيح: {len(data)}" if isinstance(data, dict) else ""))
    return data


def load_preprocessed_data_from_drive(
    file_id: str,
    output_filename: str = 'preprocessing_output.pkl.gz',
    download_dir: str = '.',
    quiet: bool = False,
    cleanup: bool = False
) -> Any:
    """[توافق خلفي فقط] تحميل ملف عام مُشارَك عبر رابط Drive (``gdown``).

    لا تستخدمها لقراءة مخرَج ``save_data_to_drive`` — استخدم
    ``load_data_from_drive`` بدلاً منها؛ هذه الدالة تحتاج ``file_id``
    لملف مُشارَك بامتياز "Anyone with the link"، وهو أسلوب مختلف تماماً
    عن Drive المُركَّب."""
    import gdown

    output_path = os.path.join(download_dir, output_filename)
    url = f'https://drive.google.com/uc?id={file_id}'

    try:
        print(f"🔄 جاري التحميل من Google Drive...")
        gdown.download(url, output_path, quiet=quiet)

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"فشل تحميل الملف")

        with gzip.open(output_path, 'rb') as f:
            data = pickle.load(f)

        print(f"✅ تم التحميل! نوع: {type(data)}")
        if isinstance(data, dict):
            print(f"🔑 المفاتيح: {len(data)}")

        return data

    finally:
        if cleanup and os.path.exists(output_path):
            os.remove(output_path)
