"""جسر Colab ↔ GitHub ↔ Claude (PR #7).

الفكرة: البيانات تُنسخ من Drive إلى قرص Colab المحلي **مرّة واحدة لكل جلسة** (Drive بطيء في القراءة المتكرّرة)،
والكود يُسحب من GitHub عند كل تجربة (git pull، ثوانٍ). Claude يرفع الكود وخطط التجارب إلى GitHub، وأنت تشغّل
خلية واحدة وتنسخ الملخّص المطبوع.

    import sys; sys.path.insert(0, "/content/csp/tools")
    from colab_bridge import prepare_data
    DATA_DIR = prepare_data()            # يبحث عن history_1d في Drive وينسخه إلى /content/am_data/history_1d
    DATA_DIR = prepare_data("/content/drive/MyDrive/crypto/history_1d")   # أو مساراً صريحاً
"""
import glob
import os
import shutil
import time

LOCAL_ROOT = "/content/am_data"
DRIVE_ROOT = "/content/drive"
MARKER = ".copied_from"


def _looks_like_history(d):
    return os.path.isdir(d) and os.path.exists(os.path.join(d, "BTCUSDT.csv"))


def find_history_dir(name="history_1d", roots=None):
    """يبحث عن مجلد باسم name فيه BTCUSDT.csv تحت MyDrive/Shareddrives/اختصارات «المشارَك معي»."""
    roots = roots or [os.path.join(DRIVE_ROOT, "MyDrive"), os.path.join(DRIVE_ROOT, "Shareddrives"),
                      os.path.join(DRIVE_ROOT, ".shortcut-targets-by-id")]
    found = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for depth in range(0, 5):   # بحث بالعمق المحدود بدل os.walk على Drive كاملاً (بطيء جداً)
            pattern = os.path.join(root, *(["*"] * depth), name)
            found += [d for d in glob.glob(pattern) if _looks_like_history(d)]
        if found:
            break
    return sorted(set(found), key=len)


def prepare_data(history_dir=None, local_root=LOCAL_ROOT, mount=True, force=False):
    """ينسخ ملفات CSV اليومية من Drive إلى القرص المحلي مرّة واحدة ويُرجع المسار المحلي.

    history_dir: مسار صريح على Drive. None = بحث تلقائي عن مجلد history_1d.
    force: إعادة النسخ حتى لو نُسخ من قبل في هذه الجلسة (بعد تحديث بياناتك على Drive)."""
    if mount and not os.path.isdir(os.path.join(DRIVE_ROOT, "MyDrive")):
        from google.colab import drive
        drive.mount(DRIVE_ROOT)
    local = os.path.join(local_root, "history_1d")
    if not force and os.path.exists(os.path.join(local, MARKER)):
        n = len(glob.glob(os.path.join(local, "*.csv")))
        print(f"✅ البيانات منسوخة مسبقاً في هذه الجلسة: {local} ({n} ملف) — لا قراءة من Drive")
        return local
    if history_dir is None:
        hits = find_history_dir()
        if not hits:
            raise FileNotFoundError(
                "لم أجد مجلد history_1d فيه BTCUSDT.csv. المجلد مُشارَك معك من حساب آخر: افتح Drive ← "
                "«مُشارَك معي» ← history_1d ← نقر يمين ← «إضافة اختصار إلى Drive» ← «ملفاتي»، ثم أعد التشغيل، "
                "أو مرّر المسار صراحةً: prepare_data('/content/drive/MyDrive/.../history_1d')")
        history_dir = hits[0]
        if len(hits) > 1:
            print("ℹ️ وُجد أكثر من مجلد، استُخدم الأول:", hits)
    if not _looks_like_history(history_dir):
        raise FileNotFoundError(f"{history_dir} لا يحوي BTCUSDT.csv")
    t0 = time.time()
    os.makedirs(local, exist_ok=True)
    files = sorted(glob.glob(os.path.join(history_dir, "*.csv")))
    for i, f in enumerate(files, 1):
        shutil.copy2(f, os.path.join(local, os.path.basename(f)))
        if i % 100 == 0:
            print(f"   نُسخ {i}/{len(files)}…")
    with open(os.path.join(local, MARKER), "w") as fh:
        fh.write(history_dir)
    print(f"✅ نُسخ {len(files)} ملفاً من {history_dir} إلى {local} في {time.time() - t0:.0f} ثانية")
    return local


def gpu_summary():
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        return f"GPU: {len(gpus)} ({', '.join(g.name for g in gpus)})" if gpus else "GPU: لا يوجد (CPU فقط — أبطأ بكثير)"
    except Exception as e:  # noqa: BLE001
        return f"GPU: تعذّر الفحص ({e})"


# ─────────────────────────── سجلّ الملفات: تحميل بلا تركيب Drive ───────────────────────────
# سجلّ = CSV بعمودين file_id,name (نفس صيغة files_info.csv / CONFIG['asset_registry_file_id'] في خط الأنابيب).
# مع سجلّ وملفات مُشارَكة «لأيّ شخص لديه الرابط» يمكن تحميل البيانات من أي جهاز (Colab بحساب آخر، جلسة Claude،
# جهازك) عبر رابط التحميل المباشر — بلا drive.mount وبلا حساب Google.

DOWNLOAD_URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"


def build_drive_registry(folder, out_csv=None, pattern="*.csv"):
    """ينشئ سجلّ file_id,name لكل ملف في مجلد Drive مُركَّب (Colab).

    المعرّف يُقرأ من الخاصية الممتدة user.drive.id التي يضعها Colab على كل ملف في /content/drive — بلا
    تسجيل دخول ولا Drive API. الاسم = اسم الملف بلا الامتداد (BTCUSDT.csv ← BTCUSDT).
    out_csv=None: يُكتب files_info.csv داخل نفس المجلد (فيُشارَك معه). يُرجع DataFrame."""
    import pandas as pd
    rows, missing = [], []
    for p in sorted(glob.glob(os.path.join(folder, pattern))):
        name = os.path.splitext(os.path.basename(p))[0]
        if name == "files_info":
            continue
        try:
            rows.append({"file_id": os.getxattr(p, "user.drive.id").decode(), "name": name})
        except (OSError, AttributeError):
            missing.append(name)
    if missing:
        raise RuntimeError(f"تعذّر قراءة معرّف Drive لـ {len(missing)} ملفاً (مثلاً {missing[:3]}) — هذه الدالة تعمل على "
                           "مجلد داخل /content/drive بعد drive.mount في Colab")
    df = pd.DataFrame(rows, columns=["file_id", "name"])
    out_csv = out_csv or os.path.join(folder, "files_info.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ سجلّ {len(df)} ملفاً ← {out_csv}\n   شارك المجلد (أو الملفات) «أيّ شخص لديه الرابط — عارض» ليعمل التحميل بلا Drive.")
    return df


def _fetch(file_id, url_template=DOWNLOAD_URL, timeout=60):
    import urllib.request
    req = urllib.request.Request(url_template.format(id=file_id), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data, ctype = r.read(), r.headers.get("Content-Type", "")
    if "text/html" in ctype or data[:15].lstrip().lower().startswith((b"<!doctype", b"<html")):
        raise PermissionError(f"{file_id}: Drive أعاد صفحة HTML لا ملفاً — الملف غير مُشارَك «لأيّ شخص لديه الرابط»")
    return data


def read_registry(registry, url_template=DOWNLOAD_URL):
    """registry: مسار CSV محلي، أو رابط http(s)، أو معرّف Drive لملف السجلّ نفسه، أو قائمة قواميس
    [{'file_id': ..., 'name': ...}] (= df.to_dict('records') كما في خط الأنابيب)، أو DataFrame."""
    import io as _io
    import pandas as pd
    if isinstance(registry, pd.DataFrame):
        df = registry.copy()
    elif isinstance(registry, (list, tuple)):
        df = pd.DataFrame(list(registry))
    elif os.path.exists(str(registry)):
        df = pd.read_csv(registry, encoding="utf-8-sig")
    elif str(registry).startswith("http"):
        df = pd.read_csv(registry, encoding="utf-8-sig")
    else:
        df = pd.read_csv(_io.BytesIO(_fetch(registry, url_template)), encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    if not {"file_id", "name"} <= set(df.columns):
        raise ValueError(f"السجلّ يجب أن يحوي file_id,name — وُجد {list(df.columns)}")
    return df


def download_from_registry(registry, dest="/content/am_data/history_1d", names=None, workers=8, force=False,
                           url_template=DOWNLOAD_URL):
    """يحمّل كل ملفات السجلّ إلى dest/<name>.csv بالتوازي، مع كاش: الموجود لا يُعاد تحميله (force=True للتحديث).

    names: قائمة أسماء لتحميل جزء فقط (مثلاً ['BTCUSDT', 'ETHUSDT']). يُرجع dest."""
    from concurrent.futures import ThreadPoolExecutor
    reg = read_registry(registry, url_template)
    if names is not None:
        reg = reg[reg["name"].isin(set(names))]
    os.makedirs(dest, exist_ok=True)
    todo = [(r.file_id, os.path.join(dest, f"{r.name}.csv")) for r in reg.itertuples()
            if force or not os.path.exists(os.path.join(dest, f"{r.name}.csv"))]
    t0, errors = time.time(), []

    def one(item):
        fid, path = item
        try:
            data = _fetch(fid, url_template)
            with open(path + ".part", "wb") as f:
                f.write(data)
            os.replace(path + ".part", path)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{os.path.basename(path)}: {e}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, _ in enumerate(ex.map(one, todo), 1):
            if i % 100 == 0:
                print(f"   حُمّل {i}/{len(todo)}…")
    print(f"✅ {len(reg) - len(todo)} موجود مسبقاً + {len(todo) - len(errors)} حُمّل الآن ← {dest} "
          f"({time.time() - t0:.0f} ثانية)")
    if errors:
        print(f"⚠️ فشل {len(errors)}: {errors[:3]}")
    with open(os.path.join(dest, MARKER), "w") as fh:
        fh.write(f"registry:{registry if isinstance(registry, str) else 'records'}")
    return dest
