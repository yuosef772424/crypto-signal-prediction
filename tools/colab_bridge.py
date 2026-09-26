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
