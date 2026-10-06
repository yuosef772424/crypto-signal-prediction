"""
PURPOSE:  Helper for locating the pre-built dataset file on a mounted Google Drive (shared folders / shortcuts); used by main's section-3 load cell.
TAGS:     dataset file search, Drive shortcut, shared folder, _find_on_drive, load data
PITFALLS: Depth-limited glob under /content/drive (Drive is slow). Reads os from main's namespace (imported by the setup cell). Executed into the notebook's shared namespace by workflow/_loader.py (never imported on its own): names from other modules and the %run notebooks resolve at call time. Extracted verbatim from main.ipynb cell 7 (section 3).
"""
import os


def _find_on_drive(name, max_depth=5):
    """يبحث عن ملف باسم name تحت MyDrive واختصارات «المُشارَك معي» والمجلدات المشتركة (بعمق محدود — Drive بطيء)."""
    import glob
    for root in ("/content/drive/MyDrive", "/content/drive/.shortcut-targets-by-id", "/content/drive/Shareddrives"):
        for depth in range(max_depth):
            hits = sorted(glob.glob(os.path.join(root, *(["*"] * depth), name)), key=len)
            if hits:
                return hits[0]
    return None
