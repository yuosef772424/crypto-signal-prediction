"""
PURPOSE:  Imports and environment: IN_COLAB, mount_drive_if_needed, opt_variables, atomic_write_json.
TAGS:     imports, IN_COLAB, mount_drive_if_needed, opt_variables, atomic_write_json, colab, keras 2/3
PITFALLS: Prints the TensorFlow version at load. Works on Keras 2 (tf-keras) and Keras 3; opt_variables hides the API
          difference. Executed into the one shared trainer namespace by trainer/_loader.py (never imported on its
          own): names from other modules resolve at call time.

## 1) الاستيرادات وإعداد البيئة
"""
# @title 1) الاستيرادات وإعداد البيئة
import os
import json
import copy
import time
import shutil
import hashlib
import tempfile
import collections.abc
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Callable, Any

try:
    from google.colab import drive as _colab_drive
    IN_COLAB = True
except Exception:
    IN_COLAB = False

print(f"TensorFlow: {tf.__version__} | Colab: {IN_COLAB}")
# ملاحظة: يعمل الإطار على Keras 2 (tf-keras) وKeras 3 معًا — اختُبر على النسختين.
# يتطلب TensorFlow 2.11+ (لتوفّر tf.keras.optimizers.AdamW وواجهة tf.train.Checkpoint الحديثة).


def mount_drive_if_needed(path: str):
    """يُركِّب Google Drive تلقائيًا إن كان أي مسار مطلوب يقع تحت /content/drive"""
    if path and str(path).startswith("/content/drive") and IN_COLAB:
        try:
            _colab_drive.mount("/content/drive", force_remount=False)
        except Exception as e:
            print(f"⚠️ تعذر تركيب Drive تلقائيًا: {e}")


def opt_variables(optimizer) -> list:
    """متغيرات الـ optimizer بغضّ النظر عن النسخة: في Keras 2 هي دالة، وفي Keras 3 خاصية (list)."""
    v = optimizer.variables
    return list(v() if callable(v) else v)


def atomic_write_json(path: str, obj: Any):
    """كتابة JSON ذرّية: نكتب لملف مؤقت ثم نستبدل — انقطاع Colab أثناء الكتابة لا يترك ملفًا مقطوعًا."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
