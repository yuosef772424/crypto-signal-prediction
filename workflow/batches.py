"""
PURPOSE:  tf.data batch builders for training/validation: full index shuffle per epoch, memmap-friendly lazy gathering, float16 to float32 at batch time, +1/-1 to {0,1} class labels.
TAGS:     make_shuffled_dataset, make_eval_dataset, _y_for, _to_unit_label, memmap, float16, tf.data, batches
PITFALLS: _y_for reads main_config from the notebook namespace. X may be a dict {timeframe: array} (multi-timeframe). Do not copy a memmap X into RAM (tests/test_disk_backed.py). Executed into the notebook's shared namespace by workflow/_loader.py (never imported on its own): names from other modules and the %run notebooks resolve at call time. Extracted verbatim from main.ipynb cell 17 (section 5).
"""
import numpy as np

import tensorflow as tf


def _to_unit_label(y):
    """يحوّل ترميز اتجاه خط الأنابيب إلى {0,1} القياسية التي يفترضها
    binary_classification (sigmoid) وخسارة classification في trainer_framework
    (binary_crossentropy). يقبل الترميزين +1.0/-1.0 و1.0/0.0 (الموجب = صعود)،
    فلا يتلف التدريب إن تغيّر ترميز خط الأنابيب. لا تُطبَّق على أهداف الانحدار."""
    return (np.asarray(y) > 0).astype("float32")


def _y_for(split):
    y = {}
    for cfg in main_config["targets"].values():
        v = split["y"][cfg["true_key"]]
        y[cfg["true_key"]] = _to_unit_label(v) if cfg["task_type"] == "classification" else v
    return y


def make_shuffled_dataset(X, y_dict, batch_size, seed=None, shuffle=True):
    """خلط **كامل** كل حقبة عبر فهارس لا عبر البيانات نفسها.

    split_data يُخرج العيّنات مرتّبة عملةً عملة ثم زمنياً، و`shuffle(4096)` السابق يخلط محلياً فقط:
    قيس على 300 عملة × 1060 عيّنة — ~13 عملة فقط في كل دفعة، والحقبة تمرّ على العملات بالترتيب (أول
    10% من الحقبة ≈ العملة رقم 15، آخر 10% ≈ رقم 284)، فينتهي كل تحقق بعد تدريب شبه حصري على آخر
    عشرات العملات. خلط الفهارس (أعداد صحيحة، بضعة ميجابايت) يعطي ~58 عملة لكل دفعة بتوزيع منتظم،
    دون نسخ X (~1.8 جيجا) داخل مخزن الخلط."""
    x_keys = list(X) if isinstance(X, dict) else None      # X قاموس {فريم: مصفوفة} لنموذج متعدّد الفريمات
    x_arrays = [X[k] for k in x_keys] if x_keys else [X]
    n, keys = len(x_arrays[0]), list(y_dict)
    arrays = x_arrays + [y_dict[k] for k in keys]
    n_x = len(x_arrays)

    def _gather(idx):
        idx = np.sort(idx)                     # ترتيب داخل الدفعة لا يغيّر التدرّج ويُسرّع القراءة
        # float32 عند تكوين الدفعة: X قد تُخزَّن float16 (x_storage_dtype) لتوفير الذاكرة
        return tuple(np.asarray(a[idx], dtype="float32") for a in arrays)

    def _map(idx):
        outs = tf.numpy_function(_gather, [idx], [tf.float32] * len(arrays))
        outs = [tf.ensure_shape(o, (batch_size,) + tuple(a.shape[1:])) for o, a in zip(outs, arrays)]
        return (dict(zip(x_keys, outs[:n_x])) if x_keys else outs[0]), dict(zip(keys, outs[n_x:]))

    ds = tf.data.Dataset.range(n)
    if shuffle:
        ds = ds.shuffle(n, seed=seed, reshuffle_each_iteration=True)
    return (ds.batch(batch_size, drop_remainder=True)
            .map(_map, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))


def _to_float32_inputs(x, y):
    """X المخزَّنة float16 ترفع إلى float32 لكل دفعة (لا نسخة float32 كاملة من val)؛ float32 تبقى كما هي."""
    return tf.nest.map_structure(lambda a: tf.cast(a, tf.float32), x), y


def make_eval_dataset(X, y_dict, batch_size):
    """دفعات بالترتيب (val). X عادية: from_tensor_slices كما كانت. X memmap (بيانات disk_backed): from_tensor_slices كانت ستنسخ
    القسم كله إلى الرام، فتُجمَع كل دفعة كسولاً من القرص بنفس آلية make_shuffled_dataset (بلا خلط) ← نفس الدفعات والقيم."""
    arrays = list(X.values()) if isinstance(X, dict) else [X]
    if any(isinstance(a, np.memmap) for a in arrays):
        return make_shuffled_dataset(X, y_dict, batch_size, shuffle=False)
    return (tf.data.Dataset.from_tensor_slices((X, y_dict))
            .batch(batch_size, drop_remainder=True)
            .map(_to_float32_inputs)
            .prefetch(tf.data.AUTOTUNE))
