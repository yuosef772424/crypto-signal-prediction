"""طبقة بيانات اللوحة: تجميع عيّنات split_data حسب يوم UTC، ودفعات من عدّة أيام كاملة.

لا يُبنى موتّر أيام×عملات×نافذة×ميزات (جيجابايتات، ومعظمه حشو لأن عدد العملات يتغيّر من 1 إلى ~500 يومياً). بدلاً من
ذلك تُحفظ لكل يوم قائمة فهارس عيّناته في X الأصلية، وتُجمَع النوافذ بالفهارس وقت الدفعة فقط.

شكل الدفعة (مسطّح، بلا حشو في المُرمِّز):
    x   : (M, T, F)  نوافذ كل عيّنات أيام الدفعة
    day : (M,)       رقم اليوم داخل الدفعة 0..B-1
    pos : (M,)       موضع العملة داخل يومها 0..n_d-1
النموذج يبعثر (day, pos) إلى (B, N_max, d) مع قناع مفاتيح للانتباه فقط، ثم يجمعها ثانيةً — فالمُرمِّز (وفيه BatchNorm)
لا يرى أي حشو أبداً.
"""
import numpy as np
import pandas as pd

TARGETS = ("high", "low", "close")
DAY_NS = 86_400 * 10**9
# LAST_COLUMNS في خط الأنابيب: ['last_high', 'last_low', 'last_close', 'timestamp', 'future_close',
#                               'future_low_min', 'future_high_max']
LC = {"last_high": 0, "last_low": 1, "last_close": 2, "timestamp": 3,
      "future_close": 4, "future_low_min": 5, "future_high_max": 6}


class PanelSplit:
    """قسم واحد (train أو val أو test) مجمَّعاً حسب اليوم.

    X            : مصفوفة النوافذ (N, T, F) — مرجع لا نسخة.
    y            : قاموس خط الأنابيب {y_{هدف}_class: 1/0 (أو ±1 القديم/retarget_splits), y_{هدف}_reg: عائد}.
                   الصعود = y > 0، فالترميزان يُقرآن بلا تحويل.
    last_candles : (N, 7) بأعمدة LAST_COLUMNS.
    assets       : أسماء العملات لكل صف (اختياري؛ لازم لفحص التكرار وللتصدير).
    """

    def __init__(self, X, y, last_candles, assets=None, name="", targets=TARGETS, day_ns=DAY_NS):
        self.name, self.targets, self.X = name, tuple(targets), X
        self.n = len(X)
        self.lc = np.asarray(last_candles, dtype="float64")
        if len(self.lc) != self.n:
            raise ValueError(f"{name}: last_candles {len(self.lc)} ≠ X {self.n}")
        self.assets = None if assets is None else np.asarray(assets, dtype=object)
        if self.assets is not None and len(self.assets) != self.n:
            raise ValueError(f"{name}: assets {len(self.assets)} ≠ X {self.n}")
        self.ts = self.lc[:, LC["timestamp"]].astype("int64")
        self.day_key = (self.ts // day_ns) * day_ns          # بداية يوم UTC
        self.misaligned = self.ts != self.day_key            # طابع ليس 00:00 UTC
        self.ycls = np.stack([(np.asarray(y[f"y_{t}_class"]).ravel() > 0) for t in self.targets], 1).astype("float32")
        self.yreg = np.stack([np.asarray(y[f"y_{t}_reg"]).ravel() for t in self.targets], 1).astype("float32")

        order = np.argsort(self.day_key, kind="stable")      # داخل اليوم: ترتيب الصفوف الأصلي
        k = self.day_key[order]
        starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]]) if self.n else np.zeros(0, int)
        self.order = order
        self.bounds = np.r_[starts, self.n].astype("int64")
        self.days = k[starts]
        self.sizes = np.diff(self.bounds)
        # هدف حدّ الارتباط اليومي: رتبة عائد close النسبي داخل يومه، متمركزة في [-0.5, 0.5] (Pearson معها ≈ Spearman،
        # فلا تحكم قفزة +300% واحدة الحدّ). تُحسب هنا مرّة واحدة — هي تسمية لا تحتاج تدرّجاً.
        ci = self.targets.index("close") if "close" in self.targets else len(self.targets) - 1
        self.yrank = np.zeros(self.n, dtype="float32")
        rk = pd.Series(self.yreg[order, ci]).groupby(np.repeat(np.arange(len(self.sizes)), self.sizes)).rank(pct=True)
        n_rep = np.repeat(self.sizes, self.sizes).astype("float64")
        self.yrank[order] = (rk.to_numpy() - 0.5 / np.maximum(n_rep, 1.0) - 0.5).astype("float32")

    def take(self, idx, name=None):
        """قسم جديد من صفوف idx فقط (ينسخ X[idx] — للمجموعات الفرعية الصغيرة/الاختبار)."""
        idx = np.sort(np.asarray(idx))
        y = {}
        for i, t in enumerate(self.targets):
            y[f"y_{t}_class"] = self.ycls[idx, i].copy()       # 1/0 — ترميز خط الأنابيب الحالي
            y[f"y_{t}_reg"] = self.yreg[idx, i]
        return PanelSplit(np.asarray(self.X[idx]), y, self.lc[idx],
                          None if self.assets is None else self.assets[idx], name or self.name, self.targets)

    def subset(self, last_days=None, coins=None, seed=0):
        """مجموعة فرعية للاختبار السريع: آخر last_days يوماً متتالياً، و coins عملة على الأكثر (نفس العملات عبر
        الأقسام إن عُرفت الأسماء — مختارة بتجزئة ثابتة للاسم، وإلا عيّنة عشوائية من كل يوم)."""
        keep = np.ones(self.n, bool)
        if last_days:
            keep &= self.day_key >= self.days[max(self.n_days - int(last_days), 0)]
        if coins:
            if self.assets is not None:
                import zlib
                h = np.array([zlib.crc32(f"{seed}:{a}".encode()) for a in self.assets], dtype="int64")
                uniq = np.unique(h[keep])
                keep &= h <= np.sort(uniq)[min(int(coins), len(uniq)) - 1]
            else:
                rng, sel = np.random.default_rng(seed), np.zeros(self.n, bool)
                for d in range(self.n_days):
                    ii = self.day_indices(d)
                    ii = ii[keep[ii]]
                    sel[rng.choice(ii, min(len(ii), int(coins)), replace=False)] = True
                keep &= sel
        return self.take(np.flatnonzero(keep))

    # ── الاستعلام ──
    @property
    def n_days(self):
        return len(self.sizes)

    def day_indices(self, d):
        """فهارس عيّنات اليوم رقم d (في X الأصلية)."""
        return self.order[self.bounds[d]:self.bounds[d + 1]]

    # ── الدفعات ──
    def batch_plan(self, batch_samples=1024, max_days=16, shuffle=False, seed=0, epoch=0):
        """قائمة دفعات، كل دفعة قائمة أرقام أيام. تُملأ الدفعة بأيام كاملة حتى batch_samples عيّنة أو max_days يوماً
        (يوم أكبر من batch_samples وحده يبقى دفعة كاملة — لا يُقسَم اليوم أبداً). shuffle: ترتيب أيام حتمي من
        (seed, epoch)؛ بلا shuffle: ترتيب التاريخ (val/test)."""
        days = (np.random.default_rng([int(seed), int(epoch)]).permutation(self.n_days) if shuffle
                else np.arange(self.n_days))
        plan, cur, cur_n = [], [], 0
        for d in days:
            s = int(self.sizes[d])
            if cur and (cur_n + s > batch_samples or len(cur) >= max_days):
                plan.append(cur)
                cur, cur_n = [], 0
            cur.append(int(d))
            cur_n += s
        if cur:
            plan.append(cur)
        return plan

    def make_batch(self, day_list, max_coins=None, rng=None):
        """يجمع أيام الدفعة. max_coins (تدريب فقط، اختياري): عيّنة عشوائية من عملات الأيام الأكبر منه — يكسر
        «كل عيّنة مرّة واحدة» عمداً، لذا الافتراضي None."""
        idx, day, pos = [], [], []
        for b, d in enumerate(day_list):
            ii = self.day_indices(d)
            if max_coins and len(ii) > max_coins:
                ii = np.sort((rng or np.random.default_rng(0)).choice(ii, max_coins, replace=False))
            idx.append(ii)
            day.append(np.full(len(ii), b, dtype="int32"))
            pos.append(np.arange(len(ii), dtype="int32"))
        idx = np.concatenate(idx)
        srt = np.argsort(idx, kind="stable")      # قراءة X بترتيب الذاكرة أسرع؛ الترتيب داخل الدفعة لا يغيّر شيئاً
        idx, day, pos = idx[srt], np.concatenate(day)[srt], np.concatenate(pos)[srt]
        return {"idx": idx, "x": np.asarray(self.X[idx], dtype="float32"), "day": day, "pos": pos,
                "ycls": self.ycls[idx], "yreg": self.yreg[idx], "yrank": self.yrank[idx]}

    def iter_batches(self, batch_samples=1024, max_days=16, shuffle=False, seed=0, epoch=0, max_coins=None):
        rng = np.random.default_rng([int(seed), int(epoch), 1])
        for dl in self.batch_plan(batch_samples, max_days, shuffle, seed, epoch):
            yield self.make_batch(dl, max_coins=max_coins if shuffle else None, rng=rng)

    # ── فحوص السلامة ──
    def check(self, min_group=5):
        """يُرجع قاموس فحوص؛ ok=False إن فشل فحص جوهري (تغطية، نقاء اليوم، تكرار عملة في يوم)."""
        covered = np.sort(np.concatenate([self.day_indices(d) for d in range(self.n_days)])) if self.n else np.zeros(0)
        coverage_ok = bool(np.array_equal(covered, np.arange(self.n)))
        pure = all(len(np.unique(self.day_key[self.day_indices(d)])) == 1 for d in range(self.n_days))
        mis_assets = []
        if self.misaligned.any() and self.assets is not None:
            mis_assets = sorted(set(self.assets[self.misaligned]))
        dups = 0
        if self.assets is not None:
            dups = int(pd.DataFrame({"a": self.assets, "d": self.day_key}).duplicated().sum())
        small = self.sizes < min_group
        out = {
            "split": self.name, "samples": int(self.n), "days": int(self.n_days),
            "first_day": str(pd.Timestamp(int(self.days[0])))[:10] if self.n else "",
            "last_day": str(pd.Timestamp(int(self.days[-1])))[:10] if self.n else "",
            "coins_per_day_min": int(self.sizes.min()) if self.n else 0,
            "coins_per_day_median": float(np.median(self.sizes)) if self.n else 0,
            "coins_per_day_max": int(self.sizes.max()) if self.n else 0,
            "each_sample_once": coverage_ok, "day_groups_pure": bool(pure),
            "misaligned_timestamps": int(self.misaligned.sum()), "misaligned_assets": mis_assets[:20],
            "duplicate_asset_day": dups,
            f"days_lt_{min_group}_coins": int(small.sum()),
            f"samples_in_days_lt_{min_group}": int(self.sizes[small].sum()),
        }
        out["ok"] = coverage_ok and pure and dups == 0
        return out


def _concat_dict(split_dict, model_tf):
    names = list(split_dict)
    parts = [split_dict[a] for a in names]
    X = np.concatenate([p[f"X_{model_tf}"] for p in parts])
    lc = np.concatenate([np.asarray(p["last_candles"]) for p in parts])
    y = {k: np.concatenate([np.asarray(p["y"][k]).ravel() for p in parts]) for k in parts[0]["y"]}
    assets = np.concatenate([np.array([a] * len(p["last_candles"]), dtype=object) for a, p in zip(names, parts)])
    return X, y, lc, assets


def panel_split_from(split_or_dict, model_tf, asset_names=None, name="", targets=TARGETS):
    """من مخرَج split_data/retarget_splits: قسم مدمج (train/val) أو قاموس {عملة: قسم} (test).
    asset_names للقسم المدمج: split_asset_names(dataset, 'train'|'val') من دفتر main (بنفس أقنعة split_data)."""
    if "y" in split_or_dict:
        X, y, lc = split_or_dict[f"X_{model_tf}"], split_or_dict["y"], split_or_dict["last_candles"]
        assets = asset_names if asset_names is not None and len(asset_names) == len(lc) else None
    else:
        X, y, lc, assets = _concat_dict(split_or_dict, model_tf)
    return PanelSplit(X, y, lc, assets=assets, name=name, targets=targets)


def format_checks(checks):
    """جدول نصي لفحوص عدّة أقسام."""
    df = pd.DataFrame(checks).set_index("split").T
    return df.to_string()
