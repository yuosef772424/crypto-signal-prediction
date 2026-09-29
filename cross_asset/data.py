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
    target_scale : reg_target_scale الذي بُنيت به y_{هدف}_reg (عائد × المقياس). يُقسَم عليه قبل أي تحويل إلى سعر
                   أو عائد (asym_score، export_signals)؛ 1.0 لبيانات قديمة بلا المفتاح. رقم واحد، أو مصفوفة (K,) بترتيب
                   targets حين يختلف مقياس الأهداف (entry_range: close موقع [0,1] بلا مقياس).
    target_mode  : وضع الهدف المختوم على القسم (retarget_splits في main)؛ None = أهداف خط الأنابيب. يحدّد عكس mu إلى
                   سعر في export_signals وasym_score.
    entry_close_reg : مع target_mode="entry_range": تعريف انحدار close ("abs_return" | "range_pos")، مختوم كذلك؛ None
                   خارجه. يحدّد وحدة mu_close وطريقة عكسه (entry_range_to_prices).
    """

    def __init__(self, X, y, last_candles, assets=None, name="", targets=TARGETS, day_ns=DAY_NS, target_scale=1.0,
                 target_mode=None, entry_close_reg=None):
        self.name, self.targets, self.X = name, tuple(targets), X
        self.target_scale = (np.asarray(target_scale, dtype="float64") if np.ndim(target_scale)
                             else float(target_scale))
        self.target_mode = target_mode
        self.entry_close_reg = entry_close_reg
        self.day_ns = int(day_ns)
        self.n = len(X)
        self.lc = np.asarray(last_candles, dtype="float64")
        if len(self.lc) != self.n:
            raise ValueError(f"{name}: last_candles {len(self.lc)} ≠ X {self.n}")
        self.assets = None if assets is None else np.asarray(assets, dtype=object)
        if self.assets is not None and len(self.assets) != self.n:
            raise ValueError(f"{name}: assets {len(self.assets)} ≠ X {self.n}")
        self.ts = self.lc[:, LC["timestamp"]].astype("int64")
        # مفتاح المجموعة المقطعية: يوم UTC افتراضياً. day_ns آخر (مثلاً 32 ساعة) لبيانات لا تتطابق طوابع عملاتها —
        # كفريم الساعة بـ stride=32 حيث تبدأ نوافذ كل عملة من إدراجها: الأرضية المشتركة (من epoch) تجمع كل عملة مرّة.
        self.day_key = (self.ts // self.day_ns) * self.day_ns
        self.misaligned = self.ts != self.day_key            # طابع ليس بداية مجموعته (00:00 UTC في اليومي)
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

    def content_hash(self, max_rows=4096):
        """بصمة محتوى القسم: الأهداف وlast_candles وأسماء العملات كاملة، وعيّنة صفوف X متباعدة بانتظام (X قد تبلغ
        جيجابايتات). تفرّق بين بيانات بنفس الشكل والتواريخ ومحتوى مختلف (إعادة بناء بعد إصلاح في خط الأنابيب)."""
        import hashlib
        h = hashlib.sha1()
        for a in (self.ycls, self.yreg, self.lc):
            h.update(np.ascontiguousarray(a).tobytes())
        if self.assets is not None:
            h.update("\x1f".join(map(str, self.assets)).encode())
        rows = np.unique(np.linspace(0, self.n - 1, min(self.n, int(max_rows))).astype("int64")) if self.n else []
        h.update(np.ascontiguousarray(np.asarray(self.X[rows], dtype="float32")).tobytes())
        h.update(str((self.n,) + tuple(np.shape(self.X)[1:])).encode())
        return h.hexdigest()[:16]

    def take(self, idx, name=None):
        """قسم جديد من صفوف idx فقط (ينسخ X[idx] — للمجموعات الفرعية الصغيرة/الاختبار)."""
        idx = np.sort(np.asarray(idx))
        y = {}
        for i, t in enumerate(self.targets):
            y[f"y_{t}_class"] = self.ycls[idx, i].copy()       # 1/0 — ترميز خط الأنابيب الحالي
            y[f"y_{t}_reg"] = self.yreg[idx, i]
        return PanelSplit(np.asarray(self.X[idx]), y, self.lc[idx],
                          None if self.assets is None else self.assets[idx], name or self.name, self.targets,
                          day_ns=self.day_ns, target_scale=self.target_scale, target_mode=self.target_mode,
                          entry_close_reg=self.entry_close_reg)

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

    def make_batch(self, day_list, max_coins=None, rng=None, min_coins=None):
        """يجمع أيام الدفعة. max_coins (تدريب فقط، اختياري): عيّنة عشوائية من عملات الأيام الأكبر منه — يكسر
        «كل عيّنة مرّة واحدة» عمداً، لذا الافتراضي None.
        min_coins (تدريب فقط، اختياري): عدد عملات عشوائي لكل مجموعة k ~ U[min_coins, min(max_coins, n)] — كي يتعلّم
        النموذج العمل بأي عدد عملات يراه وقت الاستخدام (السوق كله أو قائمة مراقبة صغيرة). None = السلوك السابق حرفياً.
        المجموعة تُقسَم عشوائياً إلى ⌊n/k⌋ جزءاً (كل جزء مجموعة انتباه مستقلّة بين k و2k−1 عملة، وبحدّ max_coins إن
        أُعطي) — لا تُرمى البقية: لو أُخذت k عملة فقط لرأى المتغيّر نحو نصف عيّنات التدريب في الحقبة، فتختلط مقارنته
        بمتغيّر بلا min_coins بين «سياق متغيّر» و«بيانات أقل»."""
        rng = rng or np.random.default_rng(0)
        idx_lists = []
        for d in day_list:
            ii = self.day_indices(d)
            if min_coins:
                hi = min(int(max_coins or len(ii)), len(ii))
                k = int(rng.integers(min(int(min_coins), hi), hi + 1))
                if k < len(ii):
                    n_parts = max(len(ii) // k, -(-len(ii) // int(max_coins)) if max_coins else 1)
                    idx_lists.extend(np.sort(p) for p in np.array_split(rng.permutation(ii), n_parts))
                    continue
            elif max_coins and len(ii) > max_coins:
                ii = np.sort(rng.choice(ii, max_coins, replace=False))
            idx_lists.append(ii)
        return self.assemble(idx_lists)

    def assemble(self, idx_lists, scored=None):
        """دفعة من قائمة مجموعات (كل مجموعة فهارس صفوف تتشارك الانتباه): المجموعة b ← day=b، والمواضع 0..n_b-1.
        make_batch يمرّر أياماً كاملة (أو عيّنات منها)؛ chunk_groups يمرّر أجزاءً من الأيام لتقييم «k عملة».
        scored (اختياري، موازٍ لـ idx_lists): الصفوف التي يُؤخذ تنبؤها من هذه المجموعة ← مفتاح "score" منطقي؛
        البقية سياق فقط (تُتنبّأ في مجموعة أخرى)."""
        idx, day, pos, sc = [], [], [], []
        for b, ii in enumerate(idx_lists):
            idx.append(ii)
            day.append(np.full(len(ii), b, dtype="int32"))
            pos.append(np.arange(len(ii), dtype="int32"))
            sc.append(np.ones(len(ii), bool) if scored is None else np.isin(ii, scored[b]))
        idx = np.concatenate(idx)
        srt = np.argsort(idx, kind="stable")      # قراءة X بترتيب الذاكرة أسرع؛ الترتيب داخل الدفعة لا يغيّر شيئاً
        idx, day, pos = idx[srt], np.concatenate(day)[srt], np.concatenate(pos)[srt]
        out = {"idx": idx, "x": np.asarray(self.X[idx], dtype="float32"), "day": day, "pos": pos,
               "ycls": self.ycls[idx], "yreg": self.yreg[idx], "yrank": self.yrank[idx]}
        if scored is not None:
            out["score"] = np.concatenate(sc)[srt]
        return out

    def iter_batches(self, batch_samples=1024, max_days=16, shuffle=False, seed=0, epoch=0, max_coins=None,
                     min_coins=None):
        rng = np.random.default_rng([int(seed), int(epoch), 1])
        for dl in self.batch_plan(batch_samples, max_days, shuffle, seed, epoch):
            yield self.make_batch(dl, max_coins=max_coins if shuffle else None, rng=rng,
                                  min_coins=min_coins if shuffle else None)

    # ── تقييم بعدد عملات محدود ──
    def chunk_groups(self, k=None, seed=0):
        """يقسم كل مجموعة عشوائياً إلى أجزاء من k عملة بالضبط، فتُتنبّأ كل عيّنة مرّة واحدة وهي ترى k−1 عملة أخرى من
        طابعها فقط. يُرجع (members, scored): members[i] عملات الجزء i (ما يراه الانتباه)، وscored[i] ما يُؤخذ تنبؤه منه.
        الباقي حين لا يقسم k عدد العملات: جزء أخير من آخر k عملة في الترتيب العشوائي، يُحسب منه الباقي فقط والبقية سياق.
        مجموعة أصغر من k تبقى كاملة. k=None: المجموعات كاملة (التنبؤ العادي).
        التغطية الكاملة تجعل مقاييس كل k على نفس الصفوف بالضبط، فالفرق بينها أثر السياق لا أثر اختيار العيّنات."""
        if not k:
            groups = [self.day_indices(d) for d in range(self.n_days)]
            return groups, groups
        k, rng = int(k), np.random.default_rng(seed)
        members, scored = [], []
        for d in range(self.n_days):
            ii = rng.permutation(self.day_indices(d))
            n_full = len(ii) // k
            for j in range(n_full):
                members.append(np.sort(ii[j * k:(j + 1) * k]))
                scored.append(members[-1])
            rest = ii[n_full * k:]
            if len(rest):
                members.append(np.sort(ii[-k:]) if n_full else np.sort(ii))
                scored.append(np.sort(rest))
        return members, scored

    def iter_group_batches(self, groups, batch_samples=1024, max_days=16, scored=None):
        """دفعات من مجموعات جاهزة (مخرَج chunk_groups) بنفس قواعد batch_plan: تُملأ حتى batch_samples أو max_days.
        مع scored تحمل كل دفعة مفتاح "score" (انظر assemble)."""
        cur, cur_s, cur_n = [], [], 0
        for i, g in enumerate(groups):
            if cur and (cur_n + len(g) > batch_samples or len(cur) >= max_days):
                yield self.assemble(cur, None if scored is None else cur_s)
                cur, cur_s, cur_n = [], [], 0
            cur.append(g)
            if scored is not None:
                cur_s.append(scored[i])
            cur_n += len(g)
        if cur:
            yield self.assemble(cur, None if scored is None else cur_s)

    # ── فحوص السلامة ──
    def mixed_timestamp_groups(self):
        """عدد المجموعات التي تجمع أكثر من طابع دقيق واحد. في مجموعة كهذه تنتهي نافذة عملة عند T ونافذة قرينها عند
        T+1h…: مدخلات القرين تحوي شموع أفق الأولى، والانتباه عبر العملات يقرؤها (تسرّب من المستقبل)."""
        if not self.n:
            return 0
        k, t = self.day_key[self.order], self.ts[self.order]
        first = np.repeat(t[self.bounds[:-1]], self.sizes)
        return int(len(np.unique(k[t != first])))

    def check(self, min_group=5, allow_mixed_timestamps=False):
        """يُرجع قاموس فحوص؛ ok=False إن فشل فحص جوهري (تغطية، نقاء اليوم، تكرار عملة في يوم، أو مجموعة تخلط طوابع
        مختلفة — انظر mixed_timestamp_groups). allow_mixed_timestamps=True يتجاوز الأخير صراحةً (لتجربة تقيس التسرّب
        نفسه مثلاً)، ويبقى العدد ظاهراً في الجدول."""
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
            "assets": int(len(np.unique(self.assets))) if self.assets is not None else -1,
            "first_day": str(pd.Timestamp(int(self.days[0])))[:10] if self.n else "",
            "last_day": str(pd.Timestamp(int(self.days[-1])))[:10] if self.n else "",
            "coins_per_day_min": int(self.sizes.min()) if self.n else 0,
            "coins_per_day_median": float(np.median(self.sizes)) if self.n else 0,
            "coins_per_day_max": int(self.sizes.max()) if self.n else 0,
            "each_sample_once": coverage_ok, "day_groups_pure": bool(pure),
            "misaligned_timestamps": int(self.misaligned.sum()), "misaligned_assets": mis_assets[:20],
            "duplicate_asset_day": dups, "groups_mixed_timestamps": self.mixed_timestamp_groups(),
            f"days_lt_{min_group}_coins": int(small.sum()),
            f"samples_in_days_lt_{min_group}": int(self.sizes[small].sum()),
        }
        out["ok"] = (coverage_ok and pure and dups == 0
                     and (allow_mixed_timestamps or out["groups_mixed_timestamps"] == 0))
        return out


def _concat_dict(split_dict, model_tf):
    names = list(split_dict)
    parts = [split_dict[a] for a in names]
    X = np.concatenate([p[f"X_{model_tf}"] for p in parts])
    lc = np.concatenate([np.asarray(p["last_candles"]) for p in parts])
    y = {k: np.concatenate([np.asarray(p["y"][k]).ravel() for p in parts]) for k in parts[0]["y"]}
    assets = np.concatenate([np.array([a] * len(p["last_candles"]), dtype=object) for a, p in zip(names, parts)])
    return X, y, lc, assets


# عمودا (آخر، مستقبلي) في last_candles لكل هدف: y_{هدف}_reg = clip(مستقبلي/آخر − 1) × reg_target_scale في وضع 'return'.
_RETURN_COLS = {"high": ("last_high", "future_high_max"), "low": ("last_low", "future_low_min"),
                "close": ("last_close", "future_close")}


def implied_target_scale(y, last_candles, targets=TARGETS, min_rows=10, agree=0.9, rtol=1e-3):
    """المقياس الذي بُنيت به y_{هدف}_reg كما تشهد به البيانات نفسها: y_reg ÷ (مستقبلي/آخر − 1) من last_candles.

    يُرجع المقياس إن اتّفق عليه ≥ agree من الصفوف (عوائد بين 1e-4 و0.5، فالقصّ ودقّة float32 لا تُربكه)، وإلا None:
    أهداف ليست عائداً مضروباً في ثابت (relative/volatility من retarget_splits، 'window_scale' القديم) لا مقياس لها هنا."""
    lc = np.asarray(last_candles, dtype="float64")
    ratios = []
    for t in targets:
        key = f"y_{t}_reg"
        if key not in y or t not in _RETURN_COLS:
            continue
        a, b = _RETURN_COLS[t]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = lc[:, LC[b]] / lc[:, LC[a]] - 1.0
        v = np.asarray(y[key], dtype="float64").ravel()
        ok = np.isfinite(r) & np.isfinite(v) & (np.abs(r) > 1e-4) & (np.abs(r) < 0.5)
        ratios.append(v[ok] / r[ok])
    ratios = np.concatenate(ratios) if ratios else np.zeros(0)
    if len(ratios) < min_rows:
        return None
    s = float(np.median(ratios))
    if not np.isfinite(s) or s <= 0:
        return None
    return s if float(np.mean(np.abs(ratios / s - 1.0) < rtol)) >= agree else None


def _resolve_target_scale(members, model_targets):
    """مقياس y_reg لقسم بلا target_scale صريح: الختم 'reg_target_scale' إن وُجد، وإلا 1.0 — **ما لم** تشهد البيانات
    بمقياس آخر فيُرفع خطأ. كان الغياب يُقرأ 1.0 بصمت، فتخرج أعمدة mu_*/pred_* المصدَّرة أكبر بـ100× (1h_s8) دون أي
    عَرَض في IC/AUC (docs/research/audit/r2_06_panel_scale_unstamped.py). ملف قديم بلا مقياس: y = عائد ⇒ 1.0 كما كان."""
    stamps = {float(m["reg_target_scale"]) for m in members if "reg_target_scale" in m}
    if len(stamps) > 1:
        raise ValueError(f"أجزاء القسم مختومة بمقاييس مختلفة: {sorted(stamps)}")
    if stamps:
        return stamps.pop()
    for m in members:
        s = implied_target_scale(m["y"], m["last_candles"], model_targets)
        if s is not None and abs(s - 1.0) > 1e-3:
            raise ValueError(
                f"القسم بلا ختم 'reg_target_scale' لكن y_*_reg = عائد × {s:.6g} بحسب last_candles. قراءته 1.0 تُفسد كل "
                f"عائد وسعر مُصدَّر. اختم القسم بمقياس البيانات (split['reg_target_scale'] = dataset['reg_target_scale']، "
                f"كما يفعل دفتر main في القسم ٣) أو مرّر target_scale صراحةً.")
    return 1.0


def _per_target_scale(members, targets, scale):
    """مقياس كل هدف المختوم (reg_target_scales من retarget_splits) إن اختلف عن المشترك ← مصفوفة بترتيب targets؛
    وإلا المشترك كما هو (السلوك القديم)."""
    per = (members[0].get("reg_target_scales") or {}) if members else {}
    arr = [float(per.get(t, scale)) for t in targets]
    return np.asarray(arr) if any(abs(v - float(scale)) > 1e-12 for v in arr) else scale


def panel_split_from(split_or_dict, model_tf, asset_names=None, name="", targets=TARGETS, day_ns=DAY_NS,
                     target_scale=None):
    """من مخرَج split_data/retarget_splits: قسم مدمج (train/val) أو قاموس {عملة: قسم} (test).
    asset_names للقسم المدمج: split_asset_names(dataset, 'train'|'val') من دفتر main (بنفس أقنعة split_data).
    day_ns: عرض مجموعة الطوابع (يوم UTC افتراضياً) — انظر PanelSplit.
    target_scale: None يقرأ 'reg_target_scale' المختوم على القسم (دفتر main يختمه من البيانات)؛ غيابه = 1.0 إلا إن
    شهدت last_candles بمقياس آخر ⇒ ValueError (انظر _resolve_target_scale)."""
    members = [split_or_dict] if "y" in split_or_dict else list(split_or_dict.values())
    if target_scale is None:
        target_scale = _per_target_scale(members, tuple(targets), _resolve_target_scale(members, tuple(targets)))
    modes = {m.get("target_mode") for m in members}
    if len(modes) > 1:
        raise ValueError(f"أجزاء القسم مختومة بأوضاع هدف مختلفة: {sorted(modes, key=str)}")
    close_regs = {m.get("entry_close_reg") for m in members}
    if len(close_regs) > 1:
        raise ValueError(f"أجزاء القسم مختومة بتعريفات close مختلفة: {sorted(close_regs, key=str)}")
    if "y" in split_or_dict:
        X, y, lc = split_or_dict[f"X_{model_tf}"], split_or_dict["y"], split_or_dict["last_candles"]
        assets = asset_names if asset_names is not None and len(asset_names) == len(lc) else None
    else:
        X, y, lc, assets = _concat_dict(split_or_dict, model_tf)
    return PanelSplit(X, y, lc, assets=assets, name=name, targets=targets, day_ns=day_ns, target_scale=target_scale,
                      target_mode=modes.pop() if modes else None,
                      entry_close_reg=close_regs.pop() if close_regs else None)


def entry_range_to_prices(entry, high=None, low=None, close=None, close_reg="abs_return", p_close_up=None):
    """عكس أهداف entry_range (retarget_splits في main، القسم ٣-ب) إلى أسعار — نفس entry_range_to_prices في الدفتر.
    الأهداف بوحدة العائد (بعد القسمة على المقياس) وP = آخر إغلاق:
        high → P·(1 + high)،  low → P·(1 − low)
        close (abs_return) → P·(1 + s·close)، s = +1 إن كان p_close_up ≥ 0.5 وإلا −1؛ ويُرجَع الاتجاهان أيضاً
                              close_up = P·(1 + close)، close_down = P·(1 − close). بلا p_close_up لا يُرجَع "close".
        close (range_pos)  → pred_low + clip(close, 0, 1)·(pred_high − pred_low) — يحتاج high وlow.
    يُرجع {اسم: سعر} لما أمكن حسابه."""
    P = np.asarray(entry, dtype="float64")
    out = {}
    if high is not None:
        out["high"] = P * (1.0 + np.asarray(high, dtype="float64"))
    if low is not None:
        out["low"] = P * (1.0 - np.asarray(low, dtype="float64"))
    if close is not None:
        c = np.asarray(close, dtype="float64")
        if close_reg == "range_pos":
            if "high" in out and "low" in out:
                out["close"] = out["low"] + np.clip(c, 0.0, 1.0) * (out["high"] - out["low"])
        elif close_reg == "abs_return":
            out["close_up"], out["close_down"] = P * (1.0 + c), P * (1.0 - c)
            if p_close_up is not None:
                out["close"] = np.where(np.asarray(p_close_up, dtype="float64") >= 0.5, out["close_up"], out["close_down"])
        else:
            raise ValueError(f"close_reg غير معروف: {close_reg!r} — المتاح: abs_return, range_pos")
    return out


def group_ns_for(base_tf, stride):
    """عرض المجموعة المقطعية لبيانات محاذاة على الشبكة (align_windows_to_grid في خط الأنابيب): stride × مدة الفريم.
    اليومي بـ stride=1 ← DAY_NS (السلوك الافتراضي نفسه)؛ 1h بـ stride=8 ← 8 ساعات = الطابع الدقيق، فكل مجموعة
    تجمع العملات التي تنتهي نوافذها في اللحظة نفسها تماماً."""
    return int(max(int(stride), 1) * pd.Timedelta(str(base_tf).replace("D", "d")).value)


def format_checks(checks):
    """جدول نصي لفحوص عدّة أقسام."""
    df = pd.DataFrame(checks).set_index("split").T
    return df.to_string()
