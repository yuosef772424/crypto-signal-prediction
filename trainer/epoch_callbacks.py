"""
PURPOSE:  DriveMirror, EpochCheckpointCallback and EpochGuard: per-epoch save, Drive mirror, interruption guard.
TAGS:     DriveMirror, EpochCheckpointCallback, EpochGuard, drive mirror, epoch checkpoint, colab disconnect
PITFALLS: Mirrors to Drive only if config run.mirror_dir is set. Executed into the one shared trainer namespace by
          trainer/_loader.py (never imported on its own): names from other modules resolve at call time.
"""
# @title 7.2) DriveMirror (اختياري) + EpochCheckpointCallback + EpochGuard
class DriveMirror(tf.keras.callbacks.Callback):
    """نسخ دوري لمجلد التشغيل بالكامل إلى مسار دائم (مثل Drive) — طبقة أمان إضافية اختيارية.
    يعمل كنسخ مباشر لشجرة مجلد واحدة (بلا أي دمج/ترقيم منفصل)، لذا لا يوجد أي خطر تعارض."""

    def __init__(self, source_dir: str, mirror_dir: str, every: int = 1, verbose: int = 1):
        super().__init__()
        self.source_dir = source_dir
        self.mirror_dir = mirror_dir
        self.every = max(int(every), 1)
        self.verbose = verbose

    def _mirror(self):
        mount_drive_if_needed(self.mirror_dir)
        shutil.copytree(self.source_dir, self.mirror_dir, dirs_exist_ok=True)
        if self.verbose:
            print(f"☁️ [Mirror] {self.source_dir} → {self.mirror_dir}")

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every == 0:
            self._mirror()

    def on_train_end(self, logs=None):
        self._mirror()   # النسخة النهائية (بعد استرجاع أفضل أوزان وإعادة حفظ الـ checkpoint)


class EpochCheckpointCallback(tf.keras.callbacks.Callback):
    """يحفظ الحالة الكاملة في نهاية كل N حقبة، ويجمع حالة أي كولباك يدعم get_state().

    - عدّاد الحقب المنجزة (trainer.ckpt_epoch) يُحدَّث في **كل** حقبة، لا فقط عند الحفظ — ليكون المرجع الصادق
      لـ initial_epoch حتى لو كان save_every > 1.
    - on_train_end يُعيد حفظ الحالة: بعد fit() يكون النموذج في الذاكرة قد تغيّر (تبديل أوزان EMA من Keras، ثم
      استرجاع أفضل أوزان) — فنحفظ الحالة النهائية كي يطابق ما على القرص ما في الذاكرة، ولا تختلف «المتابعة في
      نفس الجلسة» عن «الاستئناف في جلسة جديدة»."""

    def __init__(self, checkpoint_manager: CheckpointManager, stateful_callbacks: Optional[list] = None,
                 save_every: int = 1, verbose: int = 1):
        super().__init__()
        self.ckpt_mgr = checkpoint_manager
        self.stateful_callbacks = stateful_callbacks or []
        self.save_every = max(int(save_every), 1)
        self.verbose = verbose

    def _collect_states(self) -> dict:
        return {cb.__class__.__name__: cb.get_state() for cb in self.stateful_callbacks if hasattr(cb, "get_state")}

    def on_epoch_end(self, epoch, logs=None):
        self.ckpt_mgr.trainer.ckpt_epoch.assign(epoch + 1)
        if (epoch + 1) % self.save_every != 0:
            return
        path = self.ckpt_mgr.save(epoch + 1, self._collect_states())
        if self.verbose:
            print(f"💾 [Checkpoint] Epoch {epoch + 1} → {path}")

    def on_train_end(self, logs=None):
        done = int(self.ckpt_mgr.trainer.ckpt_epoch.numpy())
        if done > 0:
            path = self.ckpt_mgr.save(done, self._collect_states())
            if self.verbose:
                print(f"💾 [Checkpoint] الحالة النهائية (Epoch {done}) → {path}")


class EpochGuard(tf.keras.callbacks.Callback):
    """يمنع الخطأ الصامت الذي يُصفّر جداول العقوبات: إن مرّرتَ initial_epoch لا يطابق عدد الحقب التي أنجزها
    المدرّب فعلًا (نسيته فصار 0 مثلًا) فإن ParamScheduler وجدول lr سيحسبان قيمهما لحقبة خاطئة — أي
    lambda_reg=0 وlambda_calib=0 وتسخين lr من جديد فوق نموذج متقارب."""

    def __init__(self):
        super().__init__()
        self._checked = False

    def on_train_begin(self, logs=None):
        self._checked = False

    def on_epoch_begin(self, epoch, logs=None):
        if self._checked:
            return
        self._checked = True
        done = int(self.model.ckpt_epoch.numpy())
        if epoch != done:
            raise RuntimeError(
                f"❌ initial_epoch={epoch} لكن المدرّب أنجز {done} حقبة فعلًا. بهذا الشكل ستُحسب جداول العقوبات "
                f"وlr للحقبة {epoch} بدل {done} (قد تعود lambda_reg وlambda_calib إلى الصفر). "
                f"استخدم initial_epoch={done} (القيمة العائدة من build_training_system)، أو ابدأ من جديد عبر "
                f"train_mode='new'، أو عطّل الحارس عمدًا بـ config['run']['strict_epoch_guard']=False.")
