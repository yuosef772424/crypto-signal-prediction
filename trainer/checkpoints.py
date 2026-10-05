"""
PURPOSE:  Save/resume system: CheckpointManager, archive_state, build_trainer_variables, stage_warm_start_weights.
TAGS:     CheckpointManager, TRAINER_REGISTRY, STATE_ENTRIES, has_saved_state, archive_state, resume, warm_start,
          checkpoint
PITFALLS: Build optimizer variables BEFORE restoring, or the optimizer momentum is lost (the whole point of this
          module). Executed into the one shared trainer namespace by trainer/_loader.py (never imported on its own):
          names from other modules resolve at call time.

## 7) نظام الحفظ والاستئناف — حلّ المشكلة المطلوبة تحديدًا

### جوهر المشكلة القديمة
كان يتم إنشاء **مدرّب جديد** (`AdvancedTrainer`) عند الاستئناف، ثم تحميل أوزان
النموذج فقط (`base_model.load_weights(...)`)، بينما `optimizer` (Adam/AdamW)
يُبنى من جديد بحالة ابتدائية صفرية (momentum = 0, variance = 0).

### الحل
1. **نحفظ كل شيء معًا**: `tf.train.Checkpoint(trainer=trainer)` يتتبّع تلقائيًا: أوزان `self.model`،
   `self.optimizer` بالكامل (بما فيها slots الزخم/الفارينس وEMA)، كل `scheduled_vars`/`loss_weights`/
   `uncertainty_layer`، و`self.ckpt_epoch`.

2. **بناء المتغيرات قبل الاسترجاع — بلا أي تدريب فعلي (🆕 مُصلَح)**: الاسترجاع يحتاج أن تكون كل
   المتغيرات (بما فيها *slots* الـ optimizer) موجودة مسبقًا. النسخة القديمة كانت تُنشئها بتنفيذ
   `train_on_batch` تجريبي ثم تتراجع يدويًا عن الأوزان القابلة للتدريب — لكن التراجع لا يشمل كل شيء:
   `optimizer.iterations` يبقى 1 لا 0، عزوم Adam **الداخلية** (لا الأوزان) لا تُعاد، وطبقات مثل
   `BatchNormalization` تُحدِّث إحصاءاتها المتحركة بلا أي تراجع عنها إطلاقًا. الآن `build_trainer_variables`
   تستدعي تمريرًا أماميًا واحدًا بـ`training=False` (لا يُحدِّث BatchNorm) ثم `optimizer.build(...)` فقط
   (يُنشئ الـ slots دون أي `apply_gradients`) — فلا يوجد أي أثر جانبي يحتاج تراجعًا عنه أصلًا.

3. **منع إعادة استخدام مدرّب بحالة غير متطابقة مع config الحالي (🆕 مُصلَح)**: `TRAINER_REGISTRY` كان
   يُعيد أي مدرّب مخزَّن لنفس `run_dir` بصرف النظر عن `config` الممرَّر — فتغيير `config` (مثلًا تفعيل
   الموازنة التلقائية) في خلية ثم إعادة تشغيلها في نفس الجلسة لا يُطبَّق. الآن تُقارَن بصمة `config`
   (`config_fingerprint`) قبل إعادة الاستخدام؛ أي اختلاف يُعيد البناء تلقائيًا.

4. **`config['run']['train_mode']` (🆕)**: بدل التخمين الضمني (checkpoint موجود ⇒ استئناف)، يمكنك
   التحكم صراحة: `auto` (نفس السلوك القديم) | `new` (يبدأ من الصفر، وأي حالة سابقة **تُؤرشَف لا تُحذف**
   في `run_dir/_archive/<timestamp>/`) | `resume` (يفشل بوضوح إن لم توجد حالة، بدل بدء صامت من الصفر) |
   `warm_start` (انظر البند التالي).

5. **`warm_start` — حلّ مباشر لمشكلة «عقوبات الثقة تبدأ من صفر»**: عندما تكمل التدريب من نموذج مُنجز
   سابقًا بطريقة `load_weights` بسيطة (لا عبر checkpoint هذا الإطار)، فإن `config['run']['train_mode']
   = 'warm_start'` مع `config['run']['warm_start'] = {'weights_path': ..., 'epochs_done': N}` يضبط
   **فورًا** كل معاملات `config['loss']['schedules']` (مثل `lambda_reg`, `lambda_calib`) على قيمتها عند
   الحقبة N بدل الصفر — وهذا يمنع بالضبط انهيار الموثوقية الذي وصفتَه (النموذج يُدرَّب حقبة إضافية
   بلا أي عقوبة ثقة فجأة). `optimizer` يبقى جديدًا بالضرورة (لا توجد حالته المحفوظة)، لذا `lr_rewarmup_epochs`
   اختياري لتسخين قصير لمعدل التعلّم بدل قفزه لقيمته الكاملة فوق عزوم Adam صفرية.

### ⚠️ تنبيه حاسم بخصوص "التخزين المؤقت" لـ Colab
مسار `/content/...` (بما فيه `/content/drive` **إن لم يُركَّب Drive فعليًا**) يُمسح بالكامل عند
انقطاع/انتهاء الجلسة. **لضمان الاستئناف عبر جلسات مختلفة، يجب أن يكون `run_dir` مسارًا دائمًا** — الأبسط:
مسار داخل `/content/drive/MyDrive/...` (بعد تركيب Drive).
"""
# @title 7.1) CheckpointManager + أرشفة الحالة + بناء المتغيرات دون تحديث
TRAINER_REGISTRY: Dict[str, Dict[str, Any]] = {}  # run_dir -> {trainer, ckpt_mgr, callbacks, fingerprint, model_sig}

# ما يُعدّ «حالة تشغيل سابقة» داخل run_dir (تُنقل كلها معًا إلى _archive عند البدء من جديد)
STATE_ENTRIES = ("checkpoint", "meta.json", "best.weights.h5", "best_meta.json", "best.prev.weights.h5")


def has_saved_state(directory: Optional[str]) -> bool:
    """هل يوجد checkpoint فعلي (لا مجرد مجلد فارغ) في هذا المسار؟"""
    if not directory:
        return False
    return tf.train.latest_checkpoint(os.path.join(directory, "checkpoint")) is not None


def archive_state(directory: str, tag: str) -> str:
    """ينقل (لا يحذف) كل حالة التشغيل السابقة إلى directory/_archive/<tag>/ — قابلة للاسترجاع يدويًا."""
    dest = os.path.join(directory, "_archive", tag)
    os.makedirs(dest, exist_ok=True)
    for name in STATE_ENTRIES:
        src = os.path.join(directory, name)
        if os.path.exists(src):
            shutil.move(src, os.path.join(dest, name))
    return dest


class CheckpointManager:
    """يغلّف tf.train.Checkpoint/CheckpointManager + ملف meta.json صغير للحالات القابلة للتسلسل"""

    def __init__(self, trainer: "GenericTrainer", run_dir: str, max_to_keep: int = 3):
        self.trainer = trainer
        self.run_dir = run_dir
        mount_drive_if_needed(run_dir)
        os.makedirs(run_dir, exist_ok=True)

        self.ckpt_dir = os.path.join(run_dir, "checkpoint")
        self.meta_path = os.path.join(run_dir, "meta.json")

        # tf.train.Checkpoint(trainer=trainer) يتتبّع trainer بالكامل بشكل متداخل:
        # trainer.model, trainer.optimizer (بعد compile), trainer.scheduled_vars,
        # trainer.loss_weights, trainer.uncertainty_layer, trainer.ckpt_epoch — كل شيء دفعة واحدة.
        self.checkpoint = tf.train.Checkpoint(trainer=trainer)
        self.manager = tf.train.CheckpointManager(self.checkpoint, self.ckpt_dir, max_to_keep=max_to_keep)

    def has_checkpoint(self) -> bool:
        return self.manager.latest_checkpoint is not None

    def restore(self) -> Tuple[int, dict]:
        """يُستدعى **بعد** build_trainer_variables() وفقط إن وُجد checkpoint. يُعيد (initial_epoch, callback_states)"""
        latest = self.manager.latest_checkpoint
        status = self.checkpoint.restore(latest)
        status.assert_existing_objects_matched()  # يفشل بوضوح إن تغيّرت بنية config/النموذج بشكل غير متوافق

        meta = {}
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception as e:
                print(f"⚠️ تعذّرت قراءة meta.json ({e}) — الأوزان والـ optimizer سليمة لكن حالة الكولباكس (best/wait) ستبدأ من جديد")

        initial_epoch = int(self.trainer.ckpt_epoch.numpy())
        print(f"✅ تم استرجاع الحالة الكاملة (أوزان + optimizer + جداول) من: {latest}")
        print(f"   الاستئناف من Epoch {initial_epoch + 1}")
        return initial_epoch, meta.get("callback_states", {})

    def save(self, epoch: int, callback_states: Optional[dict] = None):
        self.trainer.ckpt_epoch.assign(epoch)
        path = self.manager.save()
        atomic_write_json(self.meta_path, {
            "epoch": epoch, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "callback_states": callback_states or {},
        })
        return path


def build_trainer_variables(trainer: "GenericTrainer", sample_batch: Tuple[Any, ...]):
    """ينشئ كل متغيرات trainer (النموذج + slots وعدّاد الـ optimizer + متغيرات EMA) **دون أي خطوة تدريب**.

    لماذا: tf.train.Checkpoint.restore يحتاج المتغيرات موجودة، وslots الـ optimizer لا تُخلق إلا عند build().
    الطريقة القديمة كانت تنفّذ train_on_batch ثم تُعيد الأوزان القابلة للتدريب فقط — فتبقى آثارها في:
    optimizer.iterations (=1) وعزوم Adam غير الصفرية، وإحصاءات BatchNormalization المتحركة، وسلسلة RNG.
    هنا: استدعاء أمامي واحد بـ training=False (لا يحدّث BatchNorm) + optimizer.build(...) (يخلق ولا يحدّث)."""
    x_sample = sample_batch[0]
    trainer.model(x_sample, training=False)
    trainer.optimizer.build(trainer.trainable_variables)


def stage_warm_start_weights(config: dict) -> Tuple[str, int, int]:
    """يتحقق من إعدادات warm_start وينسخ ملف الأوزان إلى مكان مؤقت — قبل أرشفة run_dir (قد يكون الملف بداخله)."""
    ws = config["run"].get("warm_start") or {}
    path, epochs_done = ws.get("weights_path"), ws.get("epochs_done")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"❌ warm_start: run.warm_start.weights_path غير موجود: {path!r}")
    if epochs_done is None or int(epochs_done) < 0:
        raise ValueError("❌ warm_start: يجب تحديد run.warm_start.epochs_done (كم حقبة تدرّبها هذا النموذج فعلًا). "
                         "لا نخمّنه: تخمين 0 يعيد جداول العقوبات إلى الصفر — وهي المشكلة التي وُجد هذا الوضع لحلّها.")
    staged = os.path.join(tempfile.mkdtemp(prefix="warm_start_"), "staged.weights.h5")
    shutil.copy(path, staged)
    return staged, int(epochs_done), int(ws.get("lr_rewarmup_epochs", 0))
