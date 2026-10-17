"""
PURPOSE:  Optional purged walk-forward K-Fold training: purged_walk_forward_splits, run_kfold_training.
TAGS:     purged_walk_forward_splits, run_kfold_training, k-fold, walk-forward, purge, resume per fold
PITFALLS: Optional; not exercised by the load-time tests. Executed into the one shared trainer namespace by
          trainer/_loader.py (never imported on its own): names from other modules resolve at call time.

## 12) (اختياري) تدريب K-Fold مع استئناف كامل لكل Fold على حدة

كل خصوصية بنية بياناتك (مدخلات متعددة، إطارات زمنية متعددة، إلخ) تعيش بالكامل
داخل `dataset_builder_fn` التي تكتبها أنت — الإطار لا يفترض عنها شيئًا. كل
Fold يحصل على `run_dir` مستقل (`{base_run_dir}/fold_{i}`)، وبالتالي يستفيد من
نفس آلية الاستئناف الكاملة: إن انقطع التدريب في منتصف Fold 3 مثلًا، إعادة
تشغيل هذه الخلية تتخطى Fold 1 و 2 (اكتملا فعلًا) وتستأنف Fold 3 من حيث توقف.
"""
# @title 12) K-Fold Training (اختياري)
def purged_walk_forward_splits(n_samples: int, n_splits: int = 5, purge: int = 50, embargo: int = 50):
    fold_size = n_samples // (n_splits + 1)
    splits = []
    for i in range(1, n_splits + 1):
        val_start = i * fold_size
        val_end = min((i + 1) * fold_size, n_samples)
        train_end = max(0, val_start - purge)
        splits.append((np.arange(0, train_end), np.arange(val_start, val_end)))
    return splits


def run_kfold_training(
    n_samples: int,
    model_builder_fn: Callable[[], tf.keras.Model],
    config_template: dict,
    dataset_builder_fn: Callable[[np.ndarray, np.ndarray], Tuple[tf.data.Dataset, tf.data.Dataset, Tuple[Any, Any]]],
    n_splits: int = 5, purge: int = 50, embargo: int = 50, min_train: int = 100, min_val: int = 20,
):
    """
    dataset_builder_fn(train_idx, val_idx) -> (train_ds, val_ds, sample_batch)
    """
    splits = purged_walk_forward_splits(n_samples, n_splits, purge, embargo)
    base_run_dir = config_template["run"]["run_dir"]

    trained_models, histories = [], []
    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        print(f"\n{'=' * 80}\n📂 Fold {fold_idx}/{n_splits} — train={len(train_idx)} val={len(val_idx)}\n{'=' * 80}")
        if len(train_idx) < min_train or len(val_idx) < min_val:
            print("⚠️ بيانات غير كافية — تخطّي هذا الـ Fold")
            continue

        fold_config = copy.deepcopy(config_template)
        fold_config["run"]["run_dir"] = os.path.join(base_run_dir, f"fold_{fold_idx}")

        train_ds, val_ds, sample_batch = dataset_builder_fn(train_idx, val_idx)

        trainer, callbacks, initial_epoch = build_training_system(model_builder_fn, fold_config, sample_batch)
        history = trainer.fit(
            train_ds, validation_data=val_ds, initial_epoch=initial_epoch,
            epochs=fold_config["run"]["epochs"], callbacks=callbacks, verbose=1,
        )

        trained_models.append(trainer.model)
        histories.append(history.history)
        best_val = min(history.history.get("val_loss", [np.nan])) if history.history.get("val_loss") else float("nan")
        print(f"✅ Fold {fold_idx} انتهى — أفضل val_loss: {best_val:.4f}")

    print(f"\n{'=' * 80}\n🏁 انتهى K-Fold: {len(trained_models)} نموذج جاهز للـ Ensemble\n{'=' * 80}")
    return trained_models, histories
