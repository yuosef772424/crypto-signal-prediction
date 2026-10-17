"""
PURPOSE:  SnapshotEnsemble: optional snapshot-ensemble callback.
TAGS:     SnapshotEnsemble, snapshot ensemble, callback
PITFALLS: Optional; inactive unless added to the callbacks. Executed into the one shared trainer namespace by
          trainer/_loader.py (never imported on its own): names from other modules resolve at call time.
"""
# @title 6.4) SnapshotEnsemble (اختياري)
class SnapshotEnsemble(tf.keras.callbacks.Callback):
    """حفظ لقطات (snapshots) في حقب محددة — لبناء ensemble لاحقًا.
    (BestWeightsSaver حُذفت: وظيفتها صارت جزءًا من BestModelTracker في 6.2)"""

    def __init__(self, save_epochs: List[int], save_dir: str, base_model: tf.keras.Model, verbose=1):
        super().__init__()
        self.save_epochs = set(save_epochs)
        self.save_dir = save_dir
        self.base_model = base_model
        self.verbose = verbose
        if self.save_epochs:
            os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) in self.save_epochs:
            path = os.path.join(self.save_dir, f"snapshot_epoch_{epoch + 1}.weights.h5")
            self.base_model.save_weights(path)
            if self.verbose:
                print(f"💾 [Snapshot] {path}")
