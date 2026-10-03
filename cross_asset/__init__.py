"""
PURPOSE:  Package entry of the cross-asset 'panel' model (phase 1): one sample = a full UTC day with all coins,
          attention across same-day coins; re-exports PanelSplit, panel_split_from, VARIANTS, run_panel_experiment.
TAGS:     panel model, cross-asset attention, package exports, phase 1, anti-memorization, PanelSplit,
          run_panel_experiment
PITFALLS: Kept free of TensorFlow at import time (model/train import it) so data/report stay usable in light CPU
          analysis scripts; do not add a top-level TF import here.

نموذج «اللوحة» عبر العملات (المرحلة ١): عيّنة التدريب = يوم UTC كامل بكل عملاته، وانتباه بين عملات اليوم نفسه.

الوحدات:
  data       : تجميع عيّنات split_data حسب اليوم (فهارس فقط — لا موتّر أيام×عملات×نافذة×ميزات)، ودفعات من عدّة أيام.
  model      : مُرمِّز زمني لكل عملة (نفس مُرمِّز النموذج الحالي حرفياً) ← طبقات Transformer عبر عملات اليوم (بقناع) ← رؤوس.
  train      : خسائر لكل يوم (BCE + Huber + حدّ ارتباط يومي اختياري)، حلقة تدريب قابلة للاستئناف، تنبؤ وتصدير إشارات.
  report     : تقييم أي ملف إشارات (النموذج الحالي أو اللوحة) بنفس المقاييس: AUC، IC يومي، IC مضبوط بالتقلّب، قوس ±5%، محفظة.
  experiment : تشغيل تجربة كاملة من متغيّرات دفتر main (يستدعيه القسم ٧-ح هناك، وtools/evaluate_trained_model.py --panel).
  selftest   : اختبارات ذاتية سريعة على بيانات تركيبية (التجميع، القناع، الخسائر، التصدير).

التصميم والتشغيل على Colab: docs/research/panel_phase1.md
"""
# بلا TensorFlow عند الاستيراد (model/train يستوردانه): report وdata يُستخدمان في سكربتات تحليل خفيفة
from .data import PanelSplit, panel_split_from  # noqa: F401
from .experiment import VARIANTS, run_panel_experiment  # noqa: F401
