"""
PURPOSE:  Shared imports (numpy, tensorflow, keras layers) and the `register` decorator (keras serializable, package
          'nigts').
TAGS:     imports, register, register_keras_serializable, nigts, numpy, tensorflow, keras layers
PITFALLS: `register` must exist before any layer module runs; the registered name is 'nigts>ClassName', so renaming a
          class breaks saved models. Executed into the one shared model namespace by model/_loader.py (never imported
          on its own): names from other modules resolve at call time.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers, initializers

register = tf.keras.utils.register_keras_serializable(package="nigts")
