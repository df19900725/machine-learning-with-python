"""
This module provides some model fusions method and class
@Author: DuFei
@Created Time: 2019/11/03 11:14
"""

import numpy as np
import tensorflow as tf
from typing import List


class FusionModel(tf.keras.layers.Layer):
    def __init__(self, models: List[tf.keras.Model], **kwargs):
        self.models = models
        super().__init__(**kwargs)

    def predict(self, x):
        res = []
        for model in self.models:
            res.append(model.predict(x))

        res = np.stack(res, axis=-1)
        return np.mean(res, axis=-1)
