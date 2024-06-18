from keras.src.backend.config import epsilon
from keras.src.layers import Layer
import keras.src.ops as ops


class EuclidianDistance(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        featsA, featsB = inputs
        sumSquared = ops.sum(ops.square(featsA - featsB), axis=1, keepdims=True)

        return ops.sqrt(ops.maximum(sumSquared, epsilon()))
