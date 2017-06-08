from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class AttentionLayer(Layer):

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        # Create a trainable weight variable for this layer.
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A Attention layer should be called '
                             'on a list of 2 inputs.')
        if not input_shape[0][2] == input_shape[1][2]:
            raise ValueError('Embedding sizes should be of the same size')
        self.kernel = self.add_weight(shape=(input_shape[0][2], input_shape[0][2]),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(AttentionLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        return K.dot(K.transpose(inputs[0]),K.dot(self.kernel, inputs[1]))

    def compute_output_shape(self, input_shape):
        return (None, None, None)