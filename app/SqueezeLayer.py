from tensorflow import keras as K
from tensorflow.keras.layers import Layer

class SqueezeLayer(Layer):
    '''
    Keras squeeze layer
    Taken from milsed https://github.com/marl/milsed/
    '''
    def __init__(self, axis=-1, **kwargs):
        super(SqueezeLayer, self).__init__(**kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        # shape = np.array(input_shape)
        # shape = shape[shape != 1]
        # return tuple(shape)
        shape = list(input_shape)
        del shape[self.axis]
        return tuple(shape)

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)

    def call(self, x, mask=None):
        return K.backend.squeeze(x, axis=self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(SqueezeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
