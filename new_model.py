import tensorflow as tf
from tensorflow.keras.layers import Flatten, Conv1D, Input, MaxPool1D, GlobalAveragePooling1D, Dropout, Dense, \
    Bidirectional, GRU, TimeDistributed, Lambda, Concatenate, Reshape, Subtract, Add, Activation, BatchNormalization
from tensorflow.keras.utils import get_custom_objects

from tensorflow.keras.optimizers import Adam

import numpy as np
from tensorflow.keras import Model




# Cyclic RL is needed
def model1():
    input = Input(shape=(480, 1))
    x = Conv1D(100, 10, activation='relu', padding='same')(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x1 = Conv1D(100, 10, activation='relu', padding='same')(x)
    x = tf.keras.layers.Add()([x1, x])
    x = MaxPool1D(3)(x)
    x1 = Conv1D(100, 10, activation='relu', padding='same')(x)
    x = tf.keras.layers.Add()([x1, x])
    x = Conv1D(100, 10, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(100)(x)
    output = Dense(1)(x)
    model = Model(input, output)
    model.compile(loss='mse', optimizer='Adam', metrics=['mse'])
    return model


def model2():
    input = Input(shape=(480, 1))
    x = Bidirectional(GRU(100, return_sequences=True, activation='sigmoid'))(input)  # why sigmoid?s
    x = Bidirectional(GRU(100, return_sequences=True, activation='sigmoid'))(x)
    x = Flatten()(x)
    x = Dense(64)(x)
    output = Dense(1)(x)  # TODO time distri?
    model = Model(input, output)
    model.compile(loss='mse', optimizer='Adam', metrics=['mse'])
    return model


def model3():
    input = Input(shape=(480, 1))
    x = Flatten()(input)
    x = Dense(480, activation='relu')(x)
    x = Dense(480, activation='relu')(x)
    output = Dense(1)(x)
    model = Model(input, output)
    model.compile(loss='mse', optimizer='Adam', metrics=['mse'])
    return model






def model4():
# looking for 1 layer weight
    input = Input(shape=(480, 1))
    x = Flatten()(input)
    output = Dense(1)(x)
    model = Model(input, output)
    model.compile(loss='MSE', optimizer='Adam', metrics=['mse'])

    return model



def model5():
    # attention
    NotImplementedError



class NBeatsNet:
    GENERIC_BLOCK = 'generic'
    TREND_BLOCK = 'trend'
    SEASONALITY_BLOCK = 'seasonality'

    def __init__(self,
                 input_dim=1,
                 exo_dim=0,
                 backcast_length=480,
                 forecast_length=1,
                 stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
                 nb_blocks_per_stack=5,
                 thetas_dim=(4, 8),
                 share_weights_in_stack=False,
                 hidden_layer_units=256,
                 nb_harmonics=None
                 ):

        self.stack_types = stack_types
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim = thetas_dim
        self.units = hidden_layer_units
        self.share_weights_in_stack = share_weights_in_stack
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.input_dim = input_dim
        self.exo_dim = exo_dim
        self.input_shape = (self.backcast_length, self.input_dim)
        self.exo_shape = (self.backcast_length, self.exo_dim)
        self.output_shape = (self.forecast_length, self.input_dim)
        self.weights = {}
        self.nb_harmonics = nb_harmonics
        assert len(self.stack_types) == len(self.thetas_dim)

        x = Input(shape=self.input_shape, name='input_variable')
        x_ = {}
        for k in range(self.input_dim):
            x_[k] = Lambda(lambda z: z[..., k])(x)
        e_ = {}
        if self.has_exog():
            e = Input(shape=self.exo_shape, name='exos_variables')
            for k in range(self.exo_dim):
                e_[k] = Lambda(lambda z: z[..., k])(e)
        else:
            e = None
        y_ = {}

        for stack_id in range(len(self.stack_types)):
            stack_type = self.stack_types[stack_id]
            nb_poly = self.thetas_dim[stack_id]
            for block_id in range(self.nb_blocks_per_stack):
                backcast, forecast = self.create_block(x_, e_, stack_id, block_id, stack_type, nb_poly)
                for k in range(self.input_dim):
                    x_[k] = Subtract()([x_[k], backcast[k]])
                    if stack_id == 0 and block_id == 0:
                        y_[k] = forecast[k]
                    else:
                        y_[k] = Add()([y_[k], forecast[k]])

        for k in range(self.input_dim):
            y_[k] = Reshape(target_shape=(self.forecast_length, 1))(y_[k])
        if self.input_dim > 1:
            y_ = Concatenate(axis=-1)([y_[ll] for ll in range(self.input_dim)])
        else:
            y_ = y_[0]

        if self.has_exog():
            model = Model([x, e], y_)
        else:
            model = Model(x, y_)

        model.summary()

        self.n_beats = model

    def has_exog(self):
        return self.exo_dim > 0

    @staticmethod
    def load(filepath, custom_objects=None, compile=True):
        from tensorflow.keras.models import load_model
        return load_model(filepath, custom_objects, compile)

    def _r(self, layer_with_weights, stack_id):
        # mechanism to restore weights when block share the same weights.
        # only useful when share_weights_in_stack=True.
        if self.share_weights_in_stack:
            layer_name = layer_with_weights.name.split('/')[-1]
            try:
                reused_weights = self.weights[stack_id][layer_name]
                return reused_weights
            except KeyError:
                pass
            if stack_id not in self.weights:
                self.weights[stack_id] = {}
            self.weights[stack_id][layer_name] = layer_with_weights
        return layer_with_weights

    def create_block(self, x, e, stack_id, block_id, stack_type, nb_poly):

        # register weights (useful when share_weights_in_stack=True)
        def reg(layer):
            return self._r(layer, stack_id)

        # update name (useful when share_weights_in_stack=True)
        def n(layer_name):
            return '/'.join([str(stack_id), str(block_id), stack_type, layer_name])

        backcast_ = {}
        forecast_ = {}
        d1 = reg(Dense(self.units, activation = 'relu', name=n('d1')))
        d2 = reg(Dense(self.units, activation = 'relu', name=n('d2')))
        d3 = reg(Dense(self.units, activation = 'relu', name=n('d3')))
        d4 = reg(Dense(self.units, activation = 'relu', name=n('d4')))

        if stack_type == 'generic':
            theta_b = reg(Dense(nb_poly, activation='linear', use_bias=False, name=n('theta_b')))
            theta_f = reg(Dense(nb_poly, activation='linear', use_bias=False, name=n('theta_f')))
            backcast = reg(Dense(self.backcast_length, activation='linear', name=n('backcast')))
            forecast = reg(Dense(self.forecast_length, activation='linear', name=n('forecast')))
        elif stack_type == 'trend':
            theta_f = theta_b = reg(Dense(nb_poly, activation='linear', use_bias=False, name=n('theta_f_b')))
            backcast = Lambda(trend_model, arguments={"is_forecast": False, "backcast_length": self.backcast_length,
                                                      "forecast_length": self.forecast_length})
            forecast = Lambda(trend_model, arguments={"is_forecast": True, "backcast_length": self.backcast_length,
                                                      "forecast_length": self.forecast_length})
        else:  # 'seasonality'
            if self.nb_harmonics:
                theta_b = reg(Dense(self.nb_harmonics, activation='linear', use_bias=False, name=n('theta_b')))
            else:
                theta_b = reg(Dense(self.forecast_length, activation='linear', use_bias=False, name=n('theta_b')))
            theta_f = reg(Dense(self.forecast_length, activation='linear', use_bias=False, name=n('theta_f')))
            backcast = Lambda(seasonality_model,
                              arguments={"is_forecast": False, "backcast_length": self.backcast_length,
                                         "forecast_length": self.forecast_length})
            forecast = Lambda(seasonality_model,
                              arguments={"is_forecast": True, "backcast_length": self.backcast_length,
                                         "forecast_length": self.forecast_length})
        for k in range(self.input_dim):
            if self.has_exog():
                d0 = Concatenate()([x[k]] + [e[ll] for ll in range(self.exo_dim)])
            else:
                d0 = x[k]
            d1_ = d1(d0)

            d1_ = Dropout(0.2)(d1_)
            d2_ = d2(d1_)

            d2_ = Dropout(0.2)(d2_)
            d3_ = d3(d2_)

            d3_ = Dropout(0.2)(d3_)
            d4_ = d4(d3_)

            d4_ = Dropout(0.2)(d4_)
            theta_f_ = theta_f(d4_)
            theta_b_ = theta_b(d4_)
            backcast_[k] = backcast(theta_b_)
            forecast_[k] = forecast(theta_f_)

        return backcast_, forecast_

    def compile_model(self, loss, learning_rate):
        optimizer = Adam(lr=learning_rate)
        self.compile(loss=loss, optimizer=optimizer)

    def __getattr__(self, name):
        # https://github.com/faif/python-patterns
        # model.predict() instead of model.n_beats.predict()
        # same for fit(), train_on_batch()...
        attr = getattr(self.n_beats, name)

        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            return attr(*args, **kwargs)

        return wrapper


def linear_space(backcast_length, forecast_length, fwd_looking=True):
    ls = tf.keras.backend.arange(-float(backcast_length), float(forecast_length), 1) / backcast_length
    if fwd_looking:
        ls = ls[backcast_length:]
    else:
        ls = ls[:backcast_length]
    return ls


def seasonality_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.get_shape().as_list()[-1]
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    t = linear_space(backcast_length, forecast_length, fwd_looking=is_forecast)
    s1 = tf.stack([tf.cos(2 * np.pi * i * t) for i in range(p1)], axis=0)
    s2 = tf.stack([tf.sin(2 * np.pi * i * t) for i in range(p2)], axis=0)
    if p == 1:
        s = s2
    else:
        s = tf.keras.backend.concatenate([s1, s2], axis=0)
    s = tf.cast(s, np.float32)
    return tf.keras.backend.dot(thetas, s)


def trend_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.shape[-1]
    t = linear_space(backcast_length, forecast_length, fwd_looking=is_forecast)
    t = tf.transpose(tf.stack([t ** i for i in range(p)], axis=0))
    t = tf.cast(t, np.float32)
    return tf.keras.backend.dot(thetas, tf.transpose(t))


class Mish(tf.keras.layers.Activation):
    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'


def mish(x):
    return x * tf.keras.backend.tanh(tf.keras.backend.softplus(x))


get_custom_objects().update({'mish': Mish(mish)})

## in functional modeling
#x = Activation('mish')(x)