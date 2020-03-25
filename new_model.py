import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Input, MaxPool1D, GlobalAveragePooling1D, Dropout, Dense, Bidirectional, GRU, TimeDistributed
from tensorflow.keras import Model

def model1():
    input = Input(shape = (480,1))
    x = Conv1D(100, 10, activation='relu')(input)
    x = Conv1D(100, 10, activation='relu')(x)
    x = MaxPool1D(3)(x)
    x = Conv1D(160, 10, activation='relu')(x)
    x = Conv1D(160, 10, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    output = Dense(1)(x)
    return Model(input, output)


def model2():
    input = Input(shape = (480,1))
    x = Bidirectional(GRU(100, return_sequences= True,  activation='tanh'))(input)
    x = Bidirectional(GRU(100, return_sequences= True,  activation='tanh'))(x)
    output = TimeDistributed(Dense(1))(x)
    return Model(input, output)

def model3():
    input = Input(shape = (480,1))
    x = Dense(100, activation ='relu')(input)
    x = Dense(100, activation ='relu')(x)
    x = Dense(100, activation ='relu')(x)
    x = Dense(100, activation='relu')(x)
    output = Dense(1)(x)
    return Model(input, output)

