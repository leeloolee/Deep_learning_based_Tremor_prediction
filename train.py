import new_model
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from tensorflow import keras

from tensorflow.keras import metrics
import io
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import RectifiedAdam as RAdam
from tensorflow.keras.losses import mean_absolute_error as mae



logdir = os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
output_model_file = os.path.join(logdir, "model.h5")
file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')
length = 480


"""
def differentiable_smape(true, predicted):
    epsilon = 0.1
    true_o = true
    pred_o = predicted
    summ = tf.maximum(tf.abs(true_o) + tf.abs(pred_o) + epsilon, 0.5 + epsilon)
    smape = tf.abs(pred_o - true_o) / summ
    return smape
"""

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.

    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def figure_output(epoch, logs):
    # for i in range(1, 6):
    if epoch % 10 == 0:
        datas = pd.read_csv('dataset/myunghoon/microscope_1_0.5_' + str(1) + '.csv')
        fig_data = np.load('dataset/myunghoon/microscope_1_0.5_' + str(1) + '.npy')
        fig_data = fig_data[:, 480 - length:, :]  # 빼야할거


        figure_datas = np.array([])
        for i in range(len(fig_data)):
            v = fig_data[np.newaxis, i, :]
            if figure_datas.size == 0:
                figure_datas = np.array((v - np.min(v, axis=1)) / (np.max(v, axis=1) - np.min(v, axis=1)))
            else:
                figure_datas = np.concatenate(
                    (figure_datas, np.array((v - np.min(v, axis=1)) / (np.max(v, axis=1) - np.min(v, axis=1)))), axis=0)


        #train_data_x2 = np.append(np.zeros([len(figure_datas), 1, 1]), figure_datas [:, 1:,:] - figure_datas [:, :-1, :], axis=1)

        #figure_datas = np.append(figure_datas , train_data_x2, axis=2)



        figure = plt.figure(figsize=(16, 6))


        plt.plot(datas['x'][480:].values)
        plt.plot(datas['filt_x'][480:].values)
        plt.plot(
            model.predict(figure_datas).reshape(-1) * (np.max(fig_data[:,:,0], axis=1) - np.min(fig_data[:,:,0], axis=1)) + np.min(fig_data[:,:,0],
                                                                                                         axis=1))  #fig_data[:,] -> figdata
        cm_image = plot_to_image(figure)

        datas2 = pd.read_csv('dataset/myunghoon/microscope_0.1_' + str(1) + '.csv')
        fig_data = np.load('dataset/myunghoon/microscope_0.1_' + str(1) + '.npy')
        fig_data = fig_data[:, 480 - length:, :]  # 빼야할거

        figure_datas = np.array([])
        for i in range(len(fig_data)):
            v = fig_data[np.newaxis, i, :]  # 1 ->:
            if figure_datas.size == 0:
                figure_datas = np.array((v - np.min(v, axis=1)) / (np.max(v, axis=1) - np.min(v, axis=1)))
            else:
                figure_datas = np.concatenate(
                    (figure_datas, np.array((v - np.min(v, axis=1)) / (np.max(v, axis=1) - np.min(v, axis=1)))), axis=0)

        #train_data_x2 = np.append(np.zeros([len(figure_datas ), 1, 1]), figure_datas [:, 1:,:] - figure_datas [:, :-1, :], axis=1)

        #figure_datas = np.append(figure_datas , train_data_x2, axis=2)
        figure = plt.figure(figsize=(16, 6))

        plt.plot(datas2['x'][480:].values)
        plt.plot(datas2['filt_x'][480:].values)
        plt.plot(
            model.predict(figure_datas).reshape(-1)  * (np.max(fig_data[:,:,0], axis=1) - np.min(fig_data[:,:,0], axis=1)) + np.min(fig_data[:,:,0],
                                                                                                         axis=1))
        cm_image2 = plot_to_image(figure)

        # figure2 = plt.figure(figsize=(16, 6))
        # powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(model.predict(fig_data).reshape(-1), Fs=240, cmap='plasma')
        # # Run time Warning?
        # cm2_image = plot_to_image(figure2)

        with file_writer_cm.as_default():
            tf.summary.image("output_test" + str(1), cm_image, step=epoch)
            tf.summary.image("output_Train" + str(1), cm_image2, step=epoch)

            # tf.summary.image("spectral" + str(i), cm2_image, step=epoch)


train_data_x, train_data_y = np.load('data_train_x,task,microscope_1_0.5,myunghoon.npy'), np.load(
    'data_train_y,task,microscope_1_0.5,myunghoon.npy')
test_data_x, test_data_y = np.load('data_test_x,task,microscope_1_0.5,myunghoon.npy'), np.load(
    'data_test_y,task,microscope_1_0.5,myunghoon.npy')

train_data_x, train_data_y = train_data_x[1:,:,:], train_data_y[:-1,:,:]
test_data_x, test_data_y  = test_data_x[1:,:,:], test_data_y[:-1,:,:]

#train_data_x2 = np.append(np.zeros([len(train_data_x), 1, 1]), train_data_x[:, 1:, :] - train_data_x[:, :-1, :], axis=1)

#new_train_data = np.append(train_data_x, train_data_x2, axis=2)

#test_data_x2 = np.append(np.zeros([len(test_data_x), 1, 1]), test_data_x[:, 1:, :] - test_data_x[:, :-1, :], axis=1)

#new_test_data = np.append(test_data_x, test_data_x2, axis=2)

tf_data_train = tf.data.Dataset.from_tensor_slices((train_data_x[:, 480 - length:, :], train_data_y)).shuffle(
    200000).repeat().batch(
    128)  # 128   # 고쳐주기
# tf_data_valid = tf.data.Dataset.from_tensor_slices((train_data_x, train_data_y)).take(2662).batch(32)
tf_data_test = tf.data.Dataset.from_tensor_slices((test_data_x[:, 480 - length:, :], test_data_y)).shuffle(
    100000).repeat().batch(64)

model = new_model.NBeatsNet()

# model = model2.build_model()


model.compile(loss='mse', optimizer= RAdam(0.005), metrics=['mse']) ##

# ss

file_writer = tf.summary.create_file_writer(logdir)
# Define the per-epoch callback.
cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=figure_output)

model_path = 'saved_model\\' + '{epoch:2d}.hdf5'
ms = ModelCheckpoint(model_path, monitor='val_mse', save_best_only=True, mode='min')
es = EarlyStopping(mode='min', verbose=1, patience=50)

callbacks = [
    ReduceLROnPlateau(monitor='val_mse',
                      factor=0.5,
                      patience=5,
                      verbose=1,
                      epsilon=0.01,
                      mode='min'),
 #   es,
    tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0),
    tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_mse', save_freq=5),
    tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_mse', save_best_only=True),
    cm_callback
    
]



model.fit(tf_data_train, steps_per_epoch=100, validation_data=tf_data_test, validation_steps=20, epochs=1000,
          callbacks=callbacks)
model.save("./saved_model", overwrite=True)


