import models
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import glob
import numpy as np
#
TEST = 'microscope_1_0.05' # 'microscope_1_static' # or test = 'microscope_1_0.05'

def main():
    test_data_list = glob.glob(".\\preprocess_data\\" + TEST + '*.npy')


    total_data_list = glob.glob(".\\preprocess_data\\" + '*.npy')

    for list in test_data_list:
        total_data_list.remove(list)


    x_x, x_y, y_x, y_y, z_x, z_y = [], [], [], [], [], []
    for i in total_data_list:
        if i[-7] == 'x':
            if i[-5] == 'x':
                x_x.append(np.load(i))
            else:
                x_y.append(np.load(i))
        elif i[-7] == 'y':
            if i[-5] == 'x':
                y_x.append(np.load(i))

            else:
                y_y.append(np.load(i))
        else:
            if i[-5] == 'x':
                z_x.append(np.load(i))

            else:
                z_y.append(np.load(i))


    model_path = 'saved_model\\' + 'LSTM_'+ TEST + '.ckpt'
    es = EarlyStopping(mode='min', verbose=1, patience=3)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, save_best_only=True, save_weights_only=False,
                                                     verbose=1)

    callbacks = [
        ReduceLROnPlateau(monitor='val_mse',
                          factor=0.5,
                          patience=5,
                          verbose=1,
                          epsilon=0.00005,
                          mode='min'),
        es,
        cp_callback

    ]
    data_x_x = x_x.copy()
    data_x_y = x_y.copy()
    valid_x_x = data_x_x.pop(1)
    valid_x_y = data_x_y.pop(1)

    data_y_x = y_x.copy()
    data_y_y = y_y.copy()
    valid_y_x = data_y_x.pop(1)
    valid_y_y = data_y_y.pop(1)

    data_z_x = z_x.copy()
    data_z_y = z_y.copy()
    valid_z_x = data_z_x.pop(1)
    valid_z_y = data_z_y.pop(1)

    train_x_x = np.concatenate(data_x_x, axis=0)
    train_x_y = np.concatenate(data_x_y, axis=0)
    train_y_x = np.concatenate(data_y_x, axis=0)
    train_y_y = np.concatenate(data_y_y, axis=0)
    train_z_x = np.concatenate(data_z_x, axis=0)
    train_z_y = np.concatenate(data_z_y, axis=0)

    train_x = np.concatenate([train_x_x, train_y_x, train_z_x], axis=0)

    train_y = np.concatenate([train_x_y, train_y_y, train_z_y], axis=0)

    valid_x = np.concatenate([valid_x_x, valid_y_x, valid_z_x], axis=0)

    valid_y = np.concatenate([valid_x_y, valid_y_y, valid_z_y], axis=0)

    tf_data_train = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(
        2000).repeat().batch(
        64).prefetch(5)  # 128   # 고쳐주기
    # tf_data_valid = tf.data.Dataset.from_tensor_slices((train_data_x, train_data_y)).take(2662).batch(32)
    tf_data_test = tf.data.Dataset.from_tensor_slices((valid_x, valid_y)).shuffle(
        1000).repeat().batch(64)

    model_ann = models.ANN()


    model_ann.fit(tf_data_train, validation_data=tf_data_test, validation_steps=5, epochs=100, steps_per_epoch=10000,
                  callbacks=callbacks)
    model_path = 'saved_model' +'ANN_x축'+TEST + '.hdf5'
    model_ann.save(model_path)

if __name__ == '__main__':
    print("hello world")
    main()