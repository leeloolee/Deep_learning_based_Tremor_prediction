##
import tensorflow as tf
import glob
import models
import matplotlib.pyplot as plt
from matplotlib import gridspec
import json
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from scipy import signal

##
with open('config.json', 'r') as f:
    config = json.load(f)

TRAINED_TASK = config['TASK']
##

# Todo dir 바꾸기
models_ANN_dir = 'saved_modelANN_x축microscope_1_static.hdf5'
models_RNN_dir = 'saved_modelRNN_x축microscope_1_static.hdf5'
models_LSTM_dir = 'saved_modelLSTM_x축microscope_1_static.hdf5'
models_PHTNET_dir = 'saved_modelPHTNet_x축microscope_1_static.hdf5'

models_OURS_dir = 'saved_model/OURS_static.ckpt'

model_ANN = tf.keras.models.load_model(models_ANN_dir)
model_RNN = tf.keras.models.load_model(models_RNN_dir)
model_LSTM = tf.keras.models.load_model(models_LSTM_dir)
model_PHTNET = tf.keras.models.load_model(models_PHTNET_dir)

model_ours = models.NBeatsNet(forecast_length=20)
model_ours.load_weights(models_OURS_dir)

data1 = pd.read_csv('C:/Users/admin/Desktop/마도요/Polhemus_GUI_Version1.2/Polhemus_GUI_Version1.2/logs/myunghoon/microscope_1_static/m_microscope_1_static (3).csv')
data2 = pd.read_csv('C:/Users/admin/Desktop/마도요/Polhemus_GUI_Version1.2/Polhemus_GUI_Version1.2/logs/myunghoon/microscope_1_0.5/m_microscope_1_0.5 (3).csv')
data3 = pd.read_csv('C:/Users/admin/Desktop/마도요/Polhemus_GUI_Version1.2/Polhemus_GUI_Version1.2/logs/1_1/ground_truth_csv/Hold Still_Low_1_2.csv')
data4 = pd.read_csv('C:/Users/admin/Desktop/마도요/Polhemus_GUI_Version1.2/Polhemus_GUI_Version1.2/logs/1_1/ground_truth_csv/Draw Circle1_Low_1_2.csv')
### univariate


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])

    return np.array(data), np.array(labels)

def short2(model, col, data):

    # 정규화 일단 제외

    fs = 240
    fc = 2 # Cut-off frequency of the filter
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(7, w, 'low')



    uni_data_z = data[col]
    uni_data_filt = signal.filtfilt(b,a, uni_data_z)  # filt_x
    uni_data_z.index = data['1']

    uni_data_z = uni_data_z.values

    uni_past_history = 120
    uni_future_target = 0

    x_z_uni, _ = univariate_data(uni_data_z, 0, None,
                               uni_past_history,
                               uni_future_target)
    _, y_uni = univariate_data(uni_data_filt, 0, None,
                               uni_past_history,
                               uni_future_target)

    prediction = pd.DataFrame(model.predict(x_z_uni).reshape(1,-1))
    return prediction


def short(model, col, data):

    # 정규화 일단 제외

    fs = 240
    fc = 2 # Cut-off frequency of the filter
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(7, w, 'low')



    uni_data_z = data[col]
    uni_data_filt = signal.filtfilt(b,a, uni_data_z)  # filt_x


    uni_past_history = 120
    uni_future_target = 0

    _, y_uni = univariate_data(uni_data_filt, 0, None,
                               uni_past_history,
                               uni_future_target)

    return y_uni

###
ground_truth = short(model_ANN,'x', data1)
prediction_ANN = short2(model_ANN, 'x', data1)
prediction_RNN = short2(model_RNN, 'x', data1)
prediction_LSTM = short2(model_LSTM, 'x', data1)
prediction_PHTNET = short2(model_PHTNET, 'x', data1)
prediction_OURS = short2(model_ours, 'x', data1)


plt.figure(figsize=(15, 6))

# ax.figure.set_size_inches(18,10)
# ax = plt.plot(data1['x'].values[120:], color = '25', label ='Raw Siganl')

plt.plot(prediction_ANN, color = 'green', label = 'ANN')
plt.plot(prediction_RNN, color = 'red', label = 'RNN')
plt.plot(prediction_LSTM, color = 'blue' , label = 'LSTM')
plt.plot(prediction_PHTNET, color = 'black', label = 'PHTNet')
#plt.plot(prediction_OURS, color = 'red', label = 'OURS')


plt.show()

##

def calculate_metric(ground_truth, prediction):
    RMSE = mean_squared_error(ground_truth, prediction) ** 0.5
    return RMSE



