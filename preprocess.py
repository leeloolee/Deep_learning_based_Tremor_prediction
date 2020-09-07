import glob
import numpy as np
import pandas as pd
import os

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
        labels.append(dataset[i:i + target_size])

    return np.array(data), np.array(labels)


# 정규화 일단 제외
from scipy import signal
fs = 240
fc = 2 # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(7, w, 'low')

# 어느 라인인지 하이퍼파라미터임
def transforming(one_data_path, B):
    csv_path = one_data_path
    data = pd.read_csv(csv_path)
    uni_data_x = data[B]
    uni_data_filt = signal.filtfilt(b,a, uni_data_x)
    # uni_data_filt = data['filt_'+B]  # filt_x
    uni_data_x.index = data['1']
    #uni_data_filt.index = data['1']
    uni_data_x = uni_data_x.values
    #uni_data_filt = uni_data_filt.values
    uni_past_history = 120
    uni_future_target = 1 #20 HPARAM

    x_uni, _ = univariate_data(uni_data_x, 0, None,
                               uni_past_history,
                               uni_future_target)
    _, y_uni = univariate_data(uni_data_filt, 0, None,
                               uni_past_history,
                               uni_future_target)

    #a, b = scaling(x_uni, y_uni)
    # return np.array(a), np.array(b)
    return np.array(x_uni), np.array(y_uni)

from sklearn import preprocessing


def scaling(foo, koo):
    data = []
    labels = []

    for i in range(len(foo)):
        v = foo[i, :]
        data.append((v - min(v)) / (max(v) - min(v)))
        labels.append(([koo[i].reshape(1)] - min(v)) / (max(v) - min(v)))

    return data, labels


# 정규화 일단 제외


def appended_data(path_list,B):
    a = np.array([])
    b = np.array([])
    for i in path_list:
        data_x, data_y = transforming(i,B)
        a = np.concatenate([a, data_x], axis=0) if a.size else data_x
        b = np.concatenate([b, data_y], axis=0) if b.size else data_y
    return a, b


if __name__ == "__main__":
    dir_list = glob.glob('.\\dataset\\myunghoon\\ground_truth_csv\\*')

    for dir in dir_list:
        A=dir.split('\\')[-1]
        for B in ['x', 'y', 'z']:
            all_path_list = glob.glob(
                '.\\dataset\\myunghoon\\ground_truth_csv\\' + A + '\\*.csv')

            data_train_x, data_train_y = appended_data(all_path_list, B)
            if not os.path.isdir(".\\preprocess_data"):
                os.mkdir(".\\preprocess_data")
            np.save(".\\preprocess_data\\"+A + '_'+B+'_x', data_train_x)
            np.save(".\\preprocess_data\\"+A + '_'+B+'_y', data_train_y)
