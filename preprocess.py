# TODO normalize
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import os
import config


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


# 정규화 일단 제외


# 어느 라인인지 하이퍼파라미터임
def transforming(one_data_path):
    csv_path = one_data_path
    data = pd.read_csv(csv_path)
    uni_data_x = data['x']
    uni_data_filt = data['filt_x']  # filt_x
    uni_data_x.index = data['1']
    uni_data_filt.index = data['1']
    uni_data_x = uni_data_x.values
    uni_data_filt = uni_data_filt.values
    uni_past_history = 480
    uni_future_target = 0

    x_uni, _ = univariate_data(uni_data_x, 0, None,
                               uni_past_history,
                               uni_future_target)
    _, y_uni = univariate_data(uni_data_filt, 0, None,
                               uni_past_history,
                               uni_future_target)
    a, b = scaling(x_uni, y_uni)
    return np.array(a), np.array(b)


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


def appended_data(path_list):
    a = np.array([])
    b = np.array([])
    for i in path_list:
        data_x, data_y = transforming(i)
        a = np.concatenate([a, data_x], axis=0) if a.size else data_x
        b = np.concatenate([b, data_y], axis=0) if b.size else data_y
    return a, b


def cross_path_list(path, cross, subject=None, task=None):
    # cross_path_list ('dataset', 'subject', 'myunghoon', 'Bare_0.5')
    all_path_list = glob.glob(path + '/*/ground_truth_csv/*/*.csv')
    train_list = []
    test_list = []
    if cross == 'subject':
        assert subject != None, 'subject is needed'
        for i in all_path_list:
            if i.split(os.path.sep)[1] != subject:
                if i.split(os.path.sep)[3] == task:
                    train_list.append(i)
                elif task == None:
                    train_list.append(i)
                else:
                    pass
            else:
                if i.split(os.path.sep)[3] == task:
                    test_list.append(i)
                elif task == None:
                    test_list.append(i)
                else:
                    pass
        return train_list, test_list
    else:
        assert task != None, 'task is needed'
        for i in all_path_list:
            if i.split(os.path.sep)[3] != task:
                if i.split(os.path.sep)[1] == subject:
                    train_list.append(i)
                elif subject == None:
                    train_list.append(i)
                else:
                    pass
            else:
                if i.split(os.path.sep)[1] == subject:
                    test_list.append(i)
                elif subject == None:
                    test_list.append(i)
                else:
                    pass
        return train_list, test_list


if __name__ == '__main__':
    train_path_list, test_path_list = cross_path_list(path=config.path, cross=config.cross, task=config.task,
                                                      subject=config.subject)
    data_train_x, data_train_y = appended_data(train_path_list)
    data_test_x, data_test_y = appended_data(test_path_list)

    if config.subject == None:
        np.save('data_train_x' + ',' + config.cross + ',' + config.task, data_train_x)
        np.save('data_test_x' + ',' + config.cross + ',' + config.task, data_test_x)
        np.save('data_train_y' + ',' + config.cross + ',' + config.task, data_train_y)
        np.save('data_test_y' + ',' + config.cross + ',' + config.task, data_test_y)

    elif config.task == None:
        np.save('data_train_x' + ',' + config.cross + ',' + config.subject, data_train_x)
        np.save('data_test_x' + ',' + config.cross + ',' + config.subject, data_test_x)
        np.save('data_train_y' + ',' + config.cross + ',' + config.subject, data_train_y)
        np.save('data_test_y' + ',' + config.cross + ',' + config.subject, data_test_y)
    else:
        np.save('data_train_x' + ',' + config.cross + ',' + config.task + ',' + config.subject, data_train_x)
        np.save('data_test_x' + ',' + config.cross + ',' + config.task + ',' + config.subject, data_test_x)
        np.save('data_train_y' + ',' + config.cross + ',' + config.task + ',' + config.subject, data_train_y)
        np.save('data_test_y' + ',' + config.cross + ',' + config.task + ',' + config.subject, data_test_y)
