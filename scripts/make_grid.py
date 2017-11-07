import pandas as pd
import numpy as np
import itertools


def TimeSplit(data, deltaTime_in_days):
    data.sort_values(by='timestamp', inplace=True)
    delta = 86890000000000 * deltaTime_in_days  # to timestamp milliseconds
    time_split = [list(group) for key, group in itertools.groupby(data.values, key=lambda x: int(x[4] / delta))]
    for i in range(len(time_split)):
        time_split[i] = [x[:, None] for x in time_split[i]]
        time_split[i] = pd.DataFrame((np.concatenate(time_split[i], axis=1)).T,\
                                         columns=['x', 'y', 'z', 'magnitude', 'T'])
    return time_split


def XSplit(data, delta):
    data.sort_values(by=['x'], inplace=True)
    key_func = lambda x: int(x[0] / delta)
    split = [{'key': key, 'group': list(group)} for key, group in itertools.groupby(data.values, key=key_func)]

    for i in range(len(split)):
        split[i]['group'] = [x[:, None] for x in split[i]['group']]
        split[i]['group'] = pd.DataFrame((np.concatenate(split[i]['group'], axis=1)).T,\
                                         columns=['x', 'y', 'z', 'magnitude', 'T'])
    return split


def YSplit(data, delta):
    data.sort_values(by=['y'], inplace=True)
    key_func = lambda x: int(x[1] / delta)
    split = [{'key': key, 'group': list(group)} for key, group in itertools.groupby(data.values, key=key_func)]

    for i in range(len(split)):
        split[i]['group'] = [x[:, None] for x in split[i]['group']]
        split[i]['group'] = pd.DataFrame((np.concatenate(split[i]['group'], axis=1)).T,\
                                         columns=['x', 'y', 'z', 'magnitude', 'T'])
    return split


def TXYSplit(data, delta_time, delta_x, delta_y):
    time_split = TimeSplit(data, delta_time)
    for idx, split in enumerate(time_split): time_split[idx] = XSplit(split, delta_x)
    for T_split in time_split:
        for i in range(len(T_split)): T_split[i]['group'] = YSplit(T_split[i]['group'], delta_y)
    return time_split


def TXYMatrix(time_split, delta_x, delta_y, map_x=20.0, map_y=40.0):
    X, Y, T = int(map_x / delta_x), int(map_y / delta_y), len(time_split)
    xyt_matrix = np.empty((X, Y, T), dtype=object)

    for time, xy_split in enumerate(time_split):
        for x_split in xy_split:
            x_idx = x_split['key']
            for y_split in x_split['group']:
                y_idx = y_split['key']
                xyt_matrix[x_idx, y_idx, time] = y_split['group']
    return xyt_matrix


def target_anomaly_matrix(xyt_matrix, threshold):
    anomaly_flag = np.full(xyt_matrix.shape, 0.0)
    for x in range(xyt_matrix.shape[0]):
        for y in range(xyt_matrix.shape[1]):
            for t in range(xyt_matrix.shape[2]):
                current_slice = xyt_matrix[x,y,t]
                if current_slice is None:
                    anomaly_flag[x,y,t] = 0.0
                else:
                    anomaly_flag[x,y,t] = float(np.sum(current_slice['magnitude'] > threshold) > 0.0)
    return anomaly_flag
