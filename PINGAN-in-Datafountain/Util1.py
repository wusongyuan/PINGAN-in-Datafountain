# -*- coding:utf8 -*-
import pandas as pd
import numpy as np
import os
import csv

trip_count = 0

def read_csv(path, columns):
    """
    用于读取数据
    :param path: 文件路径 string类型
    :param is_train: 是否为训练文件 boolean类型
    :return: 读取文件后的dataframe
    """

    # 读取训练文件
    data_pd = pd.read_csv(path)

    # 对文件类型的判断
    data_pd.columns = columns

    return data_pd

def change_data_type(org_data_df):
    """
    用户改变各列的数据类型
    :param org_data_df:
    :return:
    """
    org_data_df['TERMINALNO'] = org_data_df['TERMINALNO'].astype('uint16')
    org_data_df['TIME'] = org_data_df['TIME'].astype('uint32')
    org_data_df['LONGITUDE'] = org_data_df['LONGITUDE'].astype('float32')
    org_data_df['LATITUDE'] = org_data_df['LATITUDE'].astype('float32')
    org_data_df['DIRECTION'] = org_data_df['DIRECTION'].astype('float32')
    org_data_df['SPEED'] = org_data_df['SPEED'].astype('float32')
    org_data_df['HEIGHT'] = org_data_df['HEIGHT'].astype('float32')
    org_data_df['CALLSTATE'] = org_data_df['CALLSTATE'].astype('uint8')

    return org_data_df

def get_delta(org_data, columns):
    for column in columns:
        org_data['SUB_' + column] = (org_data[column].shift(-1) - org_data[column]).shift(1) \
                                / org_data['SUB_TIME'] * 60

    return org_data

def data_mean(group_data, new_data, columns):
    df = new_data
    for column in columns:
        max_data = group_data[column].mean()
        df = df.join(max_data)
        df.rename(columns={column: 'MEAN_' + column}, inplace=True)
    return df

def data_max(group_data, new_data, columns):
    df = new_data
    for column in columns:
        max_data = group_data[column].max()
        df = df.join(max_data)
        df.rename(columns={column: 'MAX_' + column}, inplace=True)
    return df

def data_min(group_data, new_data, columns):
    df = new_data
    for column in columns:
        max_data = group_data[column].min()
        df = df.join(max_data)
        df.rename(columns={column: 'MIN_' + column}, inplace=True)
    return df

def data_std(group_data, new_data, columns):
    df = new_data
    for column in columns:
        max_data = group_data[column].std()
        df = df.join(max_data)
        df.rename(columns={column: 'STD_' + column}, inplace=True)
    return df

def deal_delta(new_data,org_data, columns,index_column):
    df = new_data
    for column in columns:
        gt_0_data = org_data.loc[org_data[column] > 0, column].groupby([index_column]).max()
        df = df.join(gt_0_data)
        df.rename(columns={column: column + '_GT_0_max'}, inplace=True)

        gt_0_data = org_data.loc[org_data[column] > 0, column].groupby([index_column]).mean()
        df = df.join(gt_0_data)
        df.rename(columns={column: column+'_GT_0_mean'}, inplace=True)

        gt_0_std = org_data.loc[org_data[column] > 0, column].groupby([index_column]).std()
        df = df.join(gt_0_std)
        df.rename(columns={column: column + '_GT_0_std'}, inplace=True)

        ls_0_data = org_data.loc[org_data[column] < 0, column].groupby([index_column]).min()
        df = df.join(ls_0_data)
        df.rename(columns={column: column + '_LS_0_min'}, inplace=True)

        ls_0_data = org_data.loc[org_data[column] < 0, column].groupby([index_column]).mean()
        df = df.join(ls_0_data)
        df.rename(columns={column: column+'_LS_0_mean'}, inplace=True)

        ls_0_std = org_data.loc[org_data[column] < 0, column].groupby([index_column]).std()
        df = df.join(ls_0_std)
        df.rename(columns={column: column + '_LS_0_std'}, inplace=True)

    return df

def type_fea(new_data, org_data, type_nums, columns, index_column):
    for type_num, column in zip(type_nums, columns):
        for i in range(type_num):
            new_data = new_data.join(
                org_data.loc[org_data[column] == i, column].groupby([index_column]).count())
            new_data.rename(columns={column: column + '_' + str(i)}, inplace=True)
    return new_data

def delta_count(new_data, org_data, columns, index_column):
    for column in columns:
        new_data = new_data.join(
            org_data.loc[org_data[column] > 0, column].groupby([index_column]).count())
        new_data.rename(columns={column: column + '_GT_0'}, inplace=True)

        new_data = new_data.join(
            org_data.loc[org_data[column] < 0, column].groupby([index_column]).count())
        new_data.rename(columns={column: column + '_LS_0'}, inplace=True)
    return new_data

def init_trip_id(x):
    global trip_count
    if x == 0:
        trip_count+=1
    return trip_count

def save(prediction, path_test_out):
    """
    将结果文件存储到预测结果路径下。
    :return:
    """

    with(open(os.path.join(path_test_out, "test.csv"), mode="w", newline='')) as outer:
        writer = csv.writer(outer)
        writer.writerow(["Id", "Pred"])
        for item in prediction:
            writer.writerow([item[0], item[1]])
        # ret_set = {}
        # for item in prediction:
        #     # 根据赛题要求，ID必须唯一。输出预测值时请注意去重
        #     if item[0] in ret_set:
        #         ret_set[item[0]].append(item[1])
        #     else:
        #         ret_set[item[0]] = [item[1]]
        #
        # for k in ret_set.keys():
        #     # 此处使用随机值模拟程序预测结果
        #     writer.writerow([k, sum(ret_set[k]) * 1.0 / len(ret_set[k])])