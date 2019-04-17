# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import lightgbm as lgb
from math import radians, cos, sin, asin, sqrt
import gc

def haversine1(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000

def get_distances(temp):
    distances = []
    for item in temp['TRIP_ID'].unique():
        temp_temp = temp.loc[temp['TRIP_ID']== item,:]
        temp_temp.index = range(len(temp_temp))
        n = temp_temp.shape[0]
        if n == 1:
            continue
        startlong = temp_temp.loc[0,'LONGITUDE']
        startlat = temp_temp.loc[0,'LATITUDE']
        endlong = temp_temp.loc[n-1,'LONGITUDE']
        endlat = temp_temp.loc[n-1, 'LATITUDE']
        distance = haversine1(startlong,startlat,endlong, endlat)
        distances.append(distance)
    if len(distances) == 0:
        distances.append(0)
    return np.array(distances)

def get_data(data,is_train):
    train1 = []
    # alluser = data['TERMINALNO'].nunique()
    # Feature Engineer, 对每一个用户生成特征:
    # trip特征, record特征(数量,state等),
    # 地理位置特征(location,海拔,经纬度等), 时间特征(星期,小时等), 驾驶行为特征(速度统计特征等)
    for item in data['TERMINALNO'].unique():
        temp = data.loc[data['TERMINALNO'] == item, :]
        temp.index = range(len(temp))
        # trip 特征
        num_of_trips = temp['TRIP_ID'].nunique()
        # record 特征
        num_of_records = temp.shape[0]
        num_of_state = temp[['TERMINALNO', 'CALLSTATE']]
        nsh = num_of_state.shape[0]
        num_of_state_0 = num_of_state.loc[num_of_state['CALLSTATE'] == 0].shape[0] / float(nsh)
        num_of_state_1 = num_of_state.loc[num_of_state['CALLSTATE'] == 1].shape[0] / float(nsh) + \
                         num_of_state.loc[num_of_state['CALLSTATE'] == 2].shape[0] / float(nsh)
        num_of_state_2 = num_of_state.loc[num_of_state['CALLSTATE'] == 3].shape[0] / float(nsh) + \
                         num_of_state.loc[num_of_state['CALLSTATE'] == 4].shape[0] / float(nsh)
        del num_of_state

        ### 地点特征
        dis = get_distances(temp)
        var_dis = np.var(dis)
        max_dis = np.max(dis)
        # 时间特征
        # temp['weekday'] = temp['TIME'].apply(lambda x:datetime.datetime.fromtimestamp(x).weekday())
        temp['hour'] = temp['TIME'].apply(lambda x: datetime.datetime.fromtimestamp(x).hour)
        dangerous_hour_state = np.zeros([2, 1])
        hour_state = np.zeros([24, 1])
        for i in range(24):
            hour_state[i] = temp.loc[temp['hour'] == i].shape[0] / float(nsh)
            if (8 <= i and i <= 10) or (20 <= i and i <= 22) or (14 <= i and i <= 16):
                dangerous_hour_state[0] += temp.loc[temp['hour'] == i].shape[0] / float(nsh)
            else:
                dangerous_hour_state[1] += temp.loc[temp['hour'] == i].shape[0] / float(nsh)

        # 驾驶行为特征
        max_speed = temp['SPEED'].max()
        var_speed = temp['SPEED'].var()
        max_height = temp['HEIGHT'].max()
        var_height = temp['HEIGHT'].var()
        var_direction = temp['DIRECTION'].var()
        mean_la = temp['LATITUDE'].mean()
        mean_lo = temp['LONGITUDE'].mean()

        # 添加label
        if is_train:
            target = temp.loc[0, 'Y']
        else:
            target = -1.0
        # 所有特征
        feature = [item, num_of_trips, num_of_records, num_of_state_0, num_of_state_1, num_of_state_2, max_speed,
                   var_speed, max_height, var_height, dangerous_hour_state[0][0], dangerous_hour_state[1][0]
            , float(hour_state[0]), float(hour_state[1]), float(hour_state[2]), float(hour_state[3]),
                   float(hour_state[4]), float(hour_state[5])
            , float(hour_state[6]), float(hour_state[7]), float(hour_state[8]), float(hour_state[9]),
                   float(hour_state[10]), float(hour_state[11])
            , float(hour_state[12]), float(hour_state[13]), float(hour_state[14]), float(hour_state[15]),
                   float(hour_state[16]), float(hour_state[17])
            , float(hour_state[18]), float(hour_state[19]), float(hour_state[20]), float(hour_state[21]),
                   float(hour_state[22]), float(hour_state[23])
            , max_dis, var_dis, var_direction,mean_la,mean_lo
            , target]
        train1.append(feature)
    train1 = pd.DataFrame(train1)

    train1.columns = featurename
    return train1

start_all = datetime.datetime.now()
# path
path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件
# path_train = "PINGAN-2018-train_demo.csv"  # 训练文件
# path_test = "PINGAN-2018-train_demo2.csv"  # 测试文件
path_result_out = "model/pro_result.csv" #预测结果文件路径

featurename = ['item', 'num_of_trips', 'num_of_records', 'num_of_state_0', 'num_of_state_1', 'num_of_state_2',
                   'max_speed', 'var_speed',
                   'max_height', 'var_height', 'hs', 'hd'
        , 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11'
        , 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23'
        , 'max_dis', 'var_dis', 'var_direction','mean_la','mean_lo'
        , 'target']

# 特征使用
feature_use = ['item', 'num_of_trips', 'num_of_records','num_of_state_0','num_of_state_1','num_of_state_2','max_speed','var_speed',\
               'max_height','var_height','hs','hd'
    ,'h0','h1','h2','h3','h4','h5','h6','h7','h8','h9','h10','h11'
    ,'h12','h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23'
    ,'max_dis','var_dis','var_direction']

# read train data
data = pd.read_csv(path_train)
# 根据用户和时间进行排序
data.sort_values(by=['TERMINALNO', 'TIME'], axis=0, ascending=True, inplace=True)
# 重新设置索引
data.reset_index(drop=True, inplace=True)

# lo_scales = np.array([[115,118],[116,123],[105,112],[109,118]])
# la_scales = np.array([[39, 41],[30, 36],[28, 33], [20, 26]])
# lo_scales = np.array([[116,123]])
# la_scales = np.array([[30, 36]])
# 北京、上海江苏、广东
lo_scales = np.array([[115,118],[116,123],[109,118]])
la_scales = np.array([[39, 42],[30, 36], [20, 26]])
def get_area_data(org_data):
    data = org_data.loc[:, ['mean_la', 'mean_lo']]
    index_set = set(data.index.tolist())
    index_list = []
    for lo_scale, la_scale in zip(lo_scales,la_scales):
        index = data.loc[(data['mean_lo'] > lo_scale[0])&(data['mean_lo'] < lo_scale[1])&(data['mean_la'] > la_scale[0])&(data['mean_la'] < la_scale[1])].index.tolist()
        index_set = index_set - set(index)
        index_list.append(np.array(index))
    index_list.append(np.array(list(index_set)))
    return index_list


features = get_data(data,True)
del data
gc.collect()
area_index = get_area_data(features)
models = []
# count = 0
for index in area_index:
    # print(count)
    # count+=1
    model_features = features.loc[index,:]
    # 采用lgb回归预测模型，具体参数设置如下
    model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                                  learning_rate=0.01, n_estimators=720,
                                  max_bin = 55, bagging_fraction = 0.8,
                                  bagging_freq = 5, feature_fraction = 0.2319,
                                  feature_fraction_seed=9, bagging_seed=25,
                                  min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
    # 训练、预测
    model_lgb.fit(model_features[feature_use].fillna(-1), model_features['target'])
    del model_features
    gc.collect()
    models.append(model_lgb)

print("train data process time:",(datetime.datetime.now()-start_all).seconds)

data = pd.read_csv(path_test)
features = get_data(data,False)
del data
gc.collect()
area_index= get_area_data(features)

items = []
pred = []
for i, index in enumerate(area_index):
    model_features = features.loc[index, :]
    pred.extend(models[i].predict(model_features[feature_use].fillna(-1)))
    items.extend(model_features['item'])

users = np.sort(np.unique(np.array(items)))
pred = np.array(pred)
y_pred = []
for user in users:
    y_pred.append(np.mean(pred[items==user]))

print("lgb success")

# output result
result = pd.DataFrame(users,columns=["item"])
result['pre'] = y_pred
result = result.rename(columns={'item':'Id','pre':'Pred'})
result.to_csv(path_result_out,header=True,index=False)
print("Time used:",(datetime.datetime.now()-start_all).seconds)
# '''