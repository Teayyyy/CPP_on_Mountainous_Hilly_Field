"""
这个脚本是 训练 模块
"""
import numpy
import numpy as np
import pandas as pd
import geopandas as gpd
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras import layers
import ast
from shapely import Point, LineString

data_path = '1m_precision_staiLine.csv'
swath_path = '../Scratch/test_Load_Shp/test_shps/swath_group_1.shp'
real_swath = gpd.read_file(swath_path).geometry.tolist()


def data_loader(data_path, swath_path):
    print('Loading data from: ', data_path, " | ", swath_path)
    raw_data = pd.read_csv(data_path)
    real_swath_data = gpd.read_file(swath_path)
    print("一共 {} 个数据".format(len(raw_data)))
    # 处理数据，将原本 string 保存的字符串恢复为 list
    raw_data['height'] = raw_data['height'].apply(ast.literal_eval)
    raw_data['aspect'] = raw_data['aspect'].apply(ast.literal_eval)
    raw_data['slope'] = raw_data['slope'].apply(ast.literal_eval)
    # 将 label 替换为 real_swath_data.geometry 对应的 LineString
    # print('Translating index to LineString...')
    # raw_data['label'] = raw_data['label'].apply(lambda x: real_swath_data.geometry[x])
    # 选取其中 height, aspect, slope 作为 train_data，origin_x, origin_y, label 作为输出
    print('Selecting data...')
    train_data = raw_data[['height', 'aspect', 'slope']]
    train_label = raw_data[['origin_x', 'origin_y', 'label']]
    # 将内部的 9 * 3 的 list 转换为 27 * 1 的 list
    print('Reshaping data...')
    train_data = np.array(train_data.values.tolist()).reshape(-1, 27)
    train_label = train_label.to_numpy()
    print('Data example: ')
    print(train_data[: 2])
    print(train_label[: 2])
    return train_data, train_label


def m_loss(labels, y_pred):
    origin_x = labels[:, 0]
    origin_y = labels[:, 1]
    new_x = y_pred[:, 0] + origin_x
    new_y = y_pred[:, 1] + origin_y
    real_line_ind = labels[:, 2]

    print('------------------------------ In Loss ------------------------------')
    print('Label shape: ', labels.shape)
    print(labels)
    print('y_pred shape: ', y_pred.shape)
    print(y_pred)

    def cal_distance(line, x, y):
        point = Point(x, y)
        return point.distance(line)

    dists = []
    for i in range(len(y_pred)):
        print("i: ", i)
        print(new_x[i])
        print(real_line_ind)
        # for i in range(tf.shape(y_pred[0])):
        dist = tf.py_function(cal_distance, [real_swath[real_line_ind[i]], new_x[i], new_y[i]], tf.float32)
        dists.append(dist)

    return tf.reduce_mean(dists)


datas, labels = data_loader(data_path, swath_path)
print('Shape of datas: ', datas.shape)
print('Shape of labels: ', labels.shape)

# ------------------------------ 训练模块 ------------------------------
model = tf.keras.Sequential([
    layers.Dense(27, activation='relu', input_shape=(27,)),
    layers.Dropout(rate=0.4),
    layers.Dense(16, activation='relu'),
    layers.Dropout(rate=0.3),
    layers.Dense(8, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(2)
])

model.compile(optimizer='adam', loss=m_loss, metrics=['mae'])

history = model.fit(datas, labels, epochs=1, batch_size=2, validation_split=0.2)
