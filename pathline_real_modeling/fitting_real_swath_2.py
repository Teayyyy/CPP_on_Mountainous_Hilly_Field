import numpy as np
import pandas as pd
import geopandas as gpd
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras import layers
import ast
from shapely import Point, LineString


class Swath_Dense_Model(Model):
    def __init__(self):
        super().__init__()
        self.den0 = layers.Dense(27, activation=tf.nn.relu)
        self.drop0 = layers.Dropout(0.4)
        self.den1 = layers.Dense(16, activation=tf.nn.relu)
        self.drop1 = layers.Dropout(0.3)
        self.den2 = layers.Dense(8, activation=tf.nn.relu)
        self.drop2 = layers.Dropout(0.2)
        self.den3 = layers.Dense(2)

    def call(self, inputs, **kwargs):
        inputs = self.den0(inputs)
        inputs = self.drop0(inputs)
        inputs = self.den1(inputs)
        inputs = self.drop1(inputs)
        inputs = self.den2(inputs)
        inputs = self.drop2(inputs)
        return self.den3(inputs)


class Data_Loader:
    @staticmethod
    def load_csv():
        data_path = '1m_precision_staiLine.csv'
        swath_path = '../Scratch/test_Load_Shp/test_shps/swath_group_1.shp'
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


real_swath = gpd.read_file('../Scratch/test_Load_Shp/test_shps/swath_group_1.shp').geometry.tolist()


def m_loss(labels, y_pred):
    print('------------------------------ In Loss ------------------------------')
    print('Label shape: ', labels.shape)
    print(labels)
    print('y_pred shape: ', y_pred.shape)
    print(y_pred[:, :, 0])
    # print(tf.get_static_value(labels))

    origin_x = labels[:, 0]
    origin_y = labels[:, 1]
    # new_x = y_pred[:, 0] + origin_x
    new_x = y_pred[:, :, 0] + tf.cast(origin_x, dtype=tf.float32)
    # new_y = y_pred[:, 1] + origin_y
    new_y = y_pred[:, :, 1] + tf.cast(origin_y, dtype=tf.float32)
    real_line_ind = labels[:, 2]

    def cal_distance(line, x, y):
        point = Point(x, y)
        return point.distance(line)

    def calc_distance(lines, xs, ys):
        dists = []
        for i in range(len(lines)):
            # x, y = xs[i], ys[i]
            point = Point(xs[i], ys[i])
            # line = lines[i]
            line = real_swath[lines[i]]
            dists.append(point.distance(line))
        return np.array(dists).mean()



    # dists = []
    # for i in range(len(y_pred)):
    #     print("i: ", i)
    #     print(new_x[i])
    #     print(real_line_ind)
    #     # for i in range(tf.shape(y_pred[0])):
    #     dist = tf.py_function(cal_distance, [real_swath[real_line_ind[i]], new_x[i], new_y[i]], tf.float32)
    #
    #     dists.append(dist)
    dists = tf.py_function(calc_distance, real_line_ind, tf.float32)
    return dists
    # return tf.reduce_mean(dists)


def trains():
    model = Swath_Dense_Model()
    model.compile(optimizer='adam', loss=m_loss)
    datas, labels = Data_Loader.load_csv()

    train_proportion = int(0.8 * int(len(datas)))
    batch_size = 200

    datas = datas[:, :, np.newaxis]
    datas = tf.transpose(datas, perm=[0, 2, 1])
    print('Data shape: ', datas.shape)
    print('Label shape: ', labels.shape)

    train_data, train_labels = datas[: train_proportion], labels[: train_proportion]
    test_data, test_labels = datas[train_proportion:], labels[train_proportion:]

    train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(batch_size)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = m_loss

    @tf.function
    def train_step(temp_datas, temp_labels):
        with tf.GradientTape() as tape:
            print("data shape: ", temp_datas.shape)
            predictions = model(temp_datas)
            print("prediction shape: ", predictions.shape)
            loss = loss_object(temp_labels, predictions)
        gradient = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))

        print("current loss: ", loss)

    @tf.function
    def test_step(temp_datas, temp_labels):
        predictions = model(temp_datas)
        temp_loss = loss_object(temp_labels, predictions)

        print("current test loss: ", temp_loss)
        pass

    print("Start Training...")
    epochs = 100
    losses = []

    for epoch in range(epochs):
        for input_data, input_label in train_ds:
            train_step(input_data, input_label)
        for input_data, input_label in test_ds:
            test_step(input_data, input_label)

    pass


trains()
