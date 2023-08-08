import pandas as pd
import numpy as np
from pyswarm import pso
from shapely import Point, LineString
import geopandas as gpd
import ast
import multiprocessing
import matplotlib.pyplot as plt

# data_path = '1m_precision_staiLine.csv'
# swath_path = '../Scratch/test_Load_Shp/test_shps/swath_group_1.shp'
# real_swaths = gpd.read_file(swath_path)
# raw_data = pd.read_csv(data_path)
real_swath = gpd.read_file('/Users/outianyi/PythonProject/Coverage_Path_Planning/Scratch/test_Load_Shp/test_shps/new_shp_file/swath_group_1.shp')
raw_data = pd.read_csv('/Users/outianyi/PythonProject/Coverage_Path_Planning/pathline_real_modeling/1m_precision_staiLine_11.csv')

data_array = raw_data.to_numpy()


def fitness_function(lambdas, origin_x, origin_y, height, aspect, slope, labels):
    # 将 aspect 转换为 弧度值
    aspect = np.deg2rad(aspect)
    x_off = lambdas[0] * np.cos(aspect) * slope
    y_off = lambdas[1] * np.sin(aspect) * slope

    new_x = origin_x + x_off
    new_y = origin_y + y_off

    # optimize
    distances = []
    for temp_x, temp_y, label in zip(new_x, new_y, labels):
        temp_point = Point(temp_x, temp_y)
        temp_real_swath = real_swath.geometry.iloc[label]
        distances.append(temp_point.distance(temp_real_swath))
    return np.mean(distances)


lower_bound, upper_bound = [-1, -1], [1, 1]
results = []
EPOCH = 300
SAMPLE_BATCH = 200


def parallel_function(seed_and_i):
    seed, i = seed_and_i
    np.random.seed(seed)
    # random choice sample data
    random_sample = np.random.choice(data_array.shape[0], SAMPLE_BATCH, replace=False)
    random_sample = data_array[random_sample]
    # read data from random sample
    batch_origin_x = random_sample[:, 0]
    batch_origin_y = random_sample[:, 1]
    batch_height = random_sample[:, 2]
    batch_aspect = random_sample[:, 3]
    batch_slope = random_sample[:, 4]
    batch_labels = random_sample[:, 5].astype(np.int32)

    args = (batch_origin_x, batch_origin_y, batch_height, batch_aspect, batch_slope, batch_labels)
    xopt, fopt = pso(fitness_function, lower_bound, upper_bound, args=args)
    # showing temp result
    print('---------------Epoch: {}---------------'.format(i))
    print('Optimized offs: ', xopt)
    print('Fopt: ', fopt)
    return [x for x in xopt] + [fopt]


if __name__ == "__main__":
    print('Start running...')
    # 使用所有可用的CPU核心
    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #     results = pool.map(parallel_function, range(EPOCH))

    num_process = multiprocessing.cpu_count()
    seeds = np.random.randint(0, 1e8, size=EPOCH)

    with multiprocessing.Pool(processes=num_process) as pool:
        results = pool.map(parallel_function, zip(seeds, range(EPOCH)))

    # 显示结果
    print('Showing result...')
    results = np.array(results)
    plt.plot(results[:, -1])
    plt.title('F opt')
    plt.savefig('pso_status.png')
    plt.show()
    results = pd.DataFrame(results)
    results.to_csv('pso_result.csv')

