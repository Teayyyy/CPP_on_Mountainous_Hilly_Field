import pandas as pd
import numpy as np
from pyswarm import pso
from shapely import Point, LineString
import geopandas as gpd
import ast
import multiprocessing
import matplotlib.pyplot as plt

"""
这份脚本用来测试多个 lambdas 参数下，模型拟合的程度
添加数据： origin_x, origin_y, dem, aspect, slope, label, curvature
"""
# 一些农机数据
vehicle_weight = 1980 + 342.5  # 拖拉机 + 耕作机械 重量（kg）
vehicle_wheel_width = 1440  # 农机轮距（mm）

# data_path = '1m_precision_staiLine.csv'
# swath_path = '../Scratch/test_Load_Shp/test_shps/swath_group_1.shp'
# real_swaths = gpd.read_file(swath_path)
# raw_data = pd.read_csv(data_path)
# real_swath = gpd.read_file('/Users/outianyi/PythonProject/Coverage_Path_Planning/Scratch/test_Load_Shp/test_shps/new_shp_file/swath_group_1.shp')
# raw_data = pd.read_csv('/Users/outianyi/PythonProject/Coverage_Path_Planning/pathline_real_modeling/1m_precision_staiLine_DASC.csv')
# # 进行当前所有地块路径优化时，使用下面的路径
real_swath = gpd.read_file('GIS_data/路径规划优化用数据/all_swaths/all_swaths_group.shp')
# ------------------------------------------------------------
# 全部地块情况
# raw_data = pd.read_csv('1m_preci_sample_DASC_all_land.csv')
# raw_data = pd.read_csv('norm_mean_1m_preci_sample_DASC_all_land.csv')
# 分离地块情况
# raw_data = pd.read_csv('1m_preci_sample_DASC_sep_land_0.csv')
# raw_data = pd.read_csv('norm_mean_1m_preci_sample_DASC_sep_land_0.csv')
# raw_data = pd.read_csv('norm_median_1m_preci_sample_DASC_sep_land_0.csv')
# raw_data = pd.read_csv(r'sep_land_datas/norm_mean_1m_preci_sample_DASC_sep_land_2.csv')
# raw_data = pd.read_csv(r'norm_mean_simp_1m_preci_sample_DASC_sep_land_0.csv')
# raw_data = pd.read_csv(r'1m_preci_sample_DASC_sep_land_0.csv')
# raw_data = pd.read_csv('exhalf_1m_preci_sample_DASC_sep_land_0.csv')
# raw_data = pd.read_csv('exhalf_median_1m_preci_sample_DASC_sep_land_0.csv')
raw_data = pd.read_csv(r'exhalf_only_small_1m_preci_sample_DASC_sep_land_0_0.csv')


data_array = raw_data.to_numpy()


def fitness_function(lambdas, origin_x, origin_y, height, aspect, slope, curvature, labels):
    # 将 aspect 转换为 弧度值
    aspect = np.deg2rad(aspect)
    x_off = lambdas[0] * np.cos(aspect) * slope + lambdas[2] * curvature
    y_off = lambdas[1] * np.sin(aspect) * slope + lambdas[3] * curvature

    new_x = origin_x + x_off
    new_y = origin_y + y_off

    # optimize
    distances = []
    for temp_x, temp_y, label in zip(new_x, new_y, labels):
        temp_point = Point(temp_x, temp_y)
        temp_real_swath = real_swath.geometry.iloc[label]
        distances.append(temp_point.distance(temp_real_swath))
    return np.mean(distances)


# 拟真模型
def fitness_function_model(lambdas, origin_x, origin_y, height, aspect, slope, curvature, labels):
    """
    * 使用一个新的模型用来拟合路径
        1. 地形数据：aspect, slope, curvature
        2. 硬件数据：vehicle_weight, wheel_width
    模拟农机在当前点所受的静摩擦力方向和稳定性，以此达到真实的路径
    lambdas: 包括 lambdas[0: 3]，用来拟合上述两个特性的参数
    注意：所有的关于角度的变量，均为弧度制 radius
    # 公式作用：
        np.cos(aspect) * slope * vehicle_weight AND np.sin(aspect) * slope * vehicle_weight: 静摩擦力方向
        vehicle_wheel_width / curvature: 衡量稳定性
    """
    aspect = np.deg2rad(aspect)
    # slope = np.deg2rad(slope)
    # x_off = lambdas[0] * np.cos(aspect) * slope * vehicle_weight + lambdas[2] * (curvature / vehicle_wheel_width)
    # y_off = lambdas[1] * np.sin(aspect) * slope * vehicle_weight + lambdas[3] * (curvature / vehicle_wheel_width)
    x_off = lambdas[0] * np.cos(aspect) * slope * vehicle_weight + lambdas[2] * (np.abs(curvature) / vehicle_wheel_width)
    y_off = lambdas[1] * np.sin(aspect) * slope * vehicle_weight + lambdas[3] * (np.abs(curvature) / vehicle_wheel_width)

    new_x = origin_x + x_off
    new_y = origin_y + y_off

    # optimize
    distances = []
    for temp_x, temp_y, label in zip(new_x, new_y, labels):
        temp_point = Point(temp_x, temp_y)
        temp_real_swath = real_swath.geometry.iloc[label]
        distances.append(temp_point.distance(temp_real_swath))
    # print(np.mean(distances))
    return np.mean(distances)


def fitness_function_model_no_triangle(lambdas, origin_x, origin_y, height, aspect, slope, curvature, labels):
    """
    * 使用一个新的模型用来拟合路径
        1. 地形数据：aspect, slope, curvature
        2. 硬件数据：vehicle_weight, wheel_width
    模拟农机在当前点所受的静摩擦力方向和稳定性，以此达到真实的路径
    lambdas: 包括 lambdas[0: 3]，用来拟合上述两个特性的参数
    注意：所有的关于角度的变量，均为弧度制 radius
    # 公式作用：
        np.cos(aspect) * slope * vehicle_weight AND np.sin(aspect) * slope * vehicle_weight: 静摩擦力方向
        vehicle_wheel_width / curvature: 衡量稳定性
    """
    aspect = np.deg2rad(aspect)
    slope = np.deg2rad(slope)
    # x_off = lambdas[0] * aspect * slope * vehicle_weight + lambdas[2] * (curvature / vehicle_wheel_width)
    # y_off = lambdas[1] * aspect * slope * vehicle_weight + lambdas[3] * (curvature / vehicle_wheel_width)
    x_off = lambdas[0] * aspect * slope * vehicle_weight + lambdas[2] * (np.abs(curvature) / vehicle_wheel_width)
    y_off = lambdas[1] * aspect * slope * vehicle_weight + lambdas[3] * (np.abs(curvature) / vehicle_wheel_width)

    new_x = origin_x + x_off
    new_y = origin_y + y_off

    # optimize
    distances = []
    for temp_x, temp_y, label in zip(new_x, new_y, labels):
        temp_point = Point(temp_x, temp_y)
        temp_real_swath = real_swath.geometry.iloc[label]
        distances.append(temp_point.distance(temp_real_swath))
    # print(np.mean(distances))
    return np.mean(distances)


def fitness_function_model_no_curvature(lambdas, origin_x, origin_y, height, aspect, slope, curvature, labels):
    aspect = np.deg2rad(aspect)
    # slope_rad = np.deg2rad(slope)
    x_off = lambdas[0] * np.cos(aspect) * vehicle_weight * slope
    y_off = lambdas[1] * np.sin(aspect) * vehicle_weight * slope

    new_x = origin_x + x_off
    new_y = origin_y + y_off

    # optimize
    distances = []
    for temp_x, temp_y, label in zip(new_x, new_y, labels):
        temp_point = Point(temp_x, temp_y)
        temp_real_swath = real_swath.geometry.iloc[label]
        # distances.append(temp_point.distance(temp_real_swath))
        distances.append(temp_real_swath.distance(temp_point))
    return np.mean(distances)


def fitness_function_model_4(lambdas, origin_x, origin_y, height, aspect, slope, curvature, labels):
    aspect = np.deg2rad(aspect)
    # new_x = origin_x + np.cos(aspect) * (
    #             lambdas[0] * slope + lambdas[1] * curvature) * vehicle_wheel_width / vehicle_weight
    # new_y = origin_y + np.sin(aspect) * (
    #             lambdas[2] * slope + lambdas[1] * curvature) * vehicle_wheel_width / vehicle_weight

    # 尝试将 curvature更为绝对值化
    new_x = origin_x + np.cos(aspect) * (lambdas[0] * slope + lambdas[1] * np.abs(curvature)) * vehicle_wheel_width / vehicle_weight
    new_y = origin_y + np.sin(aspect) * (lambdas[2] * slope + lambdas[1] * np.abs(curvature)) * vehicle_wheel_width / vehicle_weight

    # optimize
    distances = []
    for temp_x, temp_y, label in zip(new_x, new_y, labels):
        temp_point = Point(temp_x, temp_y)
        temp_real_swath = real_swath.geometry.iloc[label]
        distances.append(temp_real_swath.distance(temp_point))
    return np.mean(distances)


# lower_bound, upper_bound = [-1, -1, -0.05, -0.05], [1, 1, 0.05, 0.05]
# lower_bound, upper_bound = [-1, -1, -1, -1], [1, 1, 1, 1]
lower_bound, upper_bound = [-1, -1], [1, 1]
results = []
# ------------------------------
# EPOCH = 100
# SAMPLE_BATCH = 400
# ------------------------------
# EPOCH = 10
# SAMPLE_BATCH = 1000
# small land only --------------
EPOCH = 20
SAMPLE_BATCH = 600
# for demonstration --------------
EPOCH = 10
SAMPLE_BATCH = 160


# SAMPLE_BATCH = 400


def parallel_function(seed_and_i):
    seed, i = seed_and_i
    print('Epoch{}'.format(i), 'start')
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
    batch_curvature = random_sample[:, 5]
    batch_labels = random_sample[:, 6].astype(np.int32)

    args = (batch_origin_x, batch_origin_y, batch_height, batch_aspect, batch_slope, batch_curvature, batch_labels)
    # 用于最先的优化，没有公式
    # xopt, fopt = pso(fitness_function, lower_bound, upper_bound, args=args)
    # 用于新的优化方式，添加了农机重量和农机轮距
    # xopt, fopt = pso(fitness_function_model, lower_bound, upper_bound, args=args)
    # xopt, fopt = pso(fitness_function_model_no_triangle, lower_bound, upper_bound, args=args)
    xopt, fopt = pso(fitness_function_model_no_curvature, lower_bound, upper_bound, phip=0.5, phig=0.5, args=args)
    # xopt, fopt = pso(fitness_function_model_4, lower_bound, upper_bound, phip=0.5, phig=0.5, args=args)

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

    # num_process = multiprocessing.cpu_count()
    # When test, only use 2 processor is fine, and don't run too much epoch
    num_process = multiprocessing.cpu_count()
    seeds = np.random.randint(0, 1e8, size=EPOCH)

    print('Model fitting begin...')
    with multiprocessing.Pool(processes=num_process) as pool:
        results = pool.map(parallel_function, zip(seeds, range(EPOCH)))

    # 显示结果
    print('Showing result...')
    results = np.array(results)
    plt.plot(results[:, -1])
    plt.title('F opt')
    # plt.savefig('pso_status.png')
    plt.show()
    # results = pd.DataFrame(results)
    # results.to_csv('pso_result.csv')
