import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import geopandas as gpd

"""
这是最基础的路径规划算法，目前只能够解决非凹边形的路径规划，同时存在许多问题：
    * 当前的路径是直来直去，如果遇到曲面的地块，会导致耕作路径覆盖不全
    * 路径转弯的时候，没有考虑机器的转向半径，导致路径规划的结果不符合实际
    * 在斜边的时候，有的路径目前是有的直角有的锐角，这个也是不符合实际的
"""



def calculate_coverage_path(polygon, step_size):
    # 计算多边形的边界框
    min_x, min_y, max_x, max_y = polygon.bounds

    # 初始化路径点列表
    path_points = []

    # 设置扫描线的起始位置
    y = min_y

    # 迭代扫描线
    while y <= max_y:
        # 计算当前扫描线与多边形的交点
        intersections = []
        for i in range(len(polygon.exterior.coords) - 1):
            edge = polygon.exterior.coords[i]
            next_edge = polygon.exterior.coords[i + 1]

            if (edge[1] <= y and next_edge[1] > y) or (next_edge[1] <= y and edge[1] > y):
                x = edge[0] + (next_edge[0] - edge[0]) * (y - edge[1]) / (next_edge[1] - edge[1])
                intersections.append(x)

        # 对交点按升序进行排序
        intersections.sort()

        # 遍历交点，生成路径点
        for i in range(0, len(intersections), 2):
            start_x = intersections[i]
            end_x = intersections[i+1] if i+1 < len(intersections) else max_x

            # 生成当前行的路径点
            row_points = np.arange(start_x, end_x, step_size)

            # 添加路径点，水平行驶
            if (y - min_y) % (2 * step_size) == 0:
                path_points.extend([(x, y) for x in row_points])
            else:  # 反向行驶，两次九十度角转向
                row_points = row_points[::-1]
                path_points.extend([(x, y) for x in row_points])

        # 更新扫描线的位置
        y += step_size

    # 返回路径点
    return path_points

def get_shp_polygon():
    shp_file = gpd.read_file('test_Load_Shp/shp_file/村地1地1_旧_第二个.shp')
    print("Shape Info: ")
    for line in shp_file.geometry:
        print(line)
    polygon = shp_file.geometry[0]
    return polygon


# 定义不规则的多边形地块
# polygon = Polygon([(2, 1), (4, 2.5), (3, 5), (1, 4), (0.5, 2)])
polygon = get_shp_polygon()
scale_factor = 1
scaled_coords = np.array(polygon.exterior.coords) * scale_factor
scaled_polygon = Polygon(scaled_coords)


# 设置路径间距
step_size = 1

# 调用路径规划函数，生成路径点
path_points = calculate_coverage_path(scaled_polygon, step_size)

# 绘制路径和多边形地块
plt.figure(figsize=(8, 6))
plt.plot(*scaled_polygon.exterior.xy, 'k-', label='Polygon')
plt.plot(*zip(*path_points), 'b-', label='Coverage Path')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Complete Coverage Path Planning')
plt.legend()
plt.grid(True)
plt.show()
