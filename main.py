import numpy as np
import matplotlib.pyplot as plt

"""
这是一个基于扫描线的路径规划算法，目前只能够解决非凹边形的路径规划
calculate_coverage_path 函数的目标是根据给定的不规则多边形地块，生成一个往复式的全覆盖路径。下面是函数的详细步骤：

1. 计算多边形地块的边界框（最小x、最小y、最大x、最大y）。
2. 初始化一个空的路径点列表，用于存储生成的路径点。
3. 设置扫描线的起始位置为最小y值。
4. 迭代扫描线，直到达到最大y值：
    * 计算当前扫描线与多边形的交点：遍历多边形的边，如果边的起始y坐标和结束y坐标与扫描线的位置相交，计算交点的x坐标，并将其添加到交点列表中。
    * 对交点列表按升序进行排序，以确保路径点按顺序生成。
    * 遍历交点列表中的交点，生成当前行的路径点：
        * 如果当前行的位置与起始y值之间的距离是路径间距的整数倍，说明当前行是水平行驶的行，从左到右生成路径点。
        * 否则，当前行是反向行驶的行，从右到左生成路径点，以实现往复式路径规划。
    5. 更新扫描线的位置，将其向上移动一个步长（路径间距）。
    6. 返回生成的路径点列表。
这个算法的核心思想是通过迭代扫描线，计算交点并按顺序生成路径点。在每一行中，根据行的位置来确定行驶的方向，以实现往复式的路径规划。

请注意，上述代码中没有考虑转向半径和细节处理，它提供了一个基本的框架，您可以根据实际需求进行进一步修改和优化。
"""

def calculate_coverage_path(field_shape, turn_radius, step_size):
    # 初始化路径点列表
    path_points = []

    # 遍历地块形状的边界点
    for i in range(len(field_shape)):
        # 当前点和下一个点的坐标
        current_point = field_shape[i]
        next_point = field_shape[(i + 1) % len(field_shape)]

        # 计算当前边的长度和方向向量
        edge_length = np.linalg.norm(next_point - current_point)
        edge_direction = (next_point - current_point) / edge_length

        # 计算当前边上的路径点
        num_points = int(edge_length / step_size) + 1
        edge_points = np.linspace(current_point, next_point, num_points)

        # 添加路径点到路径列表
        path_points.extend(edge_points)

    # 返回路径点列表
    return path_points

def visualize_coverage_path(field_shape, path_points):
    # 绘制地块形状
    plt.plot(field_shape[:, 0], field_shape[:, 1], 'b-', linewidth=3)
    plt.fill(field_shape[:, 0], field_shape[:, 1], 'lightblue')

    # 绘制路径
    path_x = [point[0] for point in path_points]
    path_y = [point[1] for point in path_points]
    plt.plot(path_x, path_y, 'r--', linewidth=1)

    # 设置坐标轴范围
    plt.xlim(min(field_shape[:, 0])-1, max(field_shape[:, 0])+1)
    plt.ylim(min(field_shape[:, 1])-1, max(field_shape[:, 1])+1)

    # 添加图例和标题
    plt.legend(['Field Shape', 'Coverage Path'])
    plt.title('Coverage Path Planning')

    # 显示图形
    plt.show()

# 定义农田地块形状，示例为一个不规则的多边形
field_shape = np.array([(1, 0), (2, 0), (3, 1), (3, 3), (2, 4), (0, 4), (0, 0)])

# 设置机器转向半径和步长
turn_radius = 2.0
step_size = 1

# 调用路径规划函数，生成路径点
path_points = calculate_coverage_path(field_shape, turn_radius, step_size)

# 可视化路径规划结果
visualize_coverage_path(field_shape, path_points)
