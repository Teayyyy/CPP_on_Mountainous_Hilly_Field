import numpy as np
import matplotlib as plt
from shapely.geometry import Polygon, LineString
from shapely import affinity
import geopandas as gpd
import math

"""
这个类包含了所有的已测试成功的路径规划的算法、处理方式等
"""


class CPP_Planner_Kit:
    @staticmethod
    def load_fields(path: str):
        """
        加载一个田块区域，并且保存为 geopandas 格式返回
        :param path: 文件路径
        :return: 加载的shp文件
        """
        all_land = gpd.read_file(path)
        # 显示信息
        print("总地块数", len(all_land))
        # 将名称替换为英文，方便保存
        all_land.rename(columns={'周长': 'perimeter', '闭合面积': 'area'}, inplace=True)
        return all_land

    @staticmethod
    def get_single_shp(all_land: gpd.GeoDataFrame, ind: int):
        """
        获取其中一个shp文件，单独读取处理并且可以保存获取其中一个shp文件，单独读取处理并且可以保存
        :param all_land: 所有的地块
        :param ind: 需要单独使用地块的索引
        :return: 能够保存为 shp 和用 .plot 显示的当前地块的单独文件
        """
        if ind < 0 or ind > len(all_land):
            print("地址索引不正确")
            return
        return gpd.GeoDataFrame([all_land.iloc[ind]], crs=all_land.crs)

    @staticmethod
    def get_land_MABR_angle(temp_land) -> float:
        """
        用于计算当前单个田块的最小内接矩形的长边角度，后期用于将单个田块水平便于路径规划
        :param temp_land: 单个田块，如果不是单个田块，就不执行后面的操作
        :return: 田块的角度
        """
        # if len(temp_land) > 1:
        #     print("田块超过了一个！轻输入单个田块\n 如果需要调用单个田块，请调用 get_string_shp()")
        #     return
        # # 找到最小内接矩形， minimum_rotated_rectangle
        # temp_land_polygon = temp_land.geometry[0]
        mabr = temp_land.minimum_rotated_rectangle
        # 计算长边
        longest_edge = None
        max_length = -1
        # 寻找长边
        for i in range(len(mabr.exterior.coords) - 1):
            p1 = mabr.exterior.coords[i]
            p2 = mabr.exterior.coords[i + 1]
            length = LineString([p1, p2]).length
            if length > max_length:
                max_length = length
                longest_edge = LineString([p1, p2])
        # 计算角度
        angle = math.degrees(math.atan2(longest_edge.coords[1][1] - longest_edge.coords[0][1],
                                        longest_edge.coords[1][0] - longest_edge.coords[0][0]))
        print("当前田块角度：", angle)
        return angle


# end class CPP_Planner_Kit ------------------------------------------------------------

"""
这个类包含了目前能够自己实现的路径规划算法
"""


class CPP_Algorithms:
    @staticmethod
    def scanline_algorithm_single(land: gpd.GeoDataFrame, step_size: float):
        """
        将单个地块中的路径按照 “扫描线” 的方法做全覆盖路径规划
        1. 扫描线方法的基本思想是将扫描线从区域的一个边缘沿着特定方向移动，直到达到另一个边缘。在移动的过程中，扫描线将覆盖区域内的部分区域。
           通过在扫描线移动的过程中进行适当的路径选择，可以找到覆盖整个区域的路径
        2. 在算法中会将整个地块旋转到长边“水平”的角度，旋转的点是 polygon.centroid，在规划完路径后，将整个路径按照刚才的 polygon.centroid
           点反向旋转为原来的位置

        * 整个算法需要保证 polygon 在旋转和移动后，能够镜像返回

        :param land: 当前地块，默认是 geopandas.GeoDataFrame，在进行路径规划的时候，直接使用其 Polygon 属性即可
        :param step_size: 扫描线一次移动的距离，对应到实际情况就是一次耕作的宽度
        :return: 返回当前地块的扫描线路径，扫描线路径仅包含在地块内部
        """
        if len(land) > 1:
            print("scanline_algorithm_single: 地块超过了一个，当前地块大小：", len(land))
            return
        # 将 land 中的 polygon 提取出来，保证一次只有一个地块，随即获取相关的信息
        land_polygon = land.iloc[0].geometry
        land_centroid = land_polygon.centroid  # 当前地块的原始坐标位置，保存的中心位置，方便后期镜像回退
        # 获取角度
        land_angle = CPP_Planner_Kit.get_land_MABR_angle(land_polygon)
        # land_angle = 0
        # 置于水平，在后面的路径规划算法中，优先使用该旋转的地块来进行路径规划
        rotated_polygon = affinity.rotate(land_polygon, -land_angle, origin=land_centroid)
        print("地块已置于水平，开始路径规划...")

        # 计算多边形的边界框
        min_x, min_y, max_x, max_y = rotated_polygon.bounds
        # 初始化路径点列表
        path_points = []
        # 默认设置最小 y 点为起始位置
        begin_y = min_y

        # 迭代扫描线
        while begin_y <= max_y:
            # 计算当前扫描线于多边形的焦点
            intersections = []
            for i in range(len(rotated_polygon.exterior.coords) - 1):
                edge = rotated_polygon.exterior.coords[i]
                next_edge = rotated_polygon.exterior.coords[i + 1]

                if (edge[1] <= begin_y and next_edge[1] > begin_y) or (next_edge[1] <= begin_y and edge[1] > begin_y):
                    x = edge[0] + (next_edge[0] - edge[0]) * (begin_y - edge[1]) / (next_edge[1] - edge[1])
                    intersections.append(x)

            # 对交点按照升序排序
            intersections.sort()
            # 遍历交点，生成路径点
            for i in range(0, len(intersections), 2):
                start_x = intersections[i]
                end_x = intersections[i + 1] if i + 1 < len(intersections) else max_x

                # 生成当前行的路径点
                row_points = np.arange(start_x, end_x, step_size)

                # 添加路径点，并水平行驶
                if (begin_y - min_y) % (2 * step_size) == 0:
                    path_points.extend([(x, begin_y) for x in row_points])
                else:  # 反向行驶，两次九十度角的转向
                    row_points = row_points[::-1]
                    path_points.extend([x, begin_y] for x in row_points)
            # 更新扫描线的位置
            begin_y += step_size
        path_line = LineString(path_points)
        path_line = affinity.rotate(path_line, angle=land_angle, origin=land_centroid)
        path_line = gpd.GeoDataFrame(geometry=[path_line], crs=land.crs)
        return path_line
        # end while

    @staticmethod
    def scanline_algorithm_single_no_turn(land: gpd.GeoDataFrame, step_size: float, along_long_edge=False):
        """
        *** 这是上面的算法的改进，取消了转向策略，但是需要将其作为一个单独的问题点来解决
        将单个地块中的路径按照 “扫描线” 的方法做全覆盖路径规划
        1. 扫描线方法的基本思想是将扫描线从区域的一个边缘沿着特定方向移动，直到达到另一个边缘。在移动的过程中，扫描线将覆盖区域内的部分区域。
           通过在扫描线移动的过程中进行适当的路径选择，可以找到覆盖整个区域的路径
        2. 在算法中会将整个地块旋转到长边“水平”的角度，旋转的点是 polygon.centroid，在规划完路径后，将整个路径按照刚才的 polygon.centroid
           点反向旋转为原来的位置

        * 整个算法需要保证 polygon 在旋转后，能够镜像返回

        :param land: 当前地块，默认是 geopandas.GeoDataFrame，在进行路径规划的时候，直接使用其 Polygon 属性即可
        :param step_size: 扫描线一次移动的距离，对应到实际情况就是一次耕作的宽度
        :param along_long_edge: 是否按照多边形的“长边”为路径进行路径规划，默认是打开的
        :return: 返回当前地块的扫描线路径，扫描线路径仅包含在地块内部
        """
        if len(land) > 1:
            print("scanline_algorithm_single: 地块超过了一个，当前地块大小：", len(land))
            return

        # 将 land 中的 polygon 提取出来，保证一次只有一个地块，随即获取相关的信息
        land_polygon = land.iloc[0].geometry
        land_centroid = land_polygon.centroid  # 当前地块的原始坐标位置，保存的中心位置，方便后期镜像回退
        if along_long_edge:
            land_angle = 0
        else:
            land_angle = CPP_Planner_Kit.get_land_MABR_angle(land_polygon)  # 获取角度

        # 置于水平，在后面的路径规划算法中，优先使用该旋转的地块来进行路径规划
        rotated_polygon = affinity.rotate(land_polygon, -land_angle, origin=land_centroid)
        if along_long_edge:
            print("根据田块长边开始路径规划...")
        else:
            print("水平方向开始路径规划")

        # 计算多边形的边界框
        min_x, min_y, max_x, max_y = rotated_polygon.bounds

        # 初始化路径线列表
        path_lines = []

        # 迭代扫描线
        for y in np.arange(min_y, max_y + step_size, step_size):
            row_points = []
            for i in range(len(rotated_polygon.exterior.coords) - 1):
                edge = rotated_polygon.exterior.coords[i]
                next_edge = rotated_polygon.exterior.coords[i + 1]

                if (edge[1] <= y and next_edge[1] > y) or (next_edge[1] <= y and edge[1] > y):
                    x = edge[0] + (next_edge[0] - edge[0]) * (y - edge[1]) / (next_edge[1] - edge[1])
                    row_points.append((x, y))

            # 创建扫描线的 LineString 对象
            if len(row_points) > 1:
                path_line = LineString(row_points)
                path_line = affinity.rotate(path_line, land_angle, origin=land_centroid)
                path_lines.append(path_line)

        # 创建 GeoDataFrame 对象
        path_gdf = gpd.GeoDataFrame(geometry=path_lines, crs=land.crs)

        return path_gdf


# end class CPP_Algorithms ------------------------------------------------------------

"""
这个类包含了能够自己实现的CPP算法的优化方式
"""


class Algorithm_Optimizers:
    pass

# end class Algorithm_Optimizers ------------------------------------------------------------
