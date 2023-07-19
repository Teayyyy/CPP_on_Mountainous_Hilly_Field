import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry
from shapely.geometry import Polygon, LineString
from shapely import affinity
from shapely import ops
import geopandas as gpd
import math
from math import pi
import queue

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
    def get_land_MABR_angle(temp_land: Polygon) -> float:
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

    @staticmethod
    def extend_shapely_line(line: LineString, scale_factor=10, extend_mode=2):
        """
        用于延长一根 shapely 中的 LineString 线，延长的尺度为 scale_factor 倍
        :param line: 需要延长的线段，一定要是 shapely LineString 类型
        :param scale_factor: 缩放的倍数（通常是放大），若是要缩小，则为小数，默认为 10
        :param extend_mode: 延长的模式，0 为线段的“前一个点”，1 为线段的“后一个”点，2 为双向延长（默认）
                注意：当分割线长度没有超过多边形的边时，分割不会进行
        :return: 延长后的线段
        """
        # 检查当前的 line 是否是 LineString 类型
        if not isinstance(line, LineString):
            print("当前的 line 不是 LineString 类型")
            return
        # print("Extending lines...")
        # 获取线段的起始点和结束点的坐标
        x1, y1 = line.coords[0]
        x2, y2 = line.coords[1]
        # 计算线段的方向向量
        dx = x2 - x1
        dy = y2 - y1
        # 缩放
        extended_dx = dx * scale_factor
        extended_dy = dy * scale_factor
        # 按照延长的方式来选择
        extended_x1 = x1
        extended_y1 = y1
        extended_x2 = x2
        extended_y2 = y2
        if extend_mode == 0 or extend_mode == 2:
            extended_x1 -= extended_dx
            extended_y1 -= extended_dy
        if extend_mode == 1 or extend_mode == 2:
            extended_x2 += extended_dx
            extended_y2 += extended_dy

        return LineString([(extended_x1, extended_y1), (extended_x2, extended_y2)])

    @staticmethod
    def split_polygon_through_1edge(split_polygon: Polygon, split_line: LineString, line_scale_factor=10):
        """
        将一个多边形（田块）按照其上的一条边分割，分割成多个多边形
        在分割的时候，可以直接输入的边，会被自动延长10倍，当然也可以手动设定
        :param split_polygon: 被分割的多边形，一定是 shapely Polygon 类型
        :param split_line: 用于分割的线，一定是 shapely LineString 类型
        :param line_scale_factor: 分割线段缩放的大小，和 extend_shapely_line() 为同一个变量
                注意：当分割线长度没有超过多边形的边时，分割不会进行
        :return:
        """
        # # 获取当前多边形的凸包上的所有的线，存储成单独的一根根线
        # convex_hull_lines = []
        # for i in range(len(split_polygon.convex_hull.exterior.coords) - 1):
        #     convex_hull_lines.append(LineString([split_polygon.convex_hull.exterior.coords[i],
        #                                          split_polygon.convex_hull.exterior.coords[i+1]]))
        # # 获取多边形本身的所有线，同样存储为单独的一根根线
        # polygon_lines = []
        # for i in range(len(split_polygon.exterior.coords) - 1):
        #     polygon_lines.append(LineString([split_polygon.exterior.coords[i], split_polygon.exterior.coords[i+1]]))
        #
        # # 获取所有不再凸包上的线
        # not_on_convex_hull_lines = []
        # for polygon_line in polygon_lines:
        #     if polygon_line not in convex_hull_lines:
        #         not_on_convex_hull_lines.append(polygon_line)

        # print("Num of convex_hull: ", len(convex_hull_lines))
        # print("Num of Polygon lines: ", len(polygon_lines))
        # print("Not on Convex: ", len(not_on_convex_hull_lines))

        # 开始处理线以分割多边形
        split_line = CPP_Planner_Kit.extend_shapely_line(split_line, scale_factor=line_scale_factor)

        already_split_polygon = ops.split(split_polygon, split_line)
        # print("当前分割多边形个数: ", len(already_split_polygon.geoms))
        return already_split_polygon

    @staticmethod
    def get_non_convex_edges(polygon: shapely.geometry.Polygon) -> [LineString]:
        """
        输入一个多边形，返回其非凸包上的边
        :param polygon: 输入的多边形
        :return: 非凸包上的边, 为存储了所有非凸包边的列表 [LineString]
        """
        # 获取当前多边形的凸包上的所有的线，存储成单独的一根根线
        convex_hull_lines = []
        for i in range(len(polygon.convex_hull.exterior.coords) - 1):
            convex_hull_lines.append(LineString([polygon.convex_hull.exterior.coords[i],
                                                 polygon.convex_hull.exterior.coords[i + 1]]))
        # 获取多边形本身的所有线，同样存储为单独的一根根线
        polygon_lines = []
        for i in range(len(polygon.exterior.coords) - 1):
            polygon_lines.append(LineString([polygon.exterior.coords[i], polygon.exterior.coords[i + 1]]))

        # 获取所有不再凸包上的线
        not_on_convex_hull_lines = []
        for polygon_line in polygon_lines:
            if polygon_line not in convex_hull_lines:
                not_on_convex_hull_lines.append(polygon_line)

        print("Num of convex_hull: ", len(convex_hull_lines))
        print("Num of Polygon lines: ", len(polygon_lines))
        print("Not on Convex: ", len(not_on_convex_hull_lines))

        return not_on_convex_hull_lines

    @staticmethod
    def show_polygon_edge_length(polygon, color='b', text_color='w'):
        """
        显示当前多边形每一条边的长度，必须保证仅输入一个多边形
        :param polygon: 显示长度的多边形
        :param color: 多边形显示的颜色，默认为蓝色
        :param text_color: 文字显示颜色，默认为白色
            * 颜色的显示参考 matplotlib，都是加了一层借口 :D
        """
        x, y = polygon.exterior.xy
        # 存贮坐标的表
        coords_x = [temp_x for temp_x in x]
        coords_x.append(x[0])
        coords_y = [temp_y for temp_y in y]
        coords_y.append(y[0])

        # 计算长度
        edge_lengths = []
        for i in range(len(coords_x) - 1):
            length = ((coords_x[i] - coords_x[i + 1]) ** 2 + (coords_y[i] - coords_y[i + 1]) ** 2) ** 0.5
            edge_lengths.append(length)

        # 绘制图形
        plt.plot(x, y, '{}-'.format(color), label="Polygon")
        for i, length in enumerate(edge_lengths):
            # 显示文字的位置为中点
            edge_midpoint = [(coords_x[i] + coords_x[i + 1]) / 2, (coords_y[i] + coords_y[i + 1]) / 2]
            plt.text(edge_midpoint[0], edge_midpoint[1], f'{length:.2f}', ha='center', va='center')
        plt.xlabel("Longitude"), plt.ylabel("Latitude")
        plt.title("Unit: meter(s)")
        plt.show()

    @staticmethod
    def show_geometry_collection(collections: shapely.geometry.GeometryCollection, color='r'):
        """
        显示 shapely GeometryCollection，就是为了方便，默认的显示颜色是红色
        :param collections: 需要显示的几何集合
        :param color: 显示的颜色
        """
        fig, ax = plt.subplots()
        for geom in collections.geoms:
            x, y = geom.exterior.xy
            ax.plot(x, y, color)

    @staticmethod
    def is_polygon_convex_area_same(polygon: Polygon, tolerance=0.1) -> bool:
        """
        判断当前的多边形是否*接近*凸边形，判断的依据为两者面积只差和原多边形的面积只比，如果超过了 10% 则判定为不相同（非凸边形）
        :param polygon: 原多边形
        :param tolerance: （凸包面积 - 原多边形面积）/ 原多边形面积 的值是否超过 tolerance（百分比）
        :return: bool 是否相等
        """
        convex = polygon.convex_hull
        if (convex.area - polygon.area) / polygon.area < tolerance:
            return True
        else:
            return False

    @staticmethod
    def split_polygon_by_largest_area(polygon: Polygon, tolerance=0.03) -> [Polygon]:
        """
        将一个田块进行凸分割，分割的线是当前田块的非凸边，是否采用当前边作为分割边的依据是当前边所分开的凸边形是否是所能够分出的最大凸边形
        * 使用面积差判断凸边形的依据：减少部分田块边缘“不太平整”时，对多边形分割的影响，可以参考 test_split_convex_2.ipynb
        :param polygon: 被分割的多边形（田块），如果田块本身就是凸边形，那么会直接返回当前凸边形
        :param tolerance: 由于本方法判断一个多边形是否为凸边形的原则是：当前多边形和其凸包的面积差是否超过了阈值 tolerance，超过则不是凸边
               形，不超过就当作当前边是凸边形
        :return: list[Polygon]
        """
        print(f"split_polygon_by_largest_area： 开始进行田块分割，当前误差范围 tolerance = {tolerance}")
        # 存放凸边形 和 未分割为凸边形的凹边形，因为需要循环处理知道当前的多边形倍全部分为 误差范围内的凸边形，因此用队列维护
        convex = []
        concave = queue.Queue()
        # 首先判断当前的多边形是否满足 凸边形，满足就可以不用进行后面的凸边形分割了
        if CPP_Planner_Kit.is_polygon_convex_area_same(polygon, tolerance=tolerance):
            convex.append(polygon)
        else:
            concave.put(polygon)

        # 思路是，将所有的凹边形都处理为凸边形位置
        while not concave.empty():
            temp_concave = concave.get()
            split_group = []  # 用于存放所有多边形的非凸边分割后的多边形结果，其中一个元素就是一条边的分割结果
            # 尝试将凹边形按照不同的边分开
            non_convex_edges = CPP_Planner_Kit.get_non_convex_edges(temp_concave)
            for edge in non_convex_edges:
                split_group.append(CPP_Planner_Kit.split_polygon_through_1edge(temp_concave, edge))

            # 保存当前找到的分割结果中，分出的最大的凸边形的面积，以及其所在组的索引
            group_wise_max_area = 0
            group_wise_max_index = -1

            # 寻找每一个通过非凸边分出的组，其内的最大凸边形的面积，意在找到当前所有非凸边中能够分出的最大凸边形
            for i in range(len(split_group)):
                group_inside_max_area = 0
                group = split_group[i]
                for temp_geom in group.geoms:
                    # 找到的最大面积前首先要保证当前的多边形是 误差允许范围内的多边形
                    if CPP_Planner_Kit.is_polygon_convex_area_same(temp_geom, tolerance=tolerance):
                        group_inside_max_area = max(group_inside_max_area, temp_geom.area)
                # 找到组内的最大面积后，还需和组间的最大面积比较，同时保存索引
                if group_inside_max_area > group_wise_max_area:
                    group_wise_max_area = group_inside_max_area
                    group_wise_max_index = i

            biggest_split = split_group[group_wise_max_index]  # 获取最大分割
            # 将所有的凸边形保存，同时 非凸边形 继续参与分割，直到没有凸边形位置
            for temp_polygon in biggest_split.geoms:
                if CPP_Planner_Kit.is_polygon_convex_area_same(temp_polygon, tolerance=tolerance):
                    convex.append(temp_polygon)
                else:
                    concave.put(temp_polygon)
        # print("田块分割结束...")
        return convex

    @staticmethod
    def get_largest_multipolygon(multipolygon: shapely.MultiPolygon) -> Polygon:
        """
        将一个 multipolygon 中最大的 polygon 找出并返回
        :param multipolygon
        :return: 最大的多边形
        """
        # 分解出单独的 polygon
        polygons = [x for x in multipolygon.geoms]
        polygons_area = [x.area for x in polygons]
        return polygons[polygons_area.index(max(polygons_area))]

    @staticmethod
    def get_corrected_swath_width(swath_width: float, slope: float):
        """
        获取修正后的耕作宽度，由于我们在路径规划的时候，仅考虑了水平的经纬度，因此需要利用公式进行修正
        :param swath_width 原来耕地的宽度，由于是坡度所以不可以直接使用经纬度来规划，需要通过坡度来修正其原本的路径在水平方向的投影
        :param slope 坡度大小
        :return 修正后的垄的宽度
        """
        return swath_width * math.cos(math.radians(slope))

    @staticmethod
    def get_path_bound(paths: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        将原来规划好的路径，提取出其边缘点，组成其外包多边形，注意这里不是取的 convex_hull
        :param paths: 路径点集合
        :return: 路径外包多边形的 gpd
        """
        path_points = paths['geometry'].apply(lambda line: [line.coords[0], line.coords[-1]])
        path_points_side1 = []
        path_points_side2 = []
        # 将两端的点分别保存
        for point in path_points:
            path_points_side1.append(point[0])
            path_points_side2.append(point[1])
        # 将第二组点反向保存，模拟“旋转一圈遍历点”
        path_points_side2.reverse()
        path_points_side1 = path_points_side1 + path_points_side2
        return Polygon(path_points_side1)

    @staticmethod
    def calc_headland_width(turning_radius: float, swath_width: float, vehicle_length: float, vehicle_width: float,
                            buffer=0., show_info=False):
        """
        *** 不同的转向方式可能需要不同的地头生成方式 ***
        计算当前地块的地头区域，当前函数生成的地头仅适合农机单向耕作，变换耕作垄采用原地转向倒车的方式
        * 如果是坡地，这里的地头宽度是没有经过坡度修正的宽度
        :param turning_radius: 农机的转向半径
        :param swath_width: 两条垄（中线）之间的宽度
        :param vehicle_length: 农机的长度
        :param vehicle_width: 农机的宽度
        :param buffer: 是否为地头区域增加一些“间隔/缓冲区”
        :param show_info: 是否显示规划地头时候的信息
        :return: 建议的地头宽度，如果设置了 buffer，则返回添加了缓冲区的地头宽度，同时返回农机转动的角度（弧度值）
        """
        # 计算求得车转向的角度
        # theta = math.asin(swath_width / (2 * vehicle_length))
        theta = math.cos((turning_radius - swath_width / 2) / turning_radius)
        # 计算车顶点到转向圆形的距离 r2
        # r2 = math.sqrt((turning_radius + vehicle_width / 2) ** 2 + (vehicle_length / 2) ** 2)
        r2 = math.sqrt((turning_radius + vehicle_width / 2) ** 2 + (vehicle_length / 2) ** 2)
        # theta2 = math.asin(vehicle_length / (2 * r2))
        theta2 = math.atan((vehicle_length / 2) / (turning_radius + vehicle_width / 2))
        # w2 = r2 * math.sin(theta + theta2)
        width_1 = r2 * math.sin(theta + theta2)
        final_width = width_1 + vehicle_length / 2
        if show_info:
            print("Theta: ", theta)
            print("r2: ", r2)
            print("Theta2: ", theta2)
            print("w2:", width_1)
            print("Buffer: ", buffer)
            print("最终地头宽度: ", final_width + buffer)

        return final_width + buffer, theta


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
    def scanline_algorithm_single_no_turn(land: gpd.GeoDataFrame, step_size: float, along_long_edge=False) \
            -> gpd.GeoDataFrame:
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
        land_centroid = land_polygon.centroid  # 当前地块的原始坐标位置，保存的中心位置，方便后期反向旋转回退
        if not along_long_edge:
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

        print("这次规划完成！")
        return path_gdf

    @staticmethod
    def scanline_algorithm_single_with_headland(land: gpd.GeoDataFrame, step_size: float, along_long_edge=True,
                                                headland='none', head_land_width=0.0, min_path_length=5,
                                                get_largest_headland=False) -> [gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
            基于 scanline_algorithm_single_no_turn() 添加了计算地头的功能，通过输入生成地头的方向（由于当前地块被旋转到了水平，而旋转的
        依据是将长边设置为水平，因此方向为 left, right）来生成对应方向的地头，同时可以控制需要设置的地头宽度
        :param land: 进行路径规划的地块，最好是简单的多边形，凸边形或是接近凸边形
        :param step_size: 机械耕作的宽度，即每一根路径的间隔
        :param along_long_edge: 是否沿着长边进行路径规划，判断长边的依据为当前多边形的最小外接矩形（MABR），否则默认沿着“水平方向”
        :param headland: 生成地头的方式，有：1. 两侧（both）2. 不生成（none） 3.左侧（left） 4.右侧（right）
        :param head_land_width: 生成地头的宽度，目前仅能够固定宽度生成地头
        :param min_path_length: 最小保留的耕作路径长度，默认 5m
        :param get_largest_headland: 是否获取最大的地头，如果为 True，则会获取最大的地头，否则会获取所有的地头
        :return: 当前地块的扫描路径，以及地头区域
        """
        if len(land) > 1:
            print("scanline_algorithm_single: 地块超过了一个，当前地块大小：", len(land))
            return

        # 将 land 中的 polygon 提取出来，保证一次只有一个地块，随即获取相关的信息
        land_polygon = land.iloc[0].geometry
        land_centroid = land_polygon.centroid  # 当前地块的原始坐标位置，保存的中心位置，方便后期反向旋转回退
        if not along_long_edge:
            land_angle = 0
        else:
            land_angle = CPP_Planner_Kit.get_land_MABR_angle(land_polygon)  # 获取角度

        # 置于水平，在后面的路径规划算法中，优先使用该旋转的地块来进行路径规划
        rotated_polygon = affinity.rotate(land_polygon, -land_angle, origin=land_centroid)
        # if along_long_edge:
        #     print("根据田块长边开始路径规划...")
        # else:
        #     print("水平方向开始路径规划")

        # 计算多边形的边界框
        min_x, min_y, max_x, max_y = rotated_polygon.bounds

        # 初始化路径线列表
        path_lines = []

        # 迭代扫描线
        for y in np.arange(min_y + step_size * 0.5, max_y - step_size * 0.5, step_size):
            # for y in np.arange(min_y + 0.1, max_y + step_size - 0.1, step_size):
            # for y in np.arange(min_y, max_y - 1, step_size):
            row_points = []
            for i in range(len(rotated_polygon.exterior.coords) - 1):
                edge = rotated_polygon.exterior.coords[i]
                next_edge = rotated_polygon.exterior.coords[i + 1]
                if (edge[1] <= y and next_edge[1] > y) or (next_edge[1] <= y and edge[1] > y):
                    x = edge[0] + (next_edge[0] - edge[0]) * (y - edge[1]) / (next_edge[1] - edge[1])
                    row_points.append([x, y])

            # 创建扫描线的 LineString 对象
            if len(row_points) > 1:
                # 处理地头，首先找到 x 最小和最大的点的索引，代表当前扫描线的边界
                min_x_index = min(range(len(row_points)), key=lambda i: row_points[i][0])
                max_x_index = max(range(len(row_points)), key=lambda i: row_points[i][0])
                if row_points[max_x_index][0] - row_points[min_x_index][0] > 10:
                    # 生成地头
                    if headland == 'left' or headland == 'both':
                        # row_points[min_x_index][0] = row_points[min_x_index][0] + head_land_width
                        row_points[min_x_index][0] = min(row_points[max_x_index][0],
                                                         row_points[min_x_index][0] + head_land_width)
                    if headland == 'right' or headland == 'both':
                        # row_points[max_x_index][0] = row_points[max_x_index][0] - head_land_width
                        row_points[max_x_index][0] = max(row_points[min_x_index][0],
                                                         row_points[max_x_index][0] - head_land_width)
                    path_line = LineString(row_points)
                    # 尝试：仅保留固定长度以上的耕作路径?
                    if path_line.length > head_land_width:
                        path_line = affinity.rotate(path_line, land_angle, origin=land_centroid)
                        path_lines.append(path_line)

        # 创建 GeoDataFrame 对象
        path_gdf = gpd.GeoDataFrame(geometry=path_lines, crs=land.crs)
        # 生成地头区域，通过创建耕作路径的缓冲区，来找到地块的区域，最后通过差集来得到地头区域
        # print("生成地头区域...")
        path_buffer_1 = path_gdf.buffer(step_size + 0.2, single_sided=True).unary_union
        path_buffer_2 = path_gdf.buffer(-(step_size + 0.2), single_sided=True).unary_union
        path_buffer = gpd.GeoDataFrame(geometry=[path_buffer_1, path_buffer_2], crs=land.crs)
        path_area_union = path_buffer.unary_union
        if path_area_union != None:
            path_area_union = path_area_union.simplify(step_size * 0.8)
        # path_area_union = Polygon(path_buffer_1.union(path_buffer_2))
        # 目前无法解决锯齿的问题
        # if len(path_gdf.geometry) > 1:
        #     path_area_union = CPP_Planner_Kit.get_path_bound(path_gdf)
        # else:
        #     path_area_union = gpd.GeoDataFrame(geometry=[], crs=land.crs)
        headland_area = land_polygon.difference(path_area_union)
        # 由于多边形在计算机图形中本身就有误差，因此需要将一些零碎的小块删除，保留最大的作为地块即可
        if type(headland_area) == Polygon:
            headland_area = shapely.MultiPolygon([headland_area])
        # 将边缘平滑
        # if headland_area != None:
        #     headland_area = headland_area.simplify(step_size * 0.7)
        if get_largest_headland:
            headland_area = CPP_Planner_Kit.get_largest_multipolygon(headland_area)

        # headland_gdf = gpd.GeoDataFrame(geometry=[headland_area.convex_hull], crs=land.crs)
        headland_gdf = gpd.GeoDataFrame(geometry=[headland_area], crs=land.crs)
        # 保证当前的地头生成的区域仅在地块内
        headland_gdf = headland_gdf.intersection(land)
        # print("这次规划完成！")
        return path_gdf, headland_gdf

    @staticmethod
    def scanline_algorithm_single_with_headland_2(land: gpd.GeoDataFrame, step_size: float, along_long_edge=True,
                                                  headland='none', head_land_width=0.0, min_path_length=5,
                                                  get_largest_headland=False) -> [gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
            基于 scanline_algorithm_single_no_turn() 添加了计算地头的功能，通过输入生成地头的方向（由于当前地块被旋转到了水平，而旋转的
        依据是将长边设置为水平，因此方向为 left, right）来生成对应方向的地头，同时可以控制需要设置的地头宽度
        * 和原本 scanline 的区别：这一次生成地头的方式，是通过平移地块边界生成
        :param land: 进行路径规划的地块，最好是简单的多边形，凸边形或是接近凸边形
        :param step_size: 机械耕作的宽度，即每一根路径的间隔
        :param along_long_edge: 是否沿着长边进行路径规划，判断长边的依据为当前多边形的最小外接矩形（MABR），否则默认沿着“水平方向”
        :param headland: 生成地头的方式，有：1. 两侧（both）2. 不生成（none） 3.左侧（left） 4.右侧（right）
        :param head_land_width: 生成地头的宽度，目前仅能够固定宽度生成地头
        :param min_path_length: 最小保留的耕作路径长度，默认 5m
        :param get_largest_headland: 是否获取最大的地头，如果为 True，则会获取最大的地头，否则会获取所有的地头
        :return: 当前地块的扫描路径，以及地头区域
        """
        if len(land) > 1:
            print("scanline_algorithm_single: 地块超过了一个，当前地块大小：", len(land))
            return

        # 将 land 中的 polygon 提取出来，保证一次只有一个地块，随即获取相关的信息
        land_polygon = land.iloc[0].geometry
        land_centroid = land_polygon.centroid  # 当前地块的原始坐标位置，保存的中心位置，方便后期反向旋转回退
        if not along_long_edge:
            land_angle = 0
        else:
            land_angle = CPP_Planner_Kit.get_land_MABR_angle(land_polygon)  # 获取角度

        # 置于水平，在后面的路径规划算法中，优先使用该旋转的地块来进行路径规划
        rotated_polygon = affinity.rotate(land_polygon, -land_angle, origin=land_centroid)

        # 计算多边形的边界框
        min_x, min_y, max_x, max_y = rotated_polygon.bounds

        # 初始化路径线列表
        path_lines = []
        # 地块边缘的点列
        left_edge_points = []
        right_edge_points = []

        # 迭代扫描线
        for y in np.arange(min_y + 0.1, max_y + step_size - 0.1, step_size):
            # for y in np.arange(min_y, max_y - 1, step_size):
            row_points = []
            for i in range(len(rotated_polygon.exterior.coords) - 1):
                edge = rotated_polygon.exterior.coords[i]
                next_edge = rotated_polygon.exterior.coords[i + 1]
                if (edge[1] <= y and next_edge[1] > y) or (next_edge[1] <= y and edge[1] > y):
                    x = edge[0] + (next_edge[0] - edge[0]) * (y - edge[1]) / (next_edge[1] - edge[1])
                    row_points.append([x, y])

            # 创建扫描线的 LineString 对象
            if len(row_points) > 1:
                # 处理地头，首先找到 x 最小和最大的点的索引，代表当前扫描线的边界
                min_x_index = min(range(len(row_points)), key=lambda i: row_points[i][0])
                max_x_index = max(range(len(row_points)), key=lambda i: row_points[i][0])
                if row_points[max_x_index][0] - row_points[min_x_index][0] > 10:
                    # 生成地头
                    if headland == 'left' or headland == 'both':
                        left_edge_points.append(row_points[min_x_index].copy())
                        row_points[min_x_index][0] = row_points[min_x_index][0] + head_land_width
                    if headland == 'right' or headland == 'both':
                        right_edge_points.append(row_points[max_x_index].copy())
                        row_points[max_x_index][0] = row_points[max_x_index][0] - head_land_width
                    path_line = LineString(row_points)
                    # 尝试：仅保留固定长度以上的耕作路径?
                    if path_line.length > head_land_width:
                        path_line = affinity.rotate(path_line, land_angle, origin=land_centroid)
                        path_lines.append(path_line)

        # 测试，添加点
        if len(left_edge_points) > 1:
            left_edge_points[0][1] = min_y - 5
            left_edge_points[-1][1] = max_y + 5
        if len(right_edge_points) > 1:
            right_edge_points[0][1] = min_y - 5
            right_edge_points[-1][1] = max_y + 5

        # 创建 GeoDataFrame 对象
        path_gdf = gpd.GeoDataFrame(geometry=path_lines, crs=land.crs)
        # 处理两侧的地头，使用地块边缘平移的方式
        headland_polygons = []
        if (headland == 'left' or headland == 'both') and len(left_edge_points) > 0:
            temp_points = left_edge_points.copy()
            # 因为前面的扫描线是从下往上迭代的，因此第一个点一定比最后一个点高
            temp_points[0][1] -= 2
            temp_points[-1][1] += 2
            for point in reversed(left_edge_points):
                temp_points.append([point[0] + head_land_width, point[1]])
            temp_points[len(left_edge_points)][1] += 2
            temp_points[-1][1] -= 2
            temp_points[-1][1] = max(temp_points[-1][1] - 2, temp_points[0][1])
            if len(temp_points) > 3:
                headland_polygons.append(Polygon(temp_points))

        if (headland == 'right' or headland == 'both') and len(left_edge_points) > 0:
            temp_points = right_edge_points.copy()
            # 原理同上
            temp_points[0][1] -= 2
            temp_points[-1][1] += 2
            for point in reversed(right_edge_points):
                temp_points.append([point[0] - head_land_width, point[1]])
            temp_points[len(left_edge_points)][1] += 2
            temp_points[-1][1] -= 2
            temp_points[-1][1] = max(temp_points[-1][1] - 2, temp_points[0][1])
            if len(temp_points) > 3:
                headland_polygons.append(Polygon(temp_points))
        headland_polygons = shapely.MultiPolygon(headland_polygons)
        headland_gdf = gpd.GeoDataFrame(geometry=[headland_polygons], crs=land.crs)
        # 保证地头区域不越界
        headland_gdf = headland_gdf.buffer(0.1).simplify(0.5).intersection(land)
        return path_gdf, headland_gdf

    @staticmethod
    def scanline_algorithm_single_with_headland_3(land: gpd.GeoDataFrame, step_size: float, turning_radius: float,
                                                  vehicle_length: float, vehicle_width: float,
                                                  along_long_edge=True, headland='none', head_land_width=0.0,
                                                  min_path_length=5, get_largest_headland=False
                                                  ) -> [gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        这一次的 scanline 算法考虑到了地块的边界角度，能够一定程度上保证转向路径一定在地块内部（长边情况）
        :param land:
        :param step_size:
        :param turning_radius:
        :param vehicle_length:
        :param vehicle_width:
        :param along_long_edge:
        :param headland:
        :param head_land_width:
        :param min_path_length:
        :param get_largest_headland:
        :return:
        """
        if len(land) > 1:
            print("scanline_algorithm_single: 地块超过了一个，当前地块大小：", len(land))
            return

        # 将 land 中的 polygon 提取出来，保证一次只有一个地块，随即获取相关的信息
        land_polygon = land.iloc[0].geometry
        land_centroid = land_polygon.centroid  # 当前地块的原始坐标位置，保存的中心位置，方便后期反向旋转回退
        if not along_long_edge:
            land_angle = 0
        else:
            land_angle = CPP_Planner_Kit.get_land_MABR_angle(land_polygon)  # 获取角度

        # 置于水平，在后面的路径规划算法中，优先使用该旋转的地块来进行路径规划
        rotated_polygon = affinity.rotate(land_polygon, -land_angle, origin=land_centroid)
        # 计算一次 flat turn 需要横跨的垄
        swath_jump = CPP_Planner_TurningRail_Maker.calc_min_swath_jump(turning_radius, step_size)
        # 计算多边形的边界框
        min_x, min_y, max_x, max_y = rotated_polygon.bounds
        path_lines = []

        # 迭代扫描线
        for y in np.arange(min_y + step_size * 0.5, max_y - step_size * 0.5, step_size):
            row_points = []
            for i in range(len(land_polygon.exterior.coords) - 1):
                # edge = rotated_polygon.exterior.coords[i]
                # next_edge = rotated_polygon.exterior.coords[i + 1]
                edge = land_polygon.exterior.coords[i]
                next_edge = land_polygon.exterior.coords[i + 1]
                if (edge[1] <= y and next_edge[1] > y) or (next_edge[1] <= y and edge[1] > y):
                    x = edge[0] + (next_edge[0] - edge[0]) * (y - edge[1]) / (next_edge[1] - edge[1])
                    row_points.append([x, y])
            # 创建扫描线的 LineString 对象
            if len(row_points) > 1:
                # 处理地头，首先找到 x 最小和最大的点的索引，代表当前扫描线的边界
                min_x_index = min(range(len(row_points)), key=lambda i: row_points[i][0])
                max_x_index = max(range(len(row_points)), key=lambda i: row_points[i][0])
                # 开始计算 headland_width，分为 左右 两侧各自的
                left_detector = LineString(((row_points[min_x_index][0] - 0.1, row_points[min_x_index][1]),
                                            (row_points[min_x_index][0] + 0.1, row_points[min_x_index][1])))
                right_detector = LineString(((row_points[max_x_index][0] - 0.1, row_points[max_x_index][1]),
                                             (row_points[max_x_index][0] + 0.1, row_points[max_x_index][1])))
                left_angle, right_angle = -1, -1
                # 检测当前扫描线（耕作路径）相交与地块的地块边界
                for i in range(len(land_polygon.exterior.coords) - 1):
                    # edge = LineString((land_polygon.exterior.coords[i], land_polygon.exterior.coords[i + 1]))
                    edge = LineString((rotated_polygon.exterior.coords[i], rotated_polygon.exterior.coords[i + 1]))
                    if left_detector.intersects(edge):
                        left_angle = math.atan2(edge.coords[1][1] - edge.coords[0][1],
                                                edge.coords[1][0] - edge.coords[0][0])
                    if right_detector.intersects(edge):
                        right_angle = math.atan2(edge.coords[0][1] - edge.coords[1][1], edge.coords[0][0] - edge.coords[1][0])
                    # print('le angle: ', math.degrees(left_angle))
                    # print('ri angle: ', math.degrees(right_angle))
                        # right_angle = math.atan2(edge.coords[0][1] - edge.coords[1][1], edge.coords[0][0] - edge.coords[1][0])

                # left_headland_width = turning_radius * (1 + math.cos(pi - left_angle)) + step_size * swath_jump / 2 + vehicle_length / 2
                # left_headland_width = (step_size * swath_jump + vehicle_width/2) * abs(1 / math.tan(pi - left_angle)) + turning_radius + vehicle_width / 2
                # right_headland_width = turning_radius * (1 + math.cos(right_angle)) + step_size * swath_jump / 2 + vehicle_length / 2
                # right_headland_width = (step_size * swath_jump + vehicle_width/2) * abs(1 / math.tan(right_angle)) + turning_radius + vehicle_width / 2
                # 这里暂时没加 headland_width
                left_headland_width = abs(step_size * swath_jump * math.tan(left_angle - pi / 2)) + turning_radius + vehicle_length / 2
                right_headland_width = abs(step_size * swath_jump * math.tan(right_angle - pi / 2)) + turning_radius + vehicle_length / 2
                # print('lh wid: ', left_headland_width)
                # print('rh wid: ', right_headland_width)

                # print(left_headland_width, "  ", right_headland_width)
                # print("le an: ", math.degrees(left_angle))
                # print(abs(1 /math.tan(pi - left_angle)) * (step_size * swath_jump + vehicle_width / 2))
                # print(1 /math.tan(math.tan(right_angle)))


                # 生成地头区域
                if row_points[max_x_index][0] - row_points[min_x_index][0] > 10:
                    # 生成地头
                    if headland == 'left' or headland == 'both':
                        # row_points[min_x_index][0] = min(row_points[max_x_index][0],
                        #                                  row_points[min_x_index][0] + left_headland_width)
                        row_points[min_x_index][0] = min(row_points[max_x_index][0],
                                                         min(row_points[min_x_index][0] + left_headland_width,
                                                             row_points[max_x_index][0]))
                    if headland == 'right' or headland == 'both':
                        # row_points[max_x_index][0] = max(row_points[min_x_index][0],
                        #                                  row_points[max_x_index][0] - right_headland_width)
                        row_points[max_x_index][0] = max(row_points[min_x_index][0],
                                                         max(row_points[max_x_index][0] - right_headland_width,
                                                             row_points[min_x_index][0]))
                    path_line = LineString(row_points)
                    # 尝试：仅保留固定长度以上的耕作路径?
                    if path_line.length > max(left_headland_width, right_headland_width):
                        # path_line = affinity.rotate(path_line, land_angle, origin=land_centroid)
                        path_lines.append(path_line)
        path_gdf = gpd.GeoDataFrame(geometry=path_lines, crs=land.crs)
        # 生成地头区域，通过创建耕作路径的缓冲区，来找到地块的区域，最后通过差集来得到地头区域
        # print("生成地头区域...")
        path_buffer_1 = path_gdf.buffer(step_size + 0.2, single_sided=True).unary_union
        path_buffer_2 = path_gdf.buffer(-(step_size + 0.2), single_sided=True).unary_union
        path_buffer = gpd.GeoDataFrame(geometry=[path_buffer_1, path_buffer_2], crs=land.crs)
        path_area_union = path_buffer.unary_union
        # 将边缘平滑
        if path_area_union != None:
            path_area_union = path_area_union.simplify(step_size * 0.8)
        # path_area_union = Polygon(path_buffer_1.union(path_buffer_2))
        headland_area = land_polygon.difference(path_area_union)
        # 由于多边形在计算机图形中本身就有误差，因此需要将一些零碎的小块删除，保留最大的作为地块即可
        if type(headland_area) == Polygon:
            headland_area = shapely.MultiPolygon([headland_area])

        # headland_gdf = gpd.GeoDataFrame(geometry=[headland_area.convex_hull], crs=land.crs)
        headland_gdf = gpd.GeoDataFrame(geometry=[headland_area], crs=land.crs)
        # 保证当前的地头生成的区域仅在地块内
        headland_gdf = headland_gdf.intersection(land_polygon)
        # print("这次规划完成！")
        return path_gdf, headland_gdf
        pass

    @staticmethod
    def scanline_algorithm_with_headland_by_direction(land: gpd.GeoDataFrame, step_size: float, land_angle,
                                                      headland='none', head_land_width=0, min_path_length=5,
                                                      get_largest_headland=False) -> [gpd.GeoDataFrame,
                                                                                      gpd.GeoDataFrame]:
        """
        按照特定角度计算scanline路径规划，同时生成地头
        :param land: 当前的分割的地块的子块，geopandas.GeoDataFrame
        :param step_size: 每次耕作幅宽，这里需要放入修正后的机械化耕作幅宽
        :param land_angle: 当前地块的角度，经过该角度旋转后，地块能够放置为水平，方便部署 scanline 算法
        :param headland:
        :param head_land_width:
        :param min_path_length:
        :param get_largest_headland:
        :return:
        """
        if len(land) > 1:
            print("scanline_algorithm_single: 地块超过了一个，当前地块大小：", len(land))
            return

        # 将 land 中的 polygon 提取出来，保证一次只有一个地块，随即获取相关的信息
        land_polygon = land.iloc[0].geometry
        land_centroid = land_polygon.centroid  # 当前地块的原始坐标位置，保存的中心位置，方便后期反向旋转回退

        # 置于水平，在后面的路径规划算法中，优先使用该旋转的地块来进行路径规划
        rotated_polygon = affinity.rotate(land_polygon, -land_angle, origin=land_centroid)
        # if along_long_edge:
        #     print("根据田块长边开始路径规划...")
        # else:
        #     print("水平方向开始路径规划")

        # 计算多边形的边界框
        min_x, min_y, max_x, max_y = rotated_polygon.bounds

        # 初始化路径线列表
        path_lines = []

        # 迭代扫描线
        for y in np.arange(min_y + 0.1, max_y + step_size - 0.1, step_size):
            # for y in np.arange(min_y, max_y - 1, step_size):
            row_points = []
            for i in range(len(rotated_polygon.exterior.coords) - 1):
                edge = rotated_polygon.exterior.coords[i]
                next_edge = rotated_polygon.exterior.coords[i + 1]
                if (edge[1] <= y and next_edge[1] > y) or (next_edge[1] <= y and edge[1] > y):
                    x = edge[0] + (next_edge[0] - edge[0]) * (y - edge[1]) / (next_edge[1] - edge[1])
                    row_points.append([x, y])

            # 创建扫描线的 LineString 对象
            if len(row_points) > 1:
                # 处理地头，首先找到 x 最小和最大的点的索引，代表当前扫描线的边界
                min_x_index = min(range(len(row_points)), key=lambda i: row_points[i][0])
                max_x_index = max(range(len(row_points)), key=lambda i: row_points[i][0])

                # 生成地头
                if headland == 'left' or headland == 'both':
                    row_points[min_x_index][0] = row_points[min_x_index][0] + head_land_width
                if headland == 'right' or headland == 'both':
                    row_points[max_x_index][0] = row_points[max_x_index][0] - head_land_width
                path_line = LineString(row_points)
                # 尝试：仅保留固定长度以上的耕作路径?
                if path_line.length > head_land_width:
                    path_line = affinity.rotate(path_line, land_angle, origin=land_centroid)
                    path_lines.append(path_line)

        # 创建 GeoDataFrame 对象
        path_gdf = gpd.GeoDataFrame(geometry=path_lines, crs=land.crs)
        # 生成地头区域，通过创建耕作路径的缓冲区，来找到地块的区域，最后通过差集来得到地头区域
        # print("生成地头区域...")
        path_buffer_1 = path_gdf.buffer(step_size + 0.2, single_sided=True).unary_union
        path_buffer_2 = path_gdf.buffer(-(step_size + 0.2), single_sided=True).unary_union
        path_buffer = gpd.GeoDataFrame(geometry=[path_buffer_1, path_buffer_2], crs=land.crs)
        path_area_union = path_buffer.unary_union
        # path_area_union = Polygon(path_buffer_1.union(path_buffer_2))

        headland_area = land_polygon.difference(path_area_union)
        # 由于多边形在计算机图形中本身就有误差，因此需要将一些零碎的小块删除，保留最大的作为地块即可
        if type(headland_area) == Polygon:
            headland_area = shapely.MultiPolygon([headland_area])
        if get_largest_headland:
            headland_area = CPP_Planner_Kit.get_largest_multipolygon(headland_area)

        # headland_gdf = gpd.GeoDataFrame(geometry=[headland_area.convex_hull], crs=land.crs)
        headland_gdf = gpd.GeoDataFrame(geometry=[headland_area], crs=land.crs)
        # 保证当前的地头生成的区域仅在地块内
        headland_gdf = headland_gdf.intersection(land)
        # print("这次规划完成！")
        return path_gdf, headland_gdf

        pass


# end class CPP_Algorithms ------------------------------------------------------------

"""
这个类包含了能够自己实现的CPP算法的优化方式
"""


class CPP_Algorithm_Optimizers:
    @staticmethod
    def gen_path_with_minimum_headland_area(land: gpd.GeoDataFrame, step_size: float, along_long_edge=True,
                                            head_land_width=0):
        """
        对当前地块进行路径规划，使用扫描线算法，每一次输入的地块保证“接近”凸边形，输出带有 地头 的路径规划算法
        * 耕作路径为当前地块的 MABR 的长边方向
        * 地块为当前耕作路径两端之一，保证输出较小的地头区域
        :param land: 进行路径规划的地块，最好是简单的多边形，凸边形或是接近凸边形
        :param step_size: 机械耕作的宽度，即每一根路径的间隔
        :param along_long_edge: 是否沿着长边进行路径规划，判断长边的依据为当前多边形的最小外接矩形（MABR），否则默认沿着“水平方向”
        :param head_land_width: 生成地头的宽度，目前仅能够固定宽度生成地头
        :return: 耕作路径、地头区域（面积最小）
        """
        # 首先获取两侧的分别的地头
        path_1, headland_1 = CPP_Algorithms.scanline_algorithm_single_with_headland(land, step_size, along_long_edge,
                                                                                    head_land_width=head_land_width,
                                                                                    headland='left')
        path_2, headland_2 = CPP_Algorithms.scanline_algorithm_single_with_headland(land, step_size, along_long_edge,
                                                                                    head_land_width=head_land_width,
                                                                                    headland='right')
        if headland_1.geometry.area[0] > headland_2.geometry.area[0]:
            return path_2, headland_2
        else:
            return path_1, headland_1

    @staticmethod
    def gen_path_with_minimum_headland_area_by_direction(land: gpd.GeoDataFrame, step_size: float,
                                                         head_land_width=0):
        """
        通过对当前的多边形地块进行多边形旋转，并且记录当前地块作业角度下生成的地头的面积，选择面积最小的角度进行最终耕作地块的生成
        :param land: 进行路径规划的地块，最好是简单的多边形
        :param step_size: 机械耕作的宽度，如果是带坡度的地形，那么这个宽度需要进行修正
        :param head_land_width: 生成地头的宽度，按照固定的宽度预留出地头的宽度
        :return: 耕作路径和地头区域，保证当前角度耕作能够预留出最小的地头区域，即耕作面积最大
        """
        max_area = land.area[0]

        diff_headland_direction_area = []
        land_polygon = land.geometry.iloc[0]

        mabr_angle = CPP_Planner_Kit.get_land_MABR_angle(land_polygon)
        land_polygon = affinity.rotate(land_polygon, -mabr_angle, origin='centroid')

        # 旋转寻找
        # for angle in range(0, 180):
        for angle in range(-65, 65):
            single_polygon = affinity.rotate(land_polygon, -angle, origin='centroid')
            land_regen = gpd.GeoDataFrame(geometry=[single_polygon], crs=land.crs)
            path, headland = CPP_Algorithms.scanline_algorithm_single_with_headland(
                land=land_regen, step_size=step_size,
                along_long_edge=False,
                headland='left', head_land_width=head_land_width, get_largest_headland=False
            )
            diff_headland_direction_area.append(headland.geometry.area)
        # diff_headland_direction_area = [x.item() for x in diff_headland_direction_area]
        diff_headland_direction_area = [x.item() if not math.isnan(x.item()) else max_area for x in
                                        diff_headland_direction_area]
        min_area_angle = diff_headland_direction_area.index(min(diff_headland_direction_area))
        print("min angle: ", min_area_angle)

        path, headland = CPP_Algorithms.scanline_algorithm_with_headland_by_direction(land=land, step_size=step_size,
                                                                                      land_angle=min_area_angle + mabr_angle - 65,
                                                                                      headland='left',
                                                                                      head_land_width=head_land_width)
        # path = path.rotate(min_area_angle)
        # headland = headland.rotate(min_area_angle)
        return path, headland

    @staticmethod
    def gen_path_with_minimum_headland_area_by_edge(land: gpd.GeoDataFrame, step_size: float, head_land_width=6.0,
                                                    turning_radius=4.5,
                                                    headland_mode='left', compare_mode='headland', return_theta=False):
        """
        通过从当前的地块（保证当前的地块形状接近凸边形）的每一条边顺着耕作出发，预留耕作的地头，通过比较顺着各边生成的地头的面积，以及耕作路径的
        长度，找到最优解，可以是耕作路径最长，也可以是耕地面积最大
        * 需要注意，当耕地一个方向的长度低于 headland_width 时，算法会返回 nan，需要过滤掉这种情况
        :param land: 进行路径规划的地块，要求为接近 凸边形 的多边形，接近的程度为 tolerance，默认为0.05
        :param step_size: 耕作幅宽，需要根据当前地块所处的平均坡度进行修正
        :param head_land_width: 地头区域的宽度，需要保证机械能够在这里做出想要的例如：掉头、转向等动作
        :param turning_radius:
        :param headland_mode: 生成地头的模式，可以是 两侧（both），左侧（left），右侧（right）和不生成（none），默认为左侧
        :param compare_mode: 比较的模式，分别为比较路径的长度，和地头的面积（默认）
        :param return_theta: 是否返回最小地块的角度
        :return: 规划完成的路径和地头区域，保证输出的地头区域为当前情况下的最优值
        """
        land_polygon = land.geometry.iloc[0]
        # 获取当前多边形的中心点，方便后期旋转回原来的角度
        land_centroid = land.centroid[0]
        max_headland_area = land.area[0]
        # print(land_centroid)
        # 获取凸包，为了将就一些接近凸边形的多边形
        land_convex_hull = land_polygon.convex_hull
        # 开始查找边缘，以及其角度
        polygon_edge_angles = []
        for i in range(len(land_convex_hull.exterior.coords) - 1):
            temp_line = LineString([land_convex_hull.exterior.coords[i], land_convex_hull.exterior.coords[i + 1]])
            temp_edge_angle = math.degrees(math.atan2(temp_line.coords[1][1] - temp_line.coords[0][1],
                                                      temp_line.coords[1][0] - temp_line.coords[0][0]))
            polygon_edge_angles.append(temp_edge_angle)

        # 保存每一中边耕作后的路径和地头
        path_headland_collection = []
        path_lengths = []
        headland_areas = []
        for temp_angle in polygon_edge_angles:
            # 旋转到相应的角度，即将对应的边旋转为水平
            temp_rotated_polygon = affinity.rotate(land_polygon, -temp_angle, origin='centroid')
            temp_rotated_polygon_regen = gpd.GeoDataFrame(geometry=[temp_rotated_polygon], crs=land.crs)
            # land_centroid = temp_rotated_polygon_regen.centroid[0]
            # temp_path, temp_headland = CPP_Algorithms.scanline_algorithm_single_with_headland(
            #     land=temp_rotated_polygon_regen, step_size=step_size, along_long_edge=False,
            #     headland=headland_mode, head_land_width=head_land_width, get_largest_headland=False
            # )
            # TODO: 添加 vehicle_length / width
            temp_path, temp_headland = CPP_Algorithms.scanline_algorithm_single_with_headland_3(
                temp_rotated_polygon_regen, step_size, turning_radius=4.5, vehicle_length=4.5, vehicle_width=1.9,
                headland=headland_mode, along_long_edge=False
            )
            # 旋转回去
            temp_path = temp_path.rotate(temp_angle, origin=(land_centroid.coords[0][0], land_centroid.coords[0][1]))
            temp_headland = temp_headland.rotate(temp_angle,
                                                 origin=(land_centroid.coords[0][0], land_centroid.coords[0][1]))
            path_headland_collection.append([temp_path, temp_headland])
            temp_length = 0
            for line in temp_path.geometry:
                temp_length += line.length
            path_lengths.append(temp_length)
            # headland_areas.append(temp_headland.area.item() if not math.isnan(temp_headland.area.item()) or temp_headland.area.item() == 0.0
            #                       else max_headland_area)
            if not math.isnan(temp_headland.area.item()) and temp_headland.area.item() != 0:
                headland_areas.append(temp_headland.area.item())
            else:
                headland_areas.append(max_headland_area)

        # 选择最小的地头面积或是最长的耕作路径
        if compare_mode == 'headland':
            selected_edge_index = headland_areas.index(min(headland_areas))
        else:  # if compare_mode == 'path'
            selected_edge_index = path_lengths.index(max(path_lengths))

        if return_theta:
            return path_headland_collection[selected_edge_index][0], path_headland_collection[selected_edge_index][1], \
                polygon_edge_angles[selected_edge_index]

        # 返回对应最优的耕作路径和地头区域
        return path_headland_collection[selected_edge_index][0], path_headland_collection[selected_edge_index][1]


# end class Algorithm_Optimizers ------------------------------------------------------------


"""
这个类包含了转向轨迹的生成方式
"""


class CPP_Planner_TurningRail_Maker:
    """
    这是单向耕作，单向地头，倒车后继续耕作模式
    """

    @staticmethod
    def gen_single_curve(center: [float, float], radius: float, degree_step: float, theta_1: float, theta_2):
        """
        生成一个基础圆弧，默认从 theta_1 弧度 开始，逆时针旋转到 theta_2 弧度结束，由于考虑到多边形，以特定度数为精度间隔生成圆弧

        * 使用 *弧度*
        :param center 圆弧的圆心
        :param radius: 圆弧的半径
        :param degree_step: 圆弧的精度，输入为度数，后续会主动转换为弧度
        :param theta_1: 圆弧的起始 弧度
        :param theta_2: 圆弧的终止 弧度
        :return: LineString
        """
        # 默认从 0 开始
        # center_x = 0
        # center_y = 0
        center_x = center[0]
        center_y = center[1]

        # # 将起始点和终点的坐标转换为 LineString
        # start_point = (center_x + radius * np.cos(0), center_y + radius * np.sin(0),)
        # end_point = (center_x + radius * np.cos(theta), center_y + radius * np.sin(theta),)
        start_point = (center_x + radius * np.cos(theta_1), center_y + radius * np.sin(theta_1))
        end_point = (center_x + radius * np.cos(theta_2), center_y + radius * np.sin(theta_2))

        coordinates = [start_point]

        # # 依据 degree_step 来计算 start_point 和 end_point 之间的坐标
        # angles = np.arange(0, theta, np.deg2rad(degree_step))
        if theta_2 > theta_1:
            angles = np.arange(theta_1, theta_2, np.deg2rad(degree_step))
        else:
            angles = np.arange(theta_2, theta_1, np.deg2rad(degree_step))

        for angle in angles:
            temp_x = center_x + radius * np.cos(angle)
            temp_y = center_y + radius * np.sin(angle)
            coordinates.append((temp_x, temp_y))

        # 补充终点
        coordinates.append(end_point)

        curve = LineString(coordinates)
        return curve

    @staticmethod
    def gen_S_shape_curve(turning_radius: float, degree_step: float, vehicle_length: float, vehicle_width: float,
                          swath_width: float, buffer: float):
        """
        生成用于 单向 + 倒车 的转向 S 路径的生成，S 曲线的路径已经被标准化，起始位置为 (0, 0)，详细见 test_gen_turning_path
        * 使用 *弧度*
        * 生成的路径默认转向起始在路径末端坐标的 y - turning_radius 处，默认在 “水平化后的路径” 的右侧进行规划
        * 在生成路径地头，即 scanline_xxxx() 时，headland='right'，以保证正确生成地头
        * 如果转向路径过大，那么不一定车辆在重新转回水平时能刚好在下一条 swath 上
        :param turning_radius:
        :param degree_step:
        :param vehicle_length:
        :param vehicle_width:
        :param swath_width:
        :param buffer:
        :return: S 曲线的 gpd，由于是标准位置，因此 S curve 的起始位置默认为 S 的右侧顶点，设置为 0, 0
        """
        # 获得地头长度，以及旋转的弧度值
        headland_width, theta = CPP_Planner_Kit.calc_headland_width(turning_radius, swath_width, vehicle_length,
                                                                    vehicle_width, buffer)
        # 模拟两条耕作路径，但是完全按照实际情况配置比例大小，最终返回一个做好的 “标准 S 曲线路径”
        swath1 = LineString(((0, 1.45), (20, 1.45)))
        swath2 = LineString(((0, 0), (20, 0)))
        # 获取圆心的位置，此时 第二个圆心还需要修正
        center1 = (swath1.coords[1][0], swath1.coords[1][1] - turning_radius)
        center2 = (swath2.coords[1][0] - 2 * (turning_radius * math.sin(theta)), swath2.coords[1][1] + turning_radius)

        # 绘制弧线，根据两个圆心
        curve_1 = CPP_Planner_TurningRail_Maker.gen_single_curve(
            center1, turning_radius, degree_step, pi / 2, theta + pi / 2
        )
        curve_2 = CPP_Planner_TurningRail_Maker.gen_single_curve(
            center2, turning_radius, degree_step, -pi / 2, -pi / 2 + theta
        )

        # 修正曲线
        curve_correction_gap = abs(curve_2.coords[-1][1] - curve_1.coords[-1][1])
        curve_2_corrected = affinity.translate(curve_2, xoff=0, yoff=-curve_correction_gap, zoff=0)

        # 将曲线移动到（0，0）处
        curve_1_normalized = affinity.translate(curve_1, xoff=-curve_1.coords[0][0], yoff=-curve_1.coords[0][1])
        curve_2_normalized = affinity.translate(curve_2_corrected, xoff=-curve_1.coords[0][0],
                                                yoff=-curve_1.coords[0][1])

        # 作为 gpd 返回，注意这里没有设置 crs
        S_curved_line = gpd.GeoDataFrame(geometry=[curve_1_normalized, curve_2_normalized])
        return S_curved_line

    @staticmethod
    def gen_S_turning_paths_in_polygon(path: gpd.GeoDataFrame, theta: float, turning_radius: float,
                                       vehicle_length: float, vehicle_width: float, swath_width: float,
                                       headland_width: float, centroid, buffer=0, degree_step=0.2):
        """
        通过由其他算法生成的 path，来生成转向路径和反向倒车等路径
        * 本方法仅适用于单向作业，通过倒车至下一耕作路径的 “S型” 转向方式
        * 使用本方法，需要保证在生成地头的方向为 “right”
        :param path: 规划耕作路径
        :param theta: 路径与水平方向的夹角，用于旋转路径
        :param turning_radius: 转向半径
        :param vehicle_length: 农机长度
        :param vehicle_width: 农机宽度
        :param swath_width: 耕作宽度
        :param headland_width: 地头的宽度
        :param centroid: 旋转中心
        :param buffer: 缓冲区
        :param degree_step: 转向曲线的精度
        :return: 地头行驶路径，转向路径，倒车路径
        """
        # TODO: 如何保证 从上到下 和 从下到上两种路径的生成？
        # 将路径转至水平
        path = path.rotate(-theta, origin=centroid)
        paths = path.geometry.tolist()
        # 生成当前地块参数下的S曲线模版
        basic_S_curve = CPP_Planner_TurningRail_Maker.gen_S_shape_curve(turning_radius, degree_step, vehicle_length,
                                                                        vehicle_width, swath_width, buffer)

        # 开始制作转向路径、倒车路径、地头路径
        turning_curves = []
        backward_moves = []
        forward_moves = []
        for i in range(1, len(paths)):
            temp_path = paths[i]
            pre_path = paths[i - 1]
            # 找到 “右侧” 的一点
            if temp_path.coords[0][0] > temp_path.coords[-1][0]:
                temp_end_point = temp_path.coords[0]
                temp_begin_index = -1
            else:
                temp_end_point = temp_path.coords[-1]
                temp_begin_index = 0
            # temp_end_point = temp_path.coords[0]
            # 地头行驶路线
            temp_forward_line = LineString((temp_end_point, (temp_end_point[0] + headland_width, temp_end_point[1]),))
            # 转向的 S 曲线，注意这里是两根
            temp_S_line = basic_S_curve.translate(xoff=temp_forward_line.coords[-1][0],
                                                  yoff=temp_forward_line.coords[-1][1], zoff=0)
            # 倒车线
            temp_backward_line = LineString((temp_S_line.geometry[1].coords[0], pre_path.coords[temp_begin_index]))

            forward_moves.append(temp_forward_line)
            turning_curves.append(temp_S_line.geometry[0])
            turning_curves.append(temp_S_line.geometry[1])
            backward_moves.append(temp_backward_line)

        forward_moves = gpd.GeoDataFrame(geometry=forward_moves, crs=path.crs)
        turning_curves = gpd.GeoDataFrame(geometry=turning_curves, crs=path.crs)
        backward_moves = gpd.GeoDataFrame(geometry=backward_moves, crs=path.crs)

        # 转回去
        forward_moves = forward_moves.rotate(theta, origin=centroid)
        turning_curves = turning_curves.rotate(theta, origin=centroid)
        backward_moves = backward_moves.rotate(theta, origin=centroid)

        return forward_moves, turning_curves, backward_moves
        pass

    """
    这是双向耕作，首先使用套行法尽力耕作垄，再将剩下的垄用 “鱼尾” 转弯法 往复耕作
    """

    @staticmethod
    def calc_min_swath_jump(turning_radius: float, swath_width: float):
        """
        计算当前的转向半径下，距离当前垄的下一个最近可达的垄数
        例如：当前转向半径为 4.5， 垄距 1.45，则需要 (4.5 * 2) / 1.45 向上取整约为 7
             则从当前垄开始数（0），到第七垄（7）才是下一个能够够到的垄
        * 最近的垄为：在农机经历两次 90度 转向后，大于其水平距离的第一个垄
        :param turning_radius: 农机转向半径
        :param swath_width: 一垄的宽度
        :return: 从下一垄开始，最近能够够到的垄
        """
        next_reachable_swath = (2 * turning_radius) // swath_width
        if (2 * turning_radius) % swath_width == 0:
            return next_reachable_swath
        else:  # 向上取整
            return next_reachable_swath + 1

    @staticmethod
    def calc_flat_turn_headland_width(turning_radius: float, vehicle_length: float, vehicle_width: float):
        """
        计算在 “平滑转向 & 鱼尾转向“ 的时候，需要多宽的地头
        * 以农机 “中心“ 为起始点
        * 农机末尾刚好在耕作区域外
        * 以农机外侧上角为最远距离
        :param turning_radius: 农机的转向半径
        :param vehicle_length: 农机长度
        :param vehicle_width: 农机宽度
        :return: 最小地头宽度
        """
        return int(
            vehicle_length / 2 + math.sqrt((turning_radius + vehicle_width / 2) ** 2 + (vehicle_length / 2) ** 2))

    @staticmethod
    def gen_bow_shape_curve(turning_radius: float, min_jump_swaths: int, swath_width: float, side='right', plus=0,
                            degree_step=0.1):
        """
        生成一个 平滑转向 的基本形状，为两个 1/4圆中间连接一根直线的 弓 型
        * 弓 型被标准化到了原点，默认都是较上方的圆弧的点位于原点
        :param turning_radius: 当前农机的转向半径
        :param min_jump_swaths: 在当前农机的转向半径下，最小能够够到的下一个垄，这个距离绝不会小于农机耕作半径二倍
        :param swath_width: 一条垄的宽度
        :param side: 当前路径的方向，right 表示弓形的弓背朝右，反之亦然
        :param plus: 是否多计算 plus 条垄，例如原本是 7 垄，plus=1，则跳到第 8 垄
        :param degree_step: 精度设置
        :return: gpd.GeoDataFrame
        """

        if side == 'right':
            curve_1 = CPP_Planner_TurningRail_Maker.gen_single_curve((0, -turning_radius), turning_radius, degree_step,
                                                                     0, pi / 2)
            gap = swath_width * (min_jump_swaths + plus) - 2 * turning_radius
            padding_line = LineString(((turning_radius, -turning_radius), (turning_radius, -turning_radius - gap)))
            curve_2 = CPP_Planner_TurningRail_Maker.gen_single_curve((0, -turning_radius - gap), turning_radius,
                                                                     degree_step, -pi / 2, 0)
            pass
        elif side == 'left':
            curve_1 = CPP_Planner_TurningRail_Maker.gen_single_curve((0, -turning_radius), turning_radius, degree_step,
                                                                     pi / 2, pi)
            gap = swath_width * (min_jump_swaths + plus) - 2 * turning_radius
            padding_line = LineString(((-turning_radius, -turning_radius), (-turning_radius, -turning_radius - gap),))
            curve_2 = CPP_Planner_TurningRail_Maker.gen_single_curve((0, -turning_radius - gap), turning_radius,
                                                                     degree_step, pi, pi * 3 / 2)
            pass
        else:
            print("方向信息错误！")
            return
        result = gpd.GeoDataFrame(geometry=[curve_1, padding_line, curve_2])
        return result

    @staticmethod
    def gen_fishtail_shape_curve(turning_radius: float, swath_width: float, side='right', degree_step=0.1, plus=0):
        """
        生成鱼尾转向
        * 鱼尾 被标准化到了原点，“向下弯曲” 的圆弧的水平段端点位于 (0, 0)
        * 鱼尾转向默认移动到下一条路径，可以通过 plus 调整到另一条路上
        :param turning_radius: 当前农机的转向半径
        :param swath_width: 一垄的宽度
        :param side: 鱼尾朝向那一侧，默认右侧 right
        :param degree_step: 精度，建议越高越好
        :param plus: 默认就是移动到下一条，可以设置更大的跨度，例如 plus=1 则移动到下面的第二条
        :return:
        """
        if side == 'right':
            curve_1 = CPP_Planner_TurningRail_Maker.gen_single_curve((0, -turning_radius), turning_radius, degree_step,
                                                                     0, pi / 2)
            padding_1 = LineString((
                (turning_radius, -turning_radius), (turning_radius, -swath_width + turning_radius)
            ))
            curve_2 = CPP_Planner_TurningRail_Maker.gen_single_curve((0, turning_radius - swath_width), turning_radius,
                                                                     degree_step, -pi / 2, 0)
            pass
        elif side == 'left':
            curve_1 = CPP_Planner_TurningRail_Maker.gen_single_curve((0, -turning_radius), turning_radius, degree_step,
                                                                     pi / 2, pi)
            padding_1 = LineString((
                (-turning_radius, -turning_radius), (-turning_radius, -swath_width + turning_radius)
            ))
            curve_2 = CPP_Planner_TurningRail_Maker.gen_single_curve((0, -swath_width + turning_radius), turning_radius,
                                                                     degree_step, pi, pi * 3 / 2)
            pass
        else:
            print("鱼尾朝向不对！")
            return
        result = gpd.GeoDataFrame(geometry=[curve_1, padding_1, curve_2])
        return result

    @staticmethod
    def gen_path_tillage_method(path: gpd.GeoDataFrame, turning_radius: float, vehicle_length: float,
                                vehicle_width: float, swath_width: float):
        """
        当采取往复式耕作的时候，判定每一垄的耕作方向，以及下一条垄的位置，常用在 flat turn 这种无法直接通过掉头到达临近下一垄的情况
        * 将田块内的耕作路径抽象为点集合，例如 [0, 0, 0, 0, 0, 0] 因此耕作方向为向右或向左，且分上下
        * 每一垄路径的情况：
            * 0: 使用鱼尾转向
            * 1. 在上方向右到达最近可达垄 + 1垄  2. 在下方到达左侧最近可达垄  3. 在下方向右到达最近可达垄 + 1垄  4.在上方到达右侧最近可达垄
        :param path: 当前这片地块内的耕作的垄数
        :param turning_radius: 农机的转向半径
        :param vehicle_length: 农机的全长
        :param vehicle_width: 农机的宽度
        :param swath_width: 一条垄的宽度
        :return:
        """
        n = len(path)
        # n = 16
        swath = [0 for i in range(n)]
        # 找到最小的可以达到的下一垄，必须要大于转向半径的两倍
        min_jump_swaths = int(CPP_Planner_TurningRail_Maker.calc_min_swath_jump(turning_radius, swath_width))
        max_length = int((min_jump_swaths + 1) * 2 - 1)
        # iteration 是按照固定的一次能够通过 平滑转向 耕作完的一整块，大小是 max_length，当总垄数超过这个数的时候，将其拆分开处理，但需要注意方向
        iteration = int(n // max_length)
        if n % max_length != 0:
            iteration += 1

        direction = False  # 耕作的方向，当为 true 表示向右，false 向左
        is_up = False  # 当前这一批垄是按照上还是下方向，如果 true 则垄的编号只可能是 1 2，false 则只可能是 3 4

        for i in range(iteration):
            is_up ^= True
            offset = i * max_length
            ind = 0
            run = True  # 控制内部循环是否继续跑

            while run:
                direction ^= True
                if direction:
                    if offset + ind + min_jump_swaths < n - 1:
                        if is_up:
                            swath[offset + ind] = 1
                        else:  # if not is_up
                            swath[offset + ind] = 3
                        ind = ind + min_jump_swaths + 1
                    else:
                        run = False
                else:  # direction == False
                    if swath[offset + ind - min_jump_swaths] == 0:
                        if is_up:
                            swath[offset + ind] = 2
                        else:  # is_up == False
                            swath[offset + ind] = 4
                        ind = ind - min_jump_swaths
                    else:
                        run = False

        return swath, min_jump_swaths

    @staticmethod
    def gen_fishtail_in_paths(line_gdf: gpd.GeoDataFrame, turning_radius: float, swath_width: float,
                              vehicle_length: float,
                              vehicle_width: float, centroid, theta: float, begin_side=1):
        """
        生成往复式的 fishtail 转向路径，即往复式耕作，每一调转车头方向的时候，通过 fishtail 的方式调转车头回来
        :param line_gdf: 地块内路径的 GeoDataFrame
        :param turning_radius: 农机的转向半径（往往是最小转向半径）
        :param swath_width: 一垄的宽度
        :param vehicle_length: 农机全长
        :param vehicle_width: 农机的宽（最宽处）
        :param centroid: 当前地块的中心，用于将 “水平旋转的地块” 旋转回原本的角度
        :param theta: 由原来地块内的角度旋转到水平角度、或从当前的水平角度旋转回原本的角度的值，前者 -theta，后者 theta
        :param begin_side: 表示当前的鱼尾从哪一边开始，如果为 -1或者1，则左边，2 为右边
        :return: GeoDataFrame
        """
        fishtail_1 = CPP_Planner_TurningRail_Maker.gen_fishtail_shape_curve(turning_radius, swath_width, 'right')
        fishtail_2 = CPP_Planner_TurningRail_Maker.gen_fishtail_shape_curve(turning_radius, swath_width, 'left')
        # 这里通过判断上一个 flat turn 结尾的方向，1 左 2 右
        if begin_side == 1:
            direction = True
        else:
            direction = False  # true right, false left
        fishtails = []
        for i in range(len(line_gdf) - 1):
            line = line_gdf.geometry.iloc[i]
            line_2 = line_gdf.geometry.iloc[i + 1]
            direction ^= True
            if direction:
                right_point = line.coords[0] if line.coords[0][0] > line.coords[-1][0] else line.coords[-1]
                right_point_2 = line_2.coords[0] if line_2.coords[0][0] > line_2.coords[-1][0] else line_2.coords[-1]
                # 因为 path 是从上到下扫描的，所以这里要留意坐标的位置，加一个 swath_width

                if right_point[0] > right_point_2[0]:
                    gap = right_point[0] - right_point_2[0]
                    compensate_line = LineString(
                        (right_point_2, (right_point_2[0] + gap + vehicle_length / 2, right_point_2[1])))
                    compensate_line_2 = LineString((right_point, (right_point[0] + vehicle_length / 2, right_point[1])))
                    temp_fishtails = fishtail_1.translate(xoff=right_point[0] + vehicle_length / 2,
                                                          yoff=right_point[1] + swath_width)
                    pass
                else:  # right_point[0] < right_point_2[0]
                    gap = right_point_2[0] - right_point[0]
                    compensate_line = LineString(
                        (right_point, (right_point[0] + gap + vehicle_length / 2, right_point[1])))
                    compensate_line_2 = LineString(
                        (right_point_2, (right_point_2[0] + vehicle_length / 2, right_point_2[1])))
                    temp_fishtails = fishtail_1.translate(xoff=right_point_2[0] + vehicle_length / 2,
                                                          yoff=right_point[1] + swath_width)
                    pass
                temp_fishtails = gpd.GeoDataFrame(
                    geometry=list(temp_fishtails.geometry) + [compensate_line, compensate_line_2])
                fishtails.append(temp_fishtails)
                pass
            else:  # direction == False
                left_point = line.coords[0] if line.coords[0][0] < line.coords[-1][0] else line.coords[-1]
                left_point_2 = line_2.coords[0] if line_2.coords[0][0] < line_2.coords[-1][0] else line_2.coords[-1]
                if left_point[0] < left_point_2[0]:
                    gap = left_point_2[0] - left_point[0]
                    compensate_line = LineString(
                        ((left_point[0] - vehicle_length / 2, left_point_2[1]), (left_point_2[0], left_point_2[1])))
                    compensate_line_2 = LineString((left_point, (left_point[0] - vehicle_length / 2, left_point[1])))
                    temp_fishtails = fishtail_2.translate(xoff=left_point[0] - vehicle_length / 2,
                                                          yoff=left_point[1] + swath_width)
                else:  # left_point[0] > left_point_2[0]
                    gap = left_point[0] - left_point_2[0]
                    compensate_line = LineString(
                        ((left_point_2[0] - vehicle_length / 2, left_point[1]), (left_point[0], left_point[1])))
                    compensate_line_2 = LineString(
                        (left_point_2, (left_point_2[0] - vehicle_length / 2, left_point_2[1])))
                    temp_fishtails = fishtail_2.translate(xoff=left_point[0] - vehicle_length / 2,
                                                          yoff=left_point[1] + swath_width)

                # fishtails.append(temp_fishtails)
                temp_fishtails = gpd.GeoDataFrame(
                    geometry=list(temp_fishtails.geometry) + [compensate_line, compensate_line_2])
                fishtails.append(temp_fishtails)
                fishtails.append(gpd.GeoDataFrame(geometry=[compensate_line]))
        return fishtails
        pass

    @staticmethod
    def gen_path_flat_turn_tail_turn(path: gpd.GeoDataFrame, turning_radius: float, vehicle_length: float,
                                     vehicle_width: float, swath_width: float, centroid, theta: float):
        """
        生成 flat turn 转向 + 鱼尾转向的路径，先处理 flat turn， 再处理鱼尾
        原则是：首先通过 flat turn 尽量耕作，再对无法通过 flat turn 转到的 垄进行 fishtail 转向
        * 每一垄路径的情况：
            * 0: 使用鱼尾转向
            * 1. 在上方向右到达最近可达垄 + 1垄  2. 在下方到达左侧最近可达垄  3. 在下方向右到达最近可达垄 + 1垄  4.在上方到达右侧最近可达垄
        :param path: 当前地块内的 垄 LineString
        :param turning_radius: 农机（最小）转弯半径
        :param vehicle_length: 农机全长
        :param vehicle_width: 农机（最宽处）宽度
        :param swath_width: 一条垄的宽度
        :param centroid: 当前地块的中心，用于将 “水平旋转的地块” 旋转回原本的角度
        :param theta: 由原来地块内的角度旋转到水平角度、或从当前的水平角度旋转回原本的角度的值，前者 -theta，后者 theta
        :return:
        """
        # 旋转地块到水平
        path = path.rotate(-theta, origin=centroid)

        tillage_method, min_jump_swath = CPP_Planner_TurningRail_Maker.gen_path_tillage_method(path, turning_radius,
                                                                                               vehicle_length,
                                                                                               vehicle_width,
                                                                                               swath_width)
        normalized_bow_curve_right \
            = CPP_Planner_TurningRail_Maker.gen_bow_shape_curve(turning_radius, min_jump_swath, swath_width,
                                                                'right', 1)
        normalized_bow_curve_right_2 = CPP_Planner_TurningRail_Maker.gen_bow_shape_curve(turning_radius, min_jump_swath, swath_width,
                                                                'right', 0)
        normalized_bow_curve_left \
            = CPP_Planner_TurningRail_Maker.gen_bow_shape_curve(turning_radius, min_jump_swath, swath_width,
                                                                'left', 0)
        normalized_bow_curve_left_2 \
            = CPP_Planner_TurningRail_Maker.gen_bow_shape_curve(turning_radius, min_jump_swath, swath_width,
                                                                'left', 1)

        normalized_bow_curve_right.set_crs(path.crs)
        normalized_bow_curve_left.set_crs(path.crs)
        turning_paths = []
        # flat turn，注意path是从下到上的，需要翻过来
        for i in range(len(tillage_method)):
            # i = len(tillage_method) - j - 1
            if tillage_method[i] == 1:
                # 找到需要放置的 swath 点，因为是右侧，所以需要找到 swath 中偏右的点
                line = path.geometry.iloc[i]
                line_2 = path.geometry.iloc[i + min_jump_swath + 1]
                # 确定 右侧 的点
                right_point = line.coords[0] if line.coords[0][0] > line.coords[-1][0] else line.coords[-1]
                right_point_2 = line_2.coords[0] if line_2.coords[0][0] > line_2.coords[-1][0] else line_2.coords[-1]

                # 确定哪一条线短一点，需要额外移动一段距离才能够得到 弓形 曲线
                if right_point[0] > right_point_2[0]:
                    gap = right_point[0] - right_point_2[0]
                    compensate_line = LineString(
                        (right_point_2, (right_point_2[0] + gap + vehicle_length / 2, right_point_2[1])))
                    compensate_line_2 = LineString((right_point, (right_point[0] + vehicle_length / 2, right_point[1])))
                    gap2 = 0
                else:  # right_point[0] < right_point_2[0]
                    gap = right_point_2[0] - right_point[0]
                    compensate_line = LineString(
                        (right_point, (right_point[0] + gap + vehicle_length / 2, right_point[1])))
                    compensate_line_2 = LineString(
                        (right_point_2, (right_point_2[0] + vehicle_length / 2, right_point_2[1])))
                    gap2 = gap

                # 将标准 弓形 曲线移动到指定的位置
                temp_bow_line = normalized_bow_curve_right.translate(xoff=right_point[0] + gap2 + vehicle_length / 2,
                                                                     yoff=right_point[1] + swath_width * (
                                                                             min_jump_swath + 1))
                temp_bow_line = gpd.GeoDataFrame(
                    geometry=list(temp_bow_line.geometry) + [compensate_line, compensate_line_2])
                turning_paths.append(temp_bow_line)

                # line_1 = path.geometry.iloc[i]
                # line_2 = path.geometry.iloc[i + min_jump_swath + 1]
                # right_points = line_1.coords[0] if line_1.coords[0][0] > line_1.coords[-1][0] else line_1.coords[-1]
                # right_points_2 = line_2.coords[0] if line_2.coords[0][0] > line_2.coords[-1][0] else line_2.coords[-1]
                # if right_points[0] < right_points_2[0]:
                #     padding_2 = LineString(
                #         (right_points, (right_points_2[0], right_points[1]))
                #     )
                # else:
                #     padding_2 = LineString(
                #         (right_points_2, (right_points[0], right_points_2[1]))
                #     )
                # turning_paths.append(padding_2)
                # curve_1 = CPP_Planner_TurningRail_Maker.gen_single_curve(
                #     (right_points[0], right_points[1] + turning_radius), turning_radius, 0.1, -pi / 2, 0
                # )
                # padding = LineString((
                #     (right_points[0] + turning_radius, right_points[1] + turning_radius),
                #     (right_points[0] + turning_radius, right_points[1] + turning_radius +
                #      (swath_width * (min_jump_swath + 1) - 2 * turning_radius))
                # ))
                # curve_2 = CPP_Planner_TurningRail_Maker.gen_single_curve(
                #     (right_points[0], right_points[1] + turning_radius +
                #      (swath_width * (min_jump_swath + 1) - 2 * turning_radius)),
                #     turning_radius, 0.1, 0, pi / 2
                # )
                # turning_paths.append(curve_1)
                # turning_paths.append(padding)
                # turning_paths.append(curve_2)
                pass
            elif tillage_method[i] == 2:
                line = path.geometry.iloc[i]
                line_2 = path.geometry.iloc[i - min_jump_swath]
                left_point = line.coords[0] if line.coords[0][0] < line.coords[-1][0] else line.coords[-1]
                left_point_2 = line_2.coords[0] if line_2.coords[0][0] < line_2.coords[-1][0] else line_2.coords[-1]
                # 比较两条线的长短，需要将 弓形曲线 放到长边处
                if left_point[0] < left_point_2[0]:
                    gap = left_point_2[0] - left_point[0]
                    compensate_line = \
                        LineString(
                            ((left_point[0] - vehicle_length / 2, left_point_2[1]), (left_point_2[0], left_point_2[1])))
                    compensate_line_2 = \
                        LineString(((left_point[0] - vehicle_length / 2, left_point[1]), left_point))
                    temp_bow_line = normalized_bow_curve_left.translate(xoff=left_point[0] - vehicle_length / 2,
                                                                        yoff=left_point[1])
                else:  # left_point[0] > left_point_2[0]
                    gap = left_point[0] - left_point_2[0]
                    compensate_line = \
                        LineString(
                            ((left_point_2[0] - vehicle_length / 2, left_point[1]), (left_point[0], left_point[1])))
                    compensate_line_2 = \
                        LineString(((left_point_2[0] - vehicle_length / 2, left_point_2[1]), left_point_2))
                    temp_bow_line = normalized_bow_curve_left.translate(xoff=left_point_2[0] - vehicle_length / 2,
                                                                        yoff=left_point_2[1] + (
                                                                                swath_width * min_jump_swath))

                # 移动 标准弓形曲线

                # temp_bow_line = normalized_bow_curve_left.translate(xoff=left_point[0], yoff=left_point[1])
                temp_bow_line = gpd.GeoDataFrame(
                    geometry=list(temp_bow_line.geometry) + [compensate_line, compensate_line_2])
                turning_paths.append(temp_bow_line)
                pass
            elif tillage_method[i] == 3:
                line = path.geometry.iloc[i]
                line_2 = path.geometry.iloc[i + min_jump_swath + 1]
                left_point = line.coords[0] if line.coords[0][0] < line.coords[-1][0] else line.coords[-1]
                left_point_2 = line_2.coords[0] if line_2.coords[0][0] < line_2.coords[-1][0] else line_2.coords[-1]
                if left_point[0] < left_point_2[0]:
                    gap = left_point_2[0] - left_point[0]
                    compensate_line = \
                        LineString(
                            ((left_point[0] - vehicle_length / 2, left_point[1]), (left_point[0], left_point[1])))
                    compensate_line_2 = \
                        LineString(((left_point_2[0] - gap - vehicle_length / 2, left_point_2[1]), left_point_2))
                    temp_bow_line = normalized_bow_curve_left_2.translate(xoff=left_point[0] - vehicle_length / 2,
                                                                        yoff=left_point_2[1])
                else:  # left_point[0] > left_point_2[0]
                    gap = left_point[0] - left_point_2[0]
                    compensate_line = \
                        LineString(
                            ((left_point_2[0] - vehicle_length / 2, left_point_2[1]), (left_point_2[0], left_point_2[1])))
                    compensate_line_2 = \
                        LineString(((left_point[0] - gap - vehicle_length / 2, left_point[1]), left_point))
                    temp_bow_line = normalized_bow_curve_left_2.translate(xoff=left_point_2[0] - vehicle_length / 2,
                                                                        yoff=left_point[1] + (
                                                                                swath_width * (min_jump_swath + 1)))
                    # 将标准 弓形 曲线移动到指定的位置
                temp_bow_line = gpd.GeoDataFrame(
                    geometry=list(temp_bow_line.geometry) + [compensate_line, compensate_line_2])
                # temp_bow_line = gpd.GeoDataFrame(
                #     geometry=list(temp_bow_line.geometry) + [compensate_line])
                turning_paths.append(temp_bow_line)

                pass
            elif tillage_method[i] == 4:
                line = path.geometry.iloc[i]
                line_2 = path.geometry.iloc[i - min_jump_swath]
                right_point = line.coords[0] if line.coords[0][0] > line.coords[-1][0] else line.coords[-1]
                right_point_2 = line_2.coords[0] if line_2.coords[0][0] > line_2.coords[-1][0] else line_2.coords[-1]

                # 确定哪一条线短一点，需要额外移动一段距离才能够得到 弓形 曲线
                if right_point[0] > right_point_2[0]:
                    gap = right_point[0] - right_point_2[0]
                    compensate_line = LineString(
                        (right_point_2, (right_point_2[0] + gap + vehicle_length / 2, right_point_2[1])))
                    compensate_line_2 = LineString((right_point, (right_point[0] + vehicle_length / 2, right_point[1])))
                    gap2 = 0
                else:  # right_point[0] < right_point_2[0]
                    gap = right_point_2[0] - right_point[0]
                    compensate_line = LineString(
                        (right_point, (right_point[0] + gap + vehicle_length / 2, right_point[1])))
                    compensate_line_2 = LineString(
                        (right_point_2, (right_point_2[0] + vehicle_length / 2, right_point_2[1])))
                    gap2 = gap

                # 将标准 弓形 曲线移动到指定的位置
                temp_bow_line = normalized_bow_curve_right_2.translate(xoff=right_point[0] + gap2 + vehicle_length / 2,
                                                                     yoff=right_point_2[1] + swath_width * (
                                                                             min_jump_swath))
                temp_bow_line = gpd.GeoDataFrame(
                    geometry=list(temp_bow_line.geometry) + [compensate_line, compensate_line_2])
                turning_paths.append(temp_bow_line)
                pass

        # fishtail turn
        begin = -1  # 记录0开始的位置
        temp_paths = []
        fishtail_paths = []
        begin_side = -1  # 记录开始的方向，这是根据前面的 flat turn 的方向来，如果为 2、4，则从左边开始，1、3从右，begin_side 的值1左2右
        print(tillage_method)
        for i in range(len(tillage_method)):
            if tillage_method[i] == 0:
                temp_paths.append(path.geometry.iloc[i])
                # 判断前一个路径的方向
                if begin_side == -1 and i - 1 >= 0:
                    if tillage_method[i - 1] == 2 or tillage_method[i - 1] == 4:
                        begin_side = 1
                    else:
                        begin_side = 2
            else:
                if len(temp_paths) > 1:
                    temp_fishtails = CPP_Planner_TurningRail_Maker.gen_fishtail_in_paths(gpd.GeoDataFrame(
                        geometry=temp_paths, crs=path.crs), turning_radius, swath_width, vehicle_length, vehicle_width,
                        centroid, theta, begin_side=begin_side
                    )
                    begin_side = -1
                    # fishtail_paths.append(temp_fishtails)
                    # 因为这里返回的 fishtails 就是 gdf
                    fishtail_paths += temp_fishtails
                    temp_fishtails = []
                begin = -1

        # 额外处理一次
        if len(temp_paths) > 1:
            temp_fishtails = CPP_Planner_TurningRail_Maker.gen_fishtail_in_paths(gpd.GeoDataFrame(
                geometry=temp_paths, crs=path.crs), turning_radius, swath_width, vehicle_length, vehicle_width,
                centroid, theta, begin_side=begin_side
            )
            # fishtail_paths.append(temp_fishtails)
            # 因为这里返回的 fishtails 就是 gdf
            fishtail_paths += temp_fishtails

        # 将所有的路径转回原来的角度
        for i in range(len(turning_paths)):
            turning_paths[i] = turning_paths[i].rotate(theta, origin=centroid)
        for i in range(len(fishtail_paths)):
            fishtail_paths[i] = fishtail_paths[i].rotate(theta, origin=centroid)
        return turning_paths, fishtail_paths
        pass

    @staticmethod
    def gen_headland_paths(headland: gpd.GeoDataFrame, swath_width: float, headland_width: float, headland_mode='left',
                           area_limit=10, tolerance=0.05):
        """
        生成地头区域转向路径的测试
        :param headland:
        :param swath_width:
        :param headland_width:
        :param headland_mode:
        :param area_limit:
        :param tolerance:
        :return:
        """
        # 去除面积小于 area_limit 的地头区域
        headland_poly = headland.geometry[0]
        headlands = []  # 符合面积规定的 地头区域
        if type(headland_poly) == Polygon:
            pass
        else:  # if type(headland_poly) == MultiPolygon
            for i in headland_poly.geoms:
                if i.area > area_limit:
                    split_headland = CPP_Planner_Kit.split_polygon_by_largest_area(i, tolerance=tolerance)
                    for single_headland in split_headland:
                        if single_headland.area > area_limit:
                            headlands.append(single_headland)

        # 进行地头区域的多边形分割
        # for i in headlands:
        #     split_headland = CPP_Planner_Kit.split_polygon_by_largest_area(i, tolerance=0.05)

        all_headland_path = []
        # 计算地头区域的路径
        for single_headland in headlands:
            headland_gdf = gpd.GeoDataFrame(geometry=[single_headland], crs=headland.crs)
            temp_headland_path, temp_headland_headland = \
                CPP_Algorithm_Optimizers.gen_path_with_minimum_headland_area_by_edge(
                    land=headland_gdf, step_size=swath_width, head_land_width=8, headland_mode=headland_mode,
                    compare_mode='headland', return_theta=False
                )
            all_headland_path.append(temp_headland_path)

        return all_headland_path
        pass

    @staticmethod
    def combine_headlands_gen_path(headland: list, swath_width: float, headland_width: float,
                                   headland_mode='left', area_limit=10.0, tolerance=0.05):
        """
        将一个完整田块内的所有地头区域，相连的地头进行整合，并再对其进行路径规划
        * 对于相邻田块的整合，需要使用到缓冲区来进行计算，如果相交则取并集
        :param headland:
        :param swath_width:
        :param headland_width:
        :param headland_mode:
        :param area_limit:
        :param tolerance:
        :return:
        """
        print("Combining headlands in one field...")
        # 将 headland 中所有的 multipolygon 转换为 polygon
        single_headlands = []
        for split_headland in headland:
            split_headland = split_headland.geometry[0]
            if type(split_headland) == shapely.MultiPolygon:
                for single_headland in split_headland.geoms:
                    if single_headland.area > area_limit:
                        single_headlands.append(single_headland.simplify(swath_width).buffer(0.1))
            elif type(split_headland) == shapely.Polygon:
                if split_headland.area > area_limit:
                    single_headlands.append(split_headland.simplify(swath_width).buffer(0.1))

        # 对相连的headland进行合并
        left, right = 0, 1
        while left < len(single_headlands) - 1:
            if single_headlands[left].intersects(single_headlands[right]):
                single_headlands[left] = ops.unary_union([single_headlands[left], single_headlands[right]])
                del single_headlands[right]
                left = 0
                right = 1

            else:  # 没有交集，那么 right 继续向后，或 left 向后
                right += 1
                if right == len(single_headlands):
                    left += 1
                    right = left + 1
        # 对所有的地块进行简化
        split_headlands = []
        for headland in single_headlands:
            temp_split_headlands = CPP_Planner_Kit.split_polygon_by_largest_area(headland, tolerance)
            for split_headland in temp_split_headlands:
                if split_headland.area > area_limit:
                    split_headland = gpd.GeoDataFrame(geometry=[split_headland])
                    split_headlands.append(split_headland)

        headland_paths = []
        headland_headlands = []
        # 对筛选、简化 留下的地头区域进行路径规划
        for split_headland in split_headlands:
            headland_path, headland_headland = CPP_Algorithm_Optimizers.gen_path_with_minimum_headland_area_by_edge(
                split_headland, swath_width, headland_width, turning_radius=4.5, headland_mode=headland_mode
            )
            headland_paths.append(headland_path)
            headland_headlands.append(headland_headland)

        return headland_paths

    """
    * 其他 *
    """
    pass
