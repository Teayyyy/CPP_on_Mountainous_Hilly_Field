import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry
from shapely.geometry import Polygon, LineString
from shapely import affinity
from shapely import ops
import geopandas as gpd
import math
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
                                                headland='none', head_land_width=0, min_path_length=5,
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
        # 目前无法解决锯齿的问题
        # if len(path_gdf.geometry) > 1:
        #     path_area_union = CPP_Planner_Kit.get_path_bound(path_gdf)
        # else:
        #     path_area_union = gpd.GeoDataFrame(geometry=[], crs=land.crs)
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
    def gen_path_with_minimum_headland_area_by_edge(land: gpd.GeoDataFrame, step_size: float, head_land_width=6,
                                                    headland_mode='left', compare_mode='headland'):
        """
        通过从当前的地块（保证当前的地块形状接近凸边形）的每一条边顺着耕作出发，预留耕作的地头，通过比较顺着各边生成的地头的面积，以及耕作路径的
        长度，找到最优解，可以是耕作路径最长，也可以是耕地面积最大
        * 需要注意，当耕地一个方向的长度低于 headland_width 时，算法会返回 nan，需要过滤掉这种情况
        :param land: 进行路径规划的地块，要求为接近 凸边形 的多边形，接近的程度为 tolerance，默认为0.05
        :param step_size: 耕作幅宽，需要根据当前地块所处的平均坡度进行修正
        :param head_land_width: 地头区域的宽度，需要保证机械能够在这里做出想要的例如：掉头、转向等动作
        :param headland_mode: 生成地头的模式，可以是 两侧（both），左侧（left），右侧（right）和不生成（none），默认为左侧
        :param compare_mode: 比较的模式，分别为比较路径的长度，和地头的面积（默认）
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
            temp_path, temp_headland = CPP_Algorithms.scanline_algorithm_single_with_headland(
                land=temp_rotated_polygon_regen, step_size=step_size, along_long_edge=False,
                headland=headland_mode, head_land_width=head_land_width, get_largest_headland=False
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
            headland_areas.append(temp_headland.area.item() if not math.isnan(temp_headland.area.item())
                                  else max_headland_area)

        # 选择最小的地头面积或是最长的耕作路径
        if compare_mode == 'headland':
            selected_edge_index = headland_areas.index(min(headland_areas))
        else:  # if compare_mode == 'path'
            selected_edge_index = path_lengths.index(max(path_lengths))

        # 返回对应最优的耕作路径和地头区域
        return path_headland_collection[selected_edge_index][0], path_headland_collection[selected_edge_index][1]


# end class Algorithm_Optimizers ------------------------------------------------------------


"""
这个类包含了转向轨迹的生成方式
"""


class CPP_Planner_TurningRail_Maker:
    pass
