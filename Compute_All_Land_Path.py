import shapely

from CPP_Planner import CPP_Planner_Kit, CPP_Algorithms, CPP_Algorithm_Optimizers, CPP_Planner_TurningRail_Maker
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import datetime
import warnings

"""
用来一次性规划所有地块的路径规划，并显示到图上
"""

warnings.filterwarnings('ignore')

mean_slope = [3.37, 7.38, 6.07, 3.94, 6.53, 7.46, 7.62]


def get_all_land_path():
    all_land = CPP_Planner_Kit.load_fields('Scratch/test_Load_Shp/shp_file/村1地_全区.shp')
    num_land = len(all_land)
    print("田块的个数: ", num_land)

    all_land_paths = []
    along_long_edge = False
    # 遍历所有的地块，同时保存其路径
    for index in range(num_land):
        single_land = CPP_Planner_Kit.get_single_shp(all_land, index)
        all_land_paths.append(CPP_Algorithms.scanline_algorithm_single_no_turn(single_land, step_size=1.3,
                                                                               along_long_edge=along_long_edge))

    # 绘制路径
    fig, ax = plt.subplots()
    all_land.plot(ax=ax)
    for temp_path in all_land_paths:
        temp_path.plot(ax=ax, color='y', linewidth=0.1)
    plt.xlabel('longitude'), plt.ylabel('latitude')
    plt.title('CPP')
    plt.grid(True)
    # 保存或展示
    # plt.show()
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig('Saved_Result/CPP_{}.pdf'.format(now))


def get_all_land_path_with_headland(swath_width, along_long_edge=True, headland_mode='none', head_land_width=6):
    all_land = CPP_Planner_Kit.load_fields('Scratch/test_Load_Shp/shp_file/村1地_全区.shp')
    num_land = len(all_land)
    print("田块的个数: ", num_land)

    all_land_path = []
    all_headland = []
    along_long_edge = True
    # 需要将每一个地块分解、规划路径
    for index in range(num_land):
        print("land: ", index + 1)
        single_land = CPP_Planner_Kit.get_single_shp(all_land, index)
        split_polygon = CPP_Planner_Kit.split_polygon_by_largest_area(single_land.geometry.iloc[0], tolerance=0.05)
        # print(type(split_polygon))
        # 对每一个分割开的多边形地块子块进行路径规划
        for polygon in split_polygon:
            if polygon.area > 20:
                polygon_land = gpd.GeoDataFrame(geometry=[polygon], crs=all_land.crs)
                t_path, t_headland = CPP_Algorithms.scanline_algorithm_single_with_headland(
                    polygon_land, step_size=swath_width, headland=headland_mode,
                    head_land_width=head_land_width)
                all_land_path.append(t_path)
                all_headland.append(t_headland)

    # 显示所有的地块和地头
    fig, ax = plt.subplots()
    # all_land.plot(ax=ax, color='b')
    exte = all_land.exterior
    exte.plot(ax=ax, color='b', linewidth=0.3)
    for path in all_land_path:
        path.plot(ax=ax, color='y', linewidth=0.2)
    for headland in all_headland:
        headland.plot(ax=ax, color='gray')

    # plt.show()
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig('Saved_Result/CPP_{}.pdf'.format(now))


def get_all_land_path_with_min_headland(swath_width, along_long_edge=True, head_land_width=6):
    all_land = CPP_Planner_Kit.load_fields('Scratch/test_Load_Shp/shp_file/村1地_全区.shp')
    num_land = len(all_land)
    print("田块的个数: ", num_land)

    all_land_path = []
    all_headland = []
    along_long_edge = True
    # 需要将每一个地块分解、规划路径
    for index in range(num_land):
        print("land: ", index + 1)
        single_land = CPP_Planner_Kit.get_single_shp(all_land, index)
        split_polygon = CPP_Planner_Kit.split_polygon_by_largest_area(single_land.geometry.iloc[0], tolerance=0.05)
        # print(type(split_polygon))
        # 对每一个分割开的多边形地块子块进行路径规划
        for polygon in split_polygon:
            if polygon.area > 20:
                polygon_land = gpd.GeoDataFrame(geometry=[polygon], crs=all_land.crs)
                # t_path, t_headland = CPP_Algorithms.scanline_algorithm_single_with_headland(
                #     polygon_land, step_size=swath_width, headland=headland_mode,
                #     head_land_width=head_land_width)
                t_path, t_headland = CPP_Algorithm_Optimizers.gen_path_with_minimum_headland_area(
                    polygon_land, step_size=swath_width, head_land_width=head_land_width)
                all_land_path.append(t_path)
                all_headland.append(t_headland)

    # 显示所有的地块和地头
    fig, ax = plt.subplots()
    # all_land.plot(ax=ax, color='b')
    exte = all_land.exterior
    exte.plot(ax=ax, color='b', linewidth=0.3)
    for path in all_land_path:
        path.plot(ax=ax, color='y', linewidth=0.2)
    for headland in all_headland:
        headland.plot(ax=ax, color='gray')

    # plt.show()
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig('Saved_Result/CPP_{}.pdf'.format(now))


def get_all_land_with_min_headland_2(swath_width, head_land_width=6):
    all_land = CPP_Planner_Kit.load_fields('Scratch/test_Load_Shp/shp_file/村1地_全区.shp')
    num_land = len(all_land)
    print("田块的个数: ", num_land)

    all_land_path = []
    all_headland = []
    # 需要将每一个地块分解、规划路径
    for index in range(num_land):
        print("land: ", index + 1)
        slope = mean_slope[index]
        print("field Slope: ", slope)
        this_swath_width = CPP_Planner_Kit.get_corrected_swath_width(swath_width, slope)
        single_land = CPP_Planner_Kit.get_single_shp(all_land, index)
        split_polygon = CPP_Planner_Kit.split_polygon_by_largest_area(single_land.geometry.iloc[0], tolerance=0.05)
        # print(type(split_polygon))
        # 对每一个分割开的多边形地块子块进行路径规划
        for polygon in split_polygon:
            if polygon.area > 20:
                polygon_land = gpd.GeoDataFrame(geometry=[polygon], crs=all_land.crs)
                # t_path, t_headland = CPP_Algorithm_Optimizers.gen_path_with_minimum_headland_area(
                #     polygon_land, step_size=swath_width, head_land_width=head_land_width)
                t_path, t_headland = CPP_Algorithm_Optimizers.gen_path_with_minimum_headland_area_by_direction(
                    land=polygon_land, step_size=this_swath_width, head_land_width=head_land_width
                )
                all_land_path.append(t_path)
                all_headland.append(t_headland)

    # 显示所有的地块和地头
    fig, ax = plt.subplots()
    # all_land.plot(ax=ax, color='b')
    exte = all_land.exterior
    exte.plot(ax=ax, color='b', linewidth=0.3)
    for path in all_land_path:
        path.plot(ax=ax, color='y', linewidth=0.2)
    for headland in all_headland:
        headland.plot(ax=ax, color='gray')

    # plt.show()
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig('Saved_Result/CPP_minareadi_{}.pdf'.format(now))


def get_all_land_with_min_headland_edge(swath_width, head_land_width=6):
    # 通过计算最小边来获得路径
    all_land = CPP_Planner_Kit.load_fields('Scratch/test_Load_Shp/shp_file/村1地_全区.shp')
    num_land = len(all_land)
    print("田块的个数: ", num_land)

    all_land_path = []
    all_headland = []
    for index in range(num_land):
        print("Land no: ", index + 1)
        slope = mean_slope[index]
        temp_corrected_swath_length = CPP_Planner_Kit.get_corrected_swath_width(swath_width, slope)
        single_land = CPP_Planner_Kit.get_single_shp(all_land, index)
        # 分割田块为简单的多边形
        split_polygon = CPP_Planner_Kit.split_polygon_by_largest_area(single_land.geometry.iloc[0], tolerance=0.05)
        # 对分割的每一个多边形进行路径规划，这一次使用的是从多边形的每一边开始顺着边开始计算路径，找到各边中地头占据面积最小的区域
        for polygon in split_polygon:
            if polygon.area > 20:
                # 计算当前多边形的每一条边，顺着进行路径规划，找到地头占据面积最小的那一条边
                polygon_regen = gpd.GeoDataFrame(geometry=[polygon], crs=all_land.crs)
                temp_path, temp_headland = CPP_Algorithm_Optimizers.gen_path_with_minimum_headland_area_by_edge(
                    land=polygon_regen, step_size=temp_corrected_swath_length, head_land_width=6,
                    headland_mode='left', compare_mode='headland'
                )
                all_land_path.append(temp_path)
                all_headland.append(temp_headland)

    # 显示所有的地头和路径在地块内，或保存
    _, ax = plt.subplots()
    exte = all_land.exterior
    exte.plot(ax=ax, color='b', linewidth=0.3)
    for path in all_land_path:
        path.plot(ax=ax, color='y', linewidth=0.2)
    for headland in all_headland:
        headland.plot(ax=ax, color='gray')

    # plt.show()
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig('Saved_Result/CPP_min_edge_{}.pdf'.format(now))
    print("Saving Name: ", 'CPP_min_edge_{}.pdf'.format(now))


def get_all_land_with_min_headland_edge_with_turnings(swath_width):
    turning_radius = 4.5
    vehicle_length = 6.3
    vehicle_width = 1.9
    headland_width, vehicle_theta = CPP_Planner_Kit.calc_headland_width(turning_radius, swath_width, vehicle_length,
                                                                        vehicle_width,
                                                                        buffer=0.3, show_info=True)

    all_land = CPP_Planner_Kit.load_fields('Scratch/test_Load_Shp/shp_file/村1地_全区.shp')
    num_land = len(all_land)
    print("田块的个数: ", num_land)

    all_land_path = []
    all_headland = []
    all_headland_move = []
    all_turning_curves = []
    all_backward_moves = []
    for index in range(num_land):
        print("Land no: ", index + 1)
        slope = mean_slope[index]
        temp_corrected_swath_length = CPP_Planner_Kit.get_corrected_swath_width(swath_width, slope)
        single_land = CPP_Planner_Kit.get_single_shp(all_land, index)
        # 分割田块为简单的多边形
        split_polygon = CPP_Planner_Kit.split_polygon_by_largest_area(single_land.geometry.iloc[0], tolerance=0.05)
        # 对分割的每一个多边形进行路径规划，这一次使用的是从多边形的每一边开始顺着边开始计算路径，找到各边中地头占据面积最小的区域
        for polygon in split_polygon:
            if polygon.area > 20:
                # 计算当前多边形的每一条边，顺着进行路径规划，找到地头占据面积最小的那一条边
                polygon_regen = gpd.GeoDataFrame(geometry=[polygon], crs=all_land.crs)
                polygon_centroid = polygon_regen.centroid[0]
                # TODO: 检查 theta 是否正确
                temp_path, temp_headland, theta = CPP_Algorithm_Optimizers.gen_path_with_minimum_headland_area_by_edge(
                    land=polygon_regen, step_size=temp_corrected_swath_length, head_land_width=headland_width,
                    headland_mode='right', compare_mode='headland', return_theta=True
                )
                print(theta)
                # 计算地头的转向等路径
                forward_moves, turning_curves, backward_moves = CPP_Planner_TurningRail_Maker.gen_S_turning_paths_in_polygon(
                    temp_path, theta, turning_radius, vehicle_length, vehicle_width, swath_width, headland_width,
                    polygon_centroid, 0.1
                )

                all_headland_move.append(forward_moves)
                all_turning_curves.append(turning_curves)
                all_backward_moves.append(backward_moves)
                all_land_path.append(temp_path)
                all_headland.append(temp_headland)

    # 显示所有的地头和路径在地块内，或保存
    _, ax = plt.subplots()
    exte = all_land.exterior
    exte.plot(ax=ax, color='b', linewidth=0.3)
    for path in all_land_path:
        path.plot(ax=ax, color='y', linewidth=0.2)
    for headland in all_headland:
        headland.plot(ax=ax, color='gray')
    for headland_move in all_headland_move:
        headland_move.plot(ax=ax, color='orange', linewidth=0.1)
    for turning_line in all_turning_curves:
        turning_line.plot(ax=ax, color='green', linewidth=0.1)
    for backward_line in all_backward_moves:
        backward_line.plot(ax=ax, color='green', linewidth=0.1)

    # plt.show()
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig('Saved_Result/CPP_min_edge_turning_{}.pdf'.format(now))
    print("Saving Name: ", 'CPP_min_edge_turning_{}.pdf'.format(now))
    pass


def get_all_land_with_min_headland_edge_with_flat_fishtail_turnings(swath_width):
    turning_radius = 4.5
    # vehicle_length = 6.3
    vehicle_length = 4.5
    vehicle_width = 1.9
    headland_width, vehicle_theta = CPP_Planner_Kit.calc_headland_width(turning_radius, swath_width, vehicle_length,
                                                                        vehicle_width,
                                                                        buffer=0.3, show_info=True)
    # TODO: 计算 headland 的宽度，用 d = r * (1 + cos(a)) + swath_width / 2 * min_jump+swath
    all_land = CPP_Planner_Kit.load_fields('Scratch/test_Load_Shp/shp_file/村1地_全区.shp')
    num_land = len(all_land)
    print("田块的个数: ", num_land)

    all_land_path = []
    all_headland = []
    all_flat_turns = []
    all_fishtail_turns = []
    for index in range(num_land):
        print("Land no: ", index + 1)
        slope = mean_slope[index]
        temp_corrected_swath_length = CPP_Planner_Kit.get_corrected_swath_width(swath_width, slope)
        single_land = CPP_Planner_Kit.get_single_shp(all_land, index)
        # 分割田块为简单的多边形
        split_polygon = CPP_Planner_Kit.split_polygon_by_largest_area(single_land.geometry.iloc[0], tolerance=0.05)
        # 对分割的每一个多边形进行路径规划，这一次使用的是从多边形的每一边开始顺着边开始计算路径，找到各边中地头占据面积最小的区域
        for polygon in split_polygon:
            if polygon.area > 20:
                # 计算当前多边形的每一条边，顺着进行路径规划，找到地头占据面积最小的那一条边
                polygon_regen = gpd.GeoDataFrame(geometry=[polygon], crs=all_land.crs)
                polygon_centroid = polygon_regen.centroid[0]
                temp_path, temp_headland, theta = CPP_Algorithm_Optimizers.gen_path_with_minimum_headland_area_by_edge(
                    land=polygon_regen, step_size=temp_corrected_swath_length, head_land_width=headland_width,
                    headland_mode='both', compare_mode='headland', return_theta=True
                )
                print(theta)
                # 计算地头的转向等路径
                flat_turn_paths, fishtail_paths = CPP_Planner_TurningRail_Maker.gen_path_flat_turn_tail_turn(
                    temp_path, turning_radius, vehicle_length, vehicle_width, swath_width, polygon_centroid, theta
                )
                all_flat_turns += flat_turn_paths
                all_fishtail_turns += fishtail_paths

                all_land_path.append(temp_path)
                all_headland.append(temp_headland)

    # 显示所有的地头和路径在地块内，或保存
    _, ax = plt.subplots()
    exte = all_land.exterior
    exte.plot(ax=ax, color='b', linewidth=0.3)
    for path in all_land_path:
        path.plot(ax=ax, color='y', linewidth=0.2)
    for headland in all_headland:
        headland.plot(ax=ax, color='gray')
    for fishtail_turn in all_fishtail_turns:
        fishtail_turn.plot(ax=ax, color='g', linewidth=0.1)
    for flat_turn in all_flat_turns:
        flat_turn.plot(ax=ax, color='black', linewidth=0.1)
    # for headland_move in all_headland_move:
    #     headland_move.plot(ax=ax, color='orange', linewidth=0.1)
    # for turning_line in all_turning_curves:
    #     turning_line.plot(ax=ax, color='green', linewidth=0.1)
    # for backward_line in all_backward_moves:
    #     backward_line.plot(ax=ax, color='green', linewidth=0.1)

    # plt.show()
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig('Saved_Result/CPP_min_edge_turning_{}.pdf'.format(now))
    print("Saving Name: ", 'CPP_min_edge_fishtail_flatturn_{}.pdf'.format(now))
    pass


def get_all_land_with_min_headland_edge_with_headland_path(swath_width):
    # get_all_land_with_min_headland_edge_with_flat_fishtail_turnings
    turning_radius = 4.5
    # vehicle_length = 6.3
    vehicle_length = 4.5
    vehicle_width = 1.9
    headland_width, vehicle_theta = CPP_Planner_Kit.calc_headland_width(turning_radius, swath_width, vehicle_length,
                                                                        vehicle_width,
                                                                        buffer=0.3, show_info=True)

    # all_land = CPP_Planner_Kit.load_fields('Scratch/test_Load_Shp/shp_file/村1地_全区.shp')
    all_land = CPP_Planner_Kit.load_fields('Scratch/test_Load_Shp/shp_file/村地和道路/村地地块区域.shp')
    num_land = len(all_land)
    print("田块的个数: ", num_land)

    all_land_path = []
    all_headland = []
    all_headland_path = []
    all_flat_turns = []
    all_fishtail_turns = []
    for index in range(num_land):
        print("Land no: ", index + 1)
        slope = mean_slope[index]
        temp_corrected_swath_length = CPP_Planner_Kit.get_corrected_swath_width(swath_width, slope)
        single_land = CPP_Planner_Kit.get_single_shp(all_land, index)
        # 分割田块为简单的多边形
        split_polygon = CPP_Planner_Kit.split_polygon_by_largest_area(single_land.geometry.iloc[0], tolerance=0.05)
        # 对分割的每一个多边形进行路径规划，这一次使用的是从多边形的每一边开始顺着边开始计算路径，找到各边中地头占据面积最小的区域
        temp_headlands = []
        for polygon in split_polygon:
            if polygon.area > 20:
                # 计算当前多边形的每一条边，顺着进行路径规划，找到地头占据面积最小的那一条边
                polygon_regen = gpd.GeoDataFrame(geometry=[polygon], crs=all_land.crs)
                polygon_centroid = polygon_regen.centroid[0]
                temp_path, temp_headland, theta = CPP_Algorithm_Optimizers.gen_path_with_minimum_headland_area_by_edge(
                    land=polygon_regen, step_size=temp_corrected_swath_length, head_land_width=headland_width,
                    headland_mode='both', compare_mode='headland', return_theta=True
                )
                print(theta)

                # 计算地头的转向等路径
                if temp_headland.geometry[0] != None:
                    temp_headlands.append(temp_headland)
                    # all_land_path += CPP_Planner_TurningRail_Maker.gen_headland_paths(temp_headland, swath_width,
                    #                                                                   headland_width, area_limit=7)
                    # all_headland_path += CPP_Planner_TurningRail_Maker.combine_headlands_gen_path(
                    #     temp_headland, swath_width, headland_width, 'none', area_limit=20
                    # )

                flat_turn_paths, fishtail_paths = CPP_Planner_TurningRail_Maker.gen_path_flat_turn_tail_turn(
                    temp_path, turning_radius, vehicle_length, vehicle_width, swath_width, polygon_centroid, theta
                )
                all_flat_turns += flat_turn_paths
                all_fishtail_turns += fishtail_paths

                all_land_path.append(temp_path)
                all_headland.append(temp_headland)

        headland_width_oneside, _ = CPP_Planner_Kit.calc_headland_width(turning_radius, swath_width, vehicle_length,
                                                                     vehicle_width, buffer=0.2)
        all_headland_path += CPP_Planner_TurningRail_Maker.combine_headlands_gen_path(
            temp_headlands, swath_width, headland_width=headland_width_oneside, headland_mode='none', area_limit=20
        )

    # 显示所有的地头和路径在地块内，或保存
    _, ax = plt.subplots()
    exte = all_land.exterior
    exte.plot(ax=ax, color='b', linewidth=0.3)
    for path in all_land_path:
        path.plot(ax=ax, color='y', linewidth=0.2)
    for headland in all_headland:
        headland.plot(ax=ax, color='gray')
    for fishtail_turn in all_fishtail_turns:
        fishtail_turn.plot(ax=ax, color='g', linewidth=0.1)
    for flat_turn in all_flat_turns:
        flat_turn.plot(ax=ax, color='black', linewidth=0.1)
    for temp_headlands in all_headland_path:
        temp_headlands.plot(ax=ax, color='y', linewidth=0.1)

    # plt.show()
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig('Saved_Result/CPP_min_edge_turning_{}.pdf'.format(now))
    print("Saving Name: ", 'CPP_min_edge_fishtail_flatturn_headland_path{}.pdf'.format(now))
    pass


# correct_swath_width = CPP_Planner_Kit.get_corrected_swath_width(swath_width=1.45, slope=6)
# get_all_land_path()
# get_all_land_path_with_headland(swath_width=1.45, along_long_edge=True, headland_mode='left', head_land_width=6)
# get_all_land_path_with_min_headland(swath_width=1.45, along_long_edge=True, head_land_width=6)
# get_all_land_with_min_headland_2(swath_width=1.45, head_land_width=6)
# get_all_land_with_min_headland_edge(swath_width=1.45, head_land_width=6)
# get_all_land_with_min_headland_edge_with_turnings(1.45)
# get_all_land_with_min_headland_edge_with_flat_fishtail_turnings(1.45)
get_all_land_with_min_headland_edge_with_headland_path(1.45)
