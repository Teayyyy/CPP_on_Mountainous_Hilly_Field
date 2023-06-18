from CPP_Planner import CPP_Planner_Kit, CPP_Algorithms, CPP_Algorithm_Optimizers
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import datetime

"""
用来一次性规划所有地块的路径规划，并显示到图上
"""


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


# get_all_land_path()
# get_all_land_path_with_headland(swath_width=1.45, along_long_edge=True, headland_mode='left', head_land_width=6)
get_all_land_path_with_min_headland(swath_width=1.45, along_long_edge=True, head_land_width=6)
