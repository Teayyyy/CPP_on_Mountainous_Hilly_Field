from CPP_Planner import CPP_Planner_Kit, CPP_Algorithms
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

"""
用来一次性规划所有地块的路径规划，并显示到图上
"""

def get_all_land_path():
    all_land = CPP_Planner_Kit.load_fields('Scratch/test_Load_Shp/shp_file/村1地_全区.shp')
    num_land = len(all_land)
    print("田块的个数: ", num_land)

    all_land_paths = []
    along_long_edge = True
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
    plt.savefig('Saved_Result/CPP.pdf')

get_all_land_path()