# """
# 临时用做测试，所有功能实现见 CPP_Planner.py，规划整个地块的实现见：Compute_All_Land_Path.py
# """
#
from CPP_Planner import CPP_Planner_Kit, CPP_Algorithms
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from CPP_Planner import CPP_Planner_Kit, CPP_Algorithms, CPP_Algorithm_Optimizers, CPP_Planner_TurningRail_Maker


#
# all_land = CPP_Planner_Kit.load_fields('Scratch/test_Load_Shp/shp_file/村1地_全区.shp')
# print("number of lands: ", len(all_land))
#
# single_land = CPP_Planner_Kit.get_single_shp(all_land, 0)
# path_line = CPP_Algorithms.scanline_algorithm_single(single_land, step_size=1.5)
#
# scaled_polygon = single_land['geometry']
#
# # 绘制路径和多边形地块
# plt.figure(figsize=(8, 6))
# fig, ax = plt.subplots()
# color = ['r', 'b']
# # scaled_polygon.plot(ax=ax, color='b')
# angle = CPP_Planner_Kit.get_land_MABR_angle(scaled_polygon[0])
# # single_land = single_land.rotate(-angle, 'centroid')
# single_land.plot(ax=ax, color='b')
# path_line.plot(ax=ax, color='y', linewidth=1)
# # plt.plot(*zip(*path_points), 'y-', label='Coverage Path')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Complete Coverage Path Planning')
# # plt.legend()
# plt.grid(True)
# plt.show()

def test_gen_path_tillage_type(t_path: gpd.GeoDataFrame, t_turning_radius: float, t_vehicle_length: float,
                               t_vehicle_width: float, t_swath_width: float):
    # 由于田块内的耕作垄是等距离的，直接抽象成点集合（横向从左到右，垄是竖立的）
    # 有三种情况：0 使用鱼尾转向，1 向右间隔 最近可达垄+1垄 耕作，2 向左最近可达垄耕作，反向：3 4
    # n = len(t_path)
    n = 16
    swaths = [0 for i in range(n)]
    min_jump_swaths = int(CPP_Planner_TurningRail_Maker.calc_min_swath_jump(t_turning_radius, t_swath_width))
    # 使用 “平滑转向” 下，最大一次可以完整耕作的垄数（由于是 0 开始，减一）
    max_length = int((min_jump_swaths + 1) * 2 - 1)
    iteration = int(n // max_length)
    print(iteration)
    if n % max_length != 0:
        iteration += 1
    direction = False  # 当为 true 表示 向右（正向），false 向左（反向）
    is_up = False  # 当作方向， true 表示上方 右，下方 左，false 表示，下方 右，上方左  为了将方向连接起来
    for i in range(iteration):
        is_up ^= True
        offset = i * max_length
        ind = 0
        run = True
        # for j in range(max_length):
        while run:
            direction ^= True
            if direction:
                if offset + ind + min_jump_swaths < n - 1:
                    if is_up:
                        swaths[offset + ind] = 1
                    else:
                        swaths[offset + ind] = 3
                    ind = ind + min_jump_swaths + 1
                else:
                    run = False
                pass
            else:  # direction == False
                if swaths[offset + ind - min_jump_swaths] == 0:  # direction == False
                    if is_up:
                        swaths[offset + ind] = 2
                    else:
                        swaths[offset + ind] = 4
                    ind = ind - min_jump_swaths
                else:
                    run = False
                pass
    print(swaths)
    pass


test_gen_path_tillage_type(None, 4.5, 6.3, 1.9, 1.45)

print(CPP_Planner_TurningRail_Maker.gen_path_tillage_method(None, 4.5, 6.3, 1.9, 1.45))
