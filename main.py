"""
临时用做测试
"""

from CPP_Planner import CPP_Planner_Kit, CPP_Algorithms
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

all_land = CPP_Planner_Kit.load_fields('Scratch/test_Load_Shp/shp_file/村1地_全区.shp')
print("number of lands: ", len(all_land))

single_land = CPP_Planner_Kit.get_single_shp(all_land, 0)
path_line = CPP_Algorithms.scanline_algorithm_single(single_land, step_size=1.5)

scaled_polygon = single_land['geometry']

# 绘制路径和多边形地块
plt.figure(figsize=(8, 6))
fig, ax = plt.subplots()
color = ['r', 'b']
# scaled_polygon.plot(ax=ax, color='b')
single_land.plot(ax=ax, color='b')
path_line.plot(ax=ax, color='y', linewidth=1)
# plt.plot(*zip(*path_points), 'y-', label='Coverage Path')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Complete Coverage Path Planning')
# plt.legend()
plt.grid(True)
plt.show()
