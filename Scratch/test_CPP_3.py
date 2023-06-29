import geopandas as gpd
from shapely import LineString, Polygon, Point
import shapely
from shapely import ops
import matplotlib.pyplot as plt
from CPP_Planner import CPP_Algorithms, CPP_Planner_Kit
from shapely import affinity

all_land = gpd.read_file('test_Load_Shp/shp_file/村1地_全区.shp')
all_land.geometry = all_land.geometry.apply(shapely.set_precision, grid_size=0.05)

single_land = CPP_Planner_Kit.get_single_shp(all_land, 6)
split_polygon = CPP_Planner_Kit.split_polygon_by_largest_area(single_land.geometry.iloc[0], tolerance=0.04)

# 仅拿去其中的一个地块来测试耕作方向
single_polygon = split_polygon[1]
single_polygon_angle = CPP_Planner_Kit.get_land_MABR_angle(single_polygon)
single_polygon = affinity.rotate(single_polygon, -single_polygon_angle, origin='centroid')
mabr = single_polygon.minimum_rotated_rectangle

corrected_swath_width = CPP_Planner_Kit.get_corrected_swath_width(swath_width=1.45, slope=6)
path, headland = CPP_Algorithms.scanline_algorithm_single_with_headland(
    land=gpd.GeoDataFrame(geometry=[single_polygon]), step_size=corrected_swath_width, along_long_edge=False,
    headland='left', head_land_width=6
)

# 显示
_, ax = plt.subplots()
ax.plot(*single_polygon.exterior.xy, 'b')
path.plot(ax=ax, color='y')
headland.plot(ax=ax, color='r')
plt.show()
