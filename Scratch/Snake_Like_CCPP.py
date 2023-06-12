import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point


def calculate_coverage_path(polygon, step_size):
    min_x, min_y, max_x, max_y = polygon.bounds

    # Get the start point inside the polygon
    start_x, start_y = get_start_point(polygon)

    # Generate the coverage path
    path_points = [np.array([start_x, start_y])]
    current_x, current_y = start_x, start_y
    direction = 1  # 1 for right, -1 for left
    vertical_step = step_size

    while True:
        next_x = current_x + direction * step_size
        next_y = current_y + vertical_step

        if not polygon.contains(Point(next_x, next_y)):
            next_x = current_x
            next_y = current_y + vertical_step

            if not polygon.contains(Point(next_x, next_y)):
                if direction == 1:
                    direction = -1
                else:
                    direction = 1
                next_x = current_x + direction * step_size
                next_y = current_y + vertical_step

                if not polygon.contains(Point(next_x, next_y)):
                    # Check if the next point is within the polygon bounds
                    if min_x <= next_x <= max_x and min_y <= next_y <= max_y:
                        path_points.append(np.array([next_x, next_y]))
                    break

            vertical_step = -vertical_step

        path_points.append(np.array([next_x, next_y]))
        print(next_x, next_y)
        current_x, current_y = next_x, next_y

    return np.array(path_points)


def get_start_point(polygon):
    min_x, min_y, max_x, max_y = polygon.bounds
    start_x = min_x + (max_x - min_x) * 0.1
    start_y = min_y + (max_y - min_y) * 0.5
    return start_x, start_y


# Define the irregular polygon
vertices = [(0, 0), (200, 0), (300, 100), (300, 400), (100, 600), (0, 400)]
polygon = Polygon(vertices)

# Set the step size for coverage path
step_size = 1

# Calculate the coverage path
path_points = calculate_coverage_path(polygon, step_size)

# Visualize the polygon and coverage path
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.add_patch(plt.Polygon(vertices, fill=None, edgecolor='black'))
ax.plot(path_points[:, 0], path_points[:, 1], 'bo-')
ax.set_xlim(polygon.bounds[0], polygon.bounds[2])
ax.set_ylim(polygon.bounds[1], polygon.bounds[3])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Snake-like Coverage Path')
plt.grid(True)
plt.show()
