"""
人工势场法（Artificial Potential Field，APF）是一种基于势场概念的路径规划方法。它将机器人或车辆的移动路径建模为在环境中受到吸引力和斥力的作用下移动的粒子。具体而言，目标点产生吸引力，障碍物产生斥力，机器人或车辆通过在势场中感受力的方向和大小来确定移动方向。

APF路径规划算法的基本思路如下：

1. 根据环境中的障碍物信息，构建势场模型。障碍物产生斥力，目标点产生吸引力，可以根据距离的远近和障碍物的大小调整斥力和吸引力的强度。
2. 在势场中，机器人或车辆感受到合力的作用，力的方向指向吸引力或远离斥力的方向，力的大小反映了在该点的受力大小。
3. 根据力的方向和大小，计算机器人或车辆的移动方向和速度。
4. 迭代更新机器人或车辆的位置，直到到达目标点或达到终止条件。

似乎只能用于点到点规划，无法实现
"""

import numpy as np
import matplotlib.pyplot as plt


def calculate_potential_gradient(position, targets, obstacles):
    attractive_force = np.zeros_like(position, dtype=float)
    for target in targets:
        attractive_force += target - position

    repulsive_force = np.zeros_like(position, dtype=float)
    for obstacle in obstacles:
        obstacle_vector = position - obstacle
        distance = np.linalg.norm(obstacle_vector)
        repulsive_force += (obstacle_vector / distance) * repulsive_force_magnitude(distance)

    return attractive_force + repulsive_force


def repulsive_force_magnitude(distance, epsilon=1e-6):
    if distance < epsilon:
        return np.inf
    else:
        return 1.0 / distance


def ccpp_path_planning(start, targets, obstacles, max_iterations=1000, step_size=0.1):
    position = start
    path = [np.array(position, dtype=float)]
    direction = np.array([1., 0.])  # 初始方向

    for _ in range(max_iterations):
        gradient = calculate_potential_gradient(position, targets, obstacles)
        if np.linalg.norm(gradient) < 1e-6:
            break

        # 往复行进
        position += step_size * direction

        # 检查是否到达目标点
        for target in targets:
            if np.linalg.norm(position - target) < 1e-6:
                direction *= -1  # 调转方向
                break

        path.append(position)

    return np.array(path)


# 测试
start = np.array([0., 0.])
targets = [np.array([3., 3.]), np.array([6., 3.]), np.array([6., 6.]), np.array([3., 6.])]
obstacles = [np.array([2., 2.]), np.array([4., 4.]), np.array([5., 2.]), np.array([4., 6.])]

path = ccpp_path_planning(start, targets, obstacles)

# 可视化路径和障碍物
plt.figure()
plt.plot(start[0], start[1], 'go', label='Start')
plt.plot(targets[0][0], targets[0][1], 'ro', label='Target')
plt.plot(path[:, 0], path[:, 1], 'b-', label='Path', linewidth=2)
for target in targets:
    plt.plot(target[0], target[1], 'ro')
for obstacle in obstacles:
    plt.plot(obstacle[0], obstacle[1], 'ks', label='Obstacle')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('CCPP Path Planning')
plt.grid(True)
plt.axis('equal')
plt.show()
