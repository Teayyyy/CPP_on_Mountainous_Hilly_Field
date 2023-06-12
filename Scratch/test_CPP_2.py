import pygame
import random

# 定义窗口尺寸
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# 定义多边形地块的顶点坐标
polygon_vertices = [(100, 100), (300, 100), (400, 200), (300, 300), (100, 300)]

# 初始化Pygame
pygame.init()
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
clock = pygame.time.Clock()

# 定义机器人初始位置和移动速度
robot_position = [polygon_vertices[0][0], polygon_vertices[0][1]]
robot_speed = 2

# 定义目标点初始位置和半径
target_position = [polygon_vertices[0][0], polygon_vertices[0][1]]
target_radius = 10

# 定义路径列表，用于存储已经覆盖的路径点
path_points = [robot_position.copy()]

# 定义往复移动的标志和方向
move_forward = True

# 游戏循环
running = True
while running:
    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 更新机器人位置
    if move_forward:
        # 沿着路径前进
        index = path_points.index(robot_position)
        if index + 1 < len(path_points):
            robot_position[0] = path_points[index + 1][0]
            robot_position[1] = path_points[index + 1][1]
        else:
            # 到达最后一个路径点，开始后退
            move_forward = False
    else:
        # 沿着路径后退
        index = path_points.index(robot_position)
        if index - 1 >= 0:
            robot_position[0] = path_points[index - 1][0]
            robot_position[1] = path_points[index - 1][1]
        else:
            # 返回到起始位置，开始前进
            move_forward = True

    # 更新目标点位置
    if robot_position[0] == target_position[0] and robot_position[1] == target_position[1]:
        # 生成新的随机目标点
        target_position[0] = random.randint(0, WINDOW_WIDTH)
        target_position[1] = random.randint(0, WINDOW_HEIGHT)

    # 更新路径列表
    path_points.append(robot_position.copy())

    # 绘制场景
    window.fill((255, 255, 255))  # 白色背景
    pygame.draw.polygon(window, (0, 0, 0), polygon_vertices, 1)  # 绘制多边形地块
    pygame.draw.circle(window, (255, 0, 0), target_position, target_radius)  # 绘制目标点
    pygame.draw.rect(window, (0, 0, 255), (robot_position[0], robot_position[1], 20, 20))  # 绘制机器人

    # 刷新窗口
    pygame.display.flip()
    clock.tick(60)  # 控制帧率

# 退出程序
pygame.quit()
