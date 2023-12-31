# 课题：山地农田路径规划的测试代码
### 指南
* 路径规划算法和相关的处理小工具：CPP_Planner.py
* 计算当前 .shp 文件下所有田块的路径 Compute_All_Land_Path.py
  * .shp 文件要求：保证所有的 .shp 文件在使用前保持仅包含地块的区域信息（若是需要考虑地块道路的信息，后续开发）
  * 文件会保存在当前 py 文件目录下的 Saved_Result/CPP_{time}.pdf，time为当前时间
* 通过 Neural Network 对真实路径进行模拟，修正生成的路径：pathline_real_modeling


### 尚未完成 （大功能）
#### TODO: 1. 考虑机器的转向半径，使得路径规划的结果更加符合实际 -- done 完成了单向倒车路径的规划
    TODO: 1.2 实现双向耕作路径的规划，需要考虑转向半径对“双向掉头”的影响
#### TODO: 2. 能够实现GIS坐标和像素坐标的转换，使得路径规划的结果能够在地图上显示 -- done，前提时GIS的坐标使用米为单位
#### TODO: 3. 考虑地块的形状，选择转向等效率最高的路径规划方案
#### TODO: 4. 考虑在弯曲非直的地块上规划路径，这要求路径能够弯曲地覆盖整个地块 -- done, 通过将原来的多边形凸分割实现
#### TODO: 5. 考虑坡度，即三维情况下的路径规划 -- temply done, 通过考虑整个地块的坡度
* 5.1 考虑坡度坡向对路径规划的影响
#### TODO: 6. 考虑不规则地块的凹边形情况 -- done
      目前考虑将田块分割成数个非凹边形从而进行路径规划  Question now：如何通过一个评价指标来保证划分区域的结果是最好的？
    TODO: 6.1 在分割的时候，保证至少有一个凸边形 -- done
    TODO: 6.2 优化田块分解算法，将和运行边垂直的非凸边忽略 -- done: 通过将面积差和凸包不大的多边形看作为凸边形
#### TODO: 7. 计算田块地头 -- done, 目前仅适用于扫描线算法，目前对地块通过 convext_hull 进行了平滑
    TODO: 7.1 地头部分超过了地块本身，需要进行修正 -- done: using gpd.intersection(land)
    TODO: 7.2 对地头区域再次规划路径
        TODO: 7.2.1 地头部分转向路径和在其上再次耕作的区域重合了，如何显示？
#### TODO: 8. 考虑到坡度是斜面，因此直接取 xy 坐标会使得地块的面积变小，需要进行修正 -- 论文
#### TODO: 9. 顺着子块的边进行耕作，找到这些边耕地中，地头区域占据最小的边 -- done
#### TODO: 10. 根据实际情况（耕地的照片），做对比 -- temp_doing: pathline_real_modeling

### 尚未完成 （小功能）
#### TODO: 1. 添加上道路信息、道路宽度等
#### TODO: 2. 转向考虑Dubins path以及另一个path -- 通过 spline 曲线，可以借由优化后的 path 生成更加光滑连续的曲线


在进行路径规划时，您需要考虑以下参数和要求：

1. 地块形状和边界：了解地块的几何形状和边界信息，这将决定路径规划的可行性和约束条件。您可以将地块建模为多边形或其他几何形状。

2. 坡度和高程信息：如果地块存在坡度或高程变化，您需要获取相关的地形数据或传感器测量数据。这些信息将影响路径规划算法的决策，以确保机器能够适应地形的变化。

3. 机器参数：了解农业机械的物理参数，如尺寸、转向半径、最大速度、最大加速度等。这些参数将在路径规划算法中使用，以确保机器在规划的路径上能够正常运行并遵守机械的物理限制。

4. 往复式运行：如果机器只能进行往复式运行，您需要考虑如何在路径规划中实现往复运动。这可能涉及到路径的起点和终点选择、路径的方向控制以及转向半径和掉头方式的设计。

5. 障碍物避让：如果地块上存在障碍物，您需要考虑如何在路径规划中避让这些障碍物。这可能需要使用避障算法和传感器数据，以确保机器能够绕过障碍物并安全运行。

6. 路径评估指标：确定路径规划的评估指标，例如路径长度、能耗、时间等。这将帮助您选择最优的路径规划算法或进行路径规划结果的比较和评估。

基于以上参数和要求，您可以选择适合的路径规划算法，并根据具体情况进行算法的调整和优化。常见的路径规划算法包括A*算法、Dijkstra算法、遗传算法、人工势场法等。您可以根据具体需求选择合适的算法，并根据实际情况进行参数的调整和优化。

除了之前提到的参数外，您还可以考虑以下指标：

1. 耕作路径长度：除了考虑耕作的面积，您可能还关注耕作路径的长度。较短的路径长度可以节省时间和能源消耗。

2. 耕作路径的平滑性：您可以考虑路径的平滑性，避免过多的急转弯或曲线，以提高机器的稳定性和操作效率。

3. 耕作路径的连通性：路径规划算法应确保生成的路径是连通的，即机器能够顺畅地从起点到终点完成耕作，并且没有不可到达的区域。

4. 耕作路径的覆盖率：除了全覆盖地块，您可以考虑路径的覆盖率。这表示路径是否足够密集，以确保地块的每个区域都被有效耕作。

5. 耕作路径的可行性：您需要确保生成的耕作路径在农机的物理限制内，例如转向半径、掉头方式等。路径规划算法应该根据农机的特性和限制来生成可行的路径。

6. 耕作路径的可视化：您可以考虑在路径规划完成后进行路径的可视化，以便更好地理解和评估路径的效果。

这些指标可以帮助您更全面地评估路径规划的性能和效果。根据您的具体需求，您可以根据这些指标选择合适的路径规划算法，并进行参数调整和优化，以实现最佳的路径规划结果。