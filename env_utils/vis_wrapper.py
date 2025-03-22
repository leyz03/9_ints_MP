'''
@Author: WANG Maonan
@Date: 2024-04-09 22:33:35
@Description: 根据 global info 来绘制图像, 这里按照 edge 进行绘制
如果要绘制图像, 首先统计每个 lane 的值, 接着根据 lane 的 shape 进行绘制即可
@LastEditTime: 2024-04-10 16:20:12
'''
import numpy as np
import gymnasium as gym
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# from typing import Union # 注意！！这是py3.9版本及以下的


class VisWrapper(gym.Wrapper): # 定义了一个名为 VisWrapper 的类，它继承自 gym.Wrapper，用于对环境进行封装，以添加可视化功能。在初始化时，调用父类的构造函数。
    def __init__(self, env: gym.Env):
        super().__init__(env)
    
    def reset(self, seed: int | None = None): # 重置环境的方法，接受一个可选的种子参数 seed，并调用原环境的 reset 方法返回重置后的状态。
    # def reset(self, seed: Union[int, None] = None): # 注意！！这是py3.9版本及以下的
        return self.env.reset(seed=seed)
    
    def step(self, action): # 执行一个步骤的方法，接收一个动作 action，并调用原环境的 step 方法，返回该步骤的结果。
        return self.env.step(action)

    def __aggregate_statistics(self, timestamp): # 定义一个私有方法 __aggregate_statistics，用于聚合在给定时间戳 timestamp 的统计数据，初始化一个空字典 aggregated_statistics。
        aggregated_statistics = {}
        for edge_id, cells in self.env.edge_cells_timeseries.get_data_point(timestamp).items(): # 遍历在指定时间戳下的每条边缘（edge）及其对应的单元格（cells）。get_data_point 方法返回一个字典，键是边缘 ID，值是对应的单元格数据。
            total_vehicles = sum(cell['vehicles'] for cell in cells)
            total_waiting_time = sum(cell['total_waiting_time'] for cell in cells)
            total_speed = sum(cell['total_speed'] for cell in cells)
            total_co2_emission = sum(cell['total_co2_emission'] for cell in cells)
            
            # 统计包含车辆的 cell 个数
            cells_with_vehicles = sum(1 for cell in cells if cell['vehicles'] > 0)
            
            average_waiting_time = (total_waiting_time / cells_with_vehicles) if cells_with_vehicles else 0 # 计算平均等待时间、平均速度和平均二氧化碳排放量。如果没有车辆，则这些平均值设为 0。
            average_speed = (total_speed / cells_with_vehicles) if cells_with_vehicles else 0
            average_co2_emission = (total_co2_emission / cells_with_vehicles) if cells_with_vehicles else 0
            
            aggregated_statistics[edge_id] = { # 将统计结果以边缘 ID 为键存入 aggregated_statistics 字典。
                'total_vehicles': total_vehicles,
                'total_waiting_time': total_waiting_time,
                'average_waiting_time': average_waiting_time,
                'total_speed': total_speed,
                'average_speed': average_speed,
                'total_co2_emission': total_co2_emission,
                'average_co2_emission': average_co2_emission
            }
        return aggregated_statistics # 返回聚合后的统计数据。
    
    def plot_map(self, timestamp, attributes=['total_vehicles', ], is_plot_edge=False): # 定义一个绘图方法 plot_map，接受时间戳 timestamp 和要绘制的属性列表 attributes（默认为车辆总数），以及一个布尔参数 is_plot_edge 指示是否绘制边框。
        aggregated_statistics = self.__aggregate_statistics(timestamp) # 这里统计每一个 edge 的信息
        # Define a colormap
        cmap = plt.cm.GnBu # 定义一个颜色映射，使用 GnBu（绿色到蓝色）的色彩映射。
        
        for attribute in attributes: # 遍历要绘制的每个属性。
            # Normalize the attribute values for color mapping
            attr_values = [aggregated_statistics[edge_id][attribute] for edge_id in aggregated_statistics] # 提取每个边缘的属性值并归一化，以便后续绘图使用。
            norm = plt.Normalize(vmin=min(attr_values), vmax=max(attr_values), clip=True)
            
            fig, ax = plt.subplots() # 创建一个新的绘图窗口和坐标轴。
            # 绘制 node
            for _, node_info in self.env.node_infos.items(): # 绘制所有节点，使用灰色线条表示节点的形状。
                node_shape = np.array(node_info.get('shape'))
                x, y = node_shape[:, 0], node_shape[:, 1]
                ax.plot(x, y, color='gray', linewidth=1) 

            # 绘制 lane
            for _, lane_info in self.env.lane_infos.items(): # 遍历所有车道信息。
                edge_id = lane_info['edge_id'] # 获得 lane 对应的 edge id
                if edge_id in aggregated_statistics:
                    # Get the attribute value for the current edge
                    attr_value = aggregated_statistics[edge_id][attribute] # 提取该边缘的属性值，并根据归一化后的值获取颜色。
                    # Normalize the attribute value
                    normalized_value = norm(attr_value)
                    # Get the color from the colormap
                    color = cmap(normalized_value)
                    
                    # Get the shape for the current edge
                    shape = np.array(lane_info['shape']) # 获取车道的形状数据。
                    
                    # Plot the edge shape
                    if is_plot_edge: # 根据是否需要绘制边框，使用指定颜色填充车道形状。
                        ax.fill(*zip(*shape), color=color, edgecolor='black')  # Add edgecolor here
                    else: # 不绘制边框
                        ax.fill(*zip(*shape), color=color)
                        
            # Create a ScalarMappable and use it to create the colorbar
            sm = cm.ScalarMappable(norm=norm, cmap=cmap) # 创建一个 ScalarMappable 对象，以便生成颜色条。设置一个空数组以初始化。
            sm.set_array([])  # Set a dummy array for the ScalarMappable.
            fig.colorbar(sm, ax=ax) # 添加颜色条，设置标题和坐标轴标签，并保证坐标轴比例相等。
            ax.set_title(f'Map Visualization for {attribute}')
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            ax.axis('equal')

            # Show the figure
            plt.show()
    
    def plot_edge_attribute(self, edge_id:str, attribute:str='vehicles'):
        """
        Plot the attribute over time for each cell in an edge.

        :param data: A 2D list where each sublist represents the congestion at each cell for a given time.
        """
        # Convert the data to a numpy array for better handling
        data = self.env.edge_cells_timeseries.calculate_edge_attribute(edge_id, attribute) # 计算该边缘的属性数据，并将其转换为 NumPy 数组。
        data_array = np.array(data)

        # Create a meshgrid for plotting # 为绘图创建网格，X 轴为时间，Y 轴为单元格索引。
        time = np.arange(data_array.shape[0] + 1)
        cell_index = np.arange(data_array.shape[1] + 1)
        T, C = np.meshgrid(time, cell_index)

        # Create the plot # 创建一个新的图形，并使用 pcolormesh 绘制二维颜色图。
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(T, C, data_array.T, shading='auto')  # Transpose data_array to align with the meshgrid

        # Set the labels and title
        plt.xlabel('Time')
        plt.ylabel('Cell Index')
        plt.title('Congestion Level Over Time')

        # Show the plot with a color bar
        plt.colorbar(label='Congestion Level')
        plt.show()