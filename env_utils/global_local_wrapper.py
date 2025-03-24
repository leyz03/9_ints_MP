'''
@Author: WANG Maonan
@Date: 2024-04-10 00:21:49
@Description: 根据 state 提取 global info 和 local info, 具体来说分为三类特征:
1. 微观特征: 车辆的属性
2. 中观特征: 路段摄像头的数据
3. 宏观特征: 6G as a sensor
LastEditTime: 2024-09-17 16:45:13
'''
import time
import numpy as np
import gymnasium as gym
from typing import Dict, Any, List
from stable_baselines3.common.monitor import ResultsWriter

from ._utils import (
    TimeSeriesData, 
    direction_to_flags, 
    merge_local_data, 
    one_hot_encode,
    calculate_distance
)

from rule_based_policy.traffic_rule_policy import TrafficPolicy

class GlobalLocalInfoWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, filepath:str, road_ids:List[str], cell_length:float=20):
        super().__init__(env)
        self.agent_tls_ids = self.env.unwrapped.agent_tls_ids # 记录 agent 的路口 id
        self.non_agent_tls_ids = self.env.unwrapped.non_agent_tls_ids # 记录非 agent 的路口 id
        self.tls_ids = sorted(self.env.unwrapped.agent_tls_ids + self.env.unwrapped.non_agent_tls_ids) # 多路口的 ids
        self.cell_length = cell_length # 每个 cell 的长度 # 每个单元格的长度，默认为 20
        self.filepath = filepath # 日志文件路径
        self.max_vehicle_num = 100 # 记录每个路口的 max_vehicle_num 数量的车

        # 记录时序数据 # 初始化多个时序数据结构，用于保存车辆信息、掩码、局部观察数据和边缘单元格的结果。
        self.vehicle_timeseries = TimeSeriesData(N=10) # 记录车辆的数据
        self.vehicle_masks_timeseries = TimeSeriesData(N=10)
        self.local_obs_timeseries = TimeSeriesData(N=10) # 将局部信息全部保存起来
        self.edge_cells_timeseries = TimeSeriesData(N=10) # edge cell 的结果

        # reset init
        self.node_infos = None # 节点的信息
        self.lane_infos = None # 地图车道的原始信息
        self.road_ids = road_ids # 地图所有的 edge id, (1). 用于记录 global feature 的顺序; (2). 用于做 one-hot 向量, 表明车辆的位置
        self.tls_nodes = None # 每个信号灯的 id 和 坐标
        self.vehicle_feature_dim = 0 # 车辆特征的维度
        self.tls_movement_id = {} # 每个路口的 movement 顺序
        self.max_num_cells = -1 # 最大的 cell 个数, 确保 global info 大小一样
        self.edge_cell_mask = [] # 记录 global cell 的时候, 哪些是 padding 的

        # init non_agents policy
        self.rule_policies = {}
        self.junction_phase_group = {} # phase 包含哪些 movement
        self.junction_movement_ids = {}
        self.rule_actions = {}

        # #######
        # Writer # 结果记录器
        # #######
        if self.filepath is not None: # 如果提供了日志文件路径，则创建一个结果记录器 ResultsWriter，用于记录仿真结果。同时记录开始时间 t_start。
            self.t_start = time.time()
            self.results_writer = ResultsWriter(
                filepath,
                header={"t_start": self.t_start},
            )
            self.rewards_writer = list()

    def __initialize_edge_cells(self):
        """根据道路信息初始化 edge cell 的信息, 这里每个时刻都需要进行初始化
        """
        edge_cells = {} # 创建一个空字典 edge_cells，然后遍历每个道路 ID，确保 max_num_cells 已经设置，并将其转换为整数，统一每个边缘的单元格大小。
        for edge_id in self.road_ids:
            assert  self.max_num_cells != -1, '检查代码关于 max_num_cells 的部分, 现在是 -1.'
            num_cells = int(self.max_num_cells) # 统一大小, 这样不同 lane 的长度是一样的
            if edge_id not in edge_cells: # 如果该边缘 ID 尚未在字典中，则初始化对应的单元格信息，记录每个单元格的车辆数量、总等待时间、总速度和总二氧化碳排放量。
                edge_cells[edge_id] = [
                    {
                        'vehicles': 0, 
                        'total_waiting_time': 0.0, 
                        'total_speed': 0.0, 
                        'total_co2_emission': 0.0,
                    } for _ in range(num_cells)
                ] # 初始化每一个 cell 的信息
        return edge_cells # 返回初始化的边缘单元格信息。

    # ####################
    # 下面开始处理每一步的特征
    # ####################
    def get_vehicle_obs(self, vehicle_data:Dict[str, Dict[str, Any]]): # 定义 get_vehicle_obs 方法，该方法用于提取车辆的观察数据，参数是车辆数据的字典。
        """用于获得每一辆车的信息, 主要包含:
        1. 车辆的速度
        2. 车辆所在的 road, 使用 one-hot
        3. 车辆所在的 lane position
        4. 车辆的 waiting time
        5. 车辆的 accumulated_waiting_time
        6. 车辆记录路口的距离

        这里我们找出与每个路口最接近的 self.max_vehicle_num 辆车

        Args:
            vehicle_data (Dict[str, Dict[str, Any]]): 仿真中车辆的信息, 下面是一个例子:
            {
                '-E6__0__background_1.0': {'id': '-E6__0__background_1.0', 'action_type': 'lane', 'vehicle_type': 'background_1', 'length': 7.0, 'width': 1.8, 'heading': 355.69461230491356, 'position': (293.83316888099887, -59.30507987749093), 'speed': 4.603267983696424, 'road_id': '-E6', 'lane_id': '-E6_1', 'lane_index': 1, 'lane_position': 13.943846113653855, 'edges': ('-E6', 'E2', 'E7'), 'waiting_time': 0.0, ...},
                '-E7__0__background_2.0': {'id': '-E7__0__background_2.0', 'action_type': 'lane', 'vehicle_type': 'background_2', 'length': 7.0, 'width': 1.8, 'heading': 269.75716619082107, 'position': (658.4361538863461, 49.21090214725246), 'speed': 4.603267983696424, 'road_id': '-E7', 'lane_id': '-E7_0', 'lane_index': 0, 'lane_position': 13.943846113653855, 'edges': ('-E7', '-E2', '-E1', '-E0'), 'waiting_time': 0.0, ...}，
                ...
            }
        """
        closest_vehicles = {intersection: [] for intersection in self.tls_ids} # 初始化两个字典，分别用于存储每个路口最近的车辆信息和填充掩码。
        padding_masks = {intersection: [] for intersection in self.tls_ids}
        
        # Step 2: Calculate distance of each vehicle from each intersection
        for veh_id, veh_data in vehicle_data.items(): # 遍历车辆数据，对于每辆车，获取其车道 ID，并确保它不是以冒号开头的特殊车道 ID。
            _lane_id = veh_data['lane_id']
            if not _lane_id.startswith(":"):
                for intersection_id, intersection_pos in self.tls_nodes.items(): # 对于每个交通信号灯，计算车辆与该路口的距离。
                    _distance = calculate_distance(veh_data['position'], intersection_pos)
                    _speed = veh_data['speed'] # 获取车辆的速度、车道位置（相对于车道长度的比例）、等待时间和累积等待时间，并对车辆所在的道路进行独热编码。
                    _lane_position = veh_data['lane_position']/self.lane_infos[_lane_id]['length']
                    _waiting_time = veh_data['waiting_time']
                    _accumulated_waiting_time = veh_data['accumulated_waiting_time']
                    _edge_id = one_hot_encode(self.road_ids, veh_data['road_id'])
                    closest_vehicles[intersection_id].append([_distance, _speed, _lane_position, _waiting_time, _accumulated_waiting_time] + _edge_id) # 将车辆信息添加到最近车辆列表中，并在掩码中记录该车辆是有效数据（用 1 表示）。25
                    padding_masks[intersection_id].append(1)  # 1 indicates actual vehicle data

        # Step 3: Sort the vehicles by distance for each intersection and take the closest N
        for intersection_id in self.tls_nodes: # 对每个交通信号灯的车辆列表进行距离排序，并保留最近的 max_vehicle_num 辆车。
            closest_vehicles[intersection_id].sort(key=lambda x: x[0]) # 按照距离进行排序
            closest_vehicles[intersection_id] = closest_vehicles[intersection_id][:self.max_vehicle_num] # 只取前 max_vehicle_num 个
            padding_masks[intersection_id] = padding_masks[intersection_id][:self.max_vehicle_num] # 同步更新掩码，确保与车辆数量一致。
            
            # Padding if there are less than N vehicles
            num_padding = self.max_vehicle_num - len(closest_vehicles[intersection_id]) # 如果最近车辆的数量少于 max_vehicle_num，则通过填充零向量和零掩码进行补全，确保每个路口的车辆信息长度一致。
            closest_vehicles[intersection_id].extend([[0] * self.vehicle_feature_dim] * num_padding) # self.vehicle_feature_dim 在 reset 的时候计算
            padding_masks[intersection_id].extend([0] * num_padding)  # 0 indicates padding

        return closest_vehicles, padding_masks # 最后返回每个路口的最近车辆信息和填充掩码。

    def get_edge_cells(self, vehicle_data:Dict[str, Dict[str, Any]]):
        """计算每一个 edge cell 每一个时刻的信息, 可以用于计算 global info, 或是用于可视化

        Args:
            vehicle_data (Dict[str, Dict[str, Any]]): 仿真中车辆的信息, 具体例子见上面

        Returns:
            _type_: _description_
        """
        edge_cells = self.__initialize_edge_cells() # 初始化 cell 信息

        # 首先统计当前时刻 vehicle 在哪一个 cell, 然后改变 cell 的统计量
        for vehicle_id, vehicle_info in vehicle_data.items(): # 迭代每辆车的信息，vehicle_id 是车辆的唯一标识，vehicle_info 是该车的详细信息。
            edge_id = vehicle_info['road_id'] # 获取道路ID: 从车辆信息中获取其所在道路的ID，并过滤掉交叉口（以":"开头的ID）。
            if not edge_id.startswith(':'): # 不考虑交叉路口里面
                lane_position = vehicle_info['lane_position'] # 计算车道位置: 根据车辆在车道上的位置，计算出它所处的单元格索引（cell index）。
                cell_index = int(lane_position // self.cell_length) # 计算属于哪一个 cell
                
                cell = edge_cells[edge_id][cell_index]
                cell['vehicles'] += 1
                cell['total_waiting_time'] += vehicle_info['waiting_time']
                cell['total_speed'] += vehicle_info['speed']
                cell['total_co2_emission'] += vehicle_info['co2_emission'] # 更新单元格信息: 找到对应的单元格后，更新该单元格中的车辆数量、总等待时间、总速度和总二氧化碳排放量。

        # 最后输出的时候计算平均值即可
        for edge_id, cells in edge_cells.items(): # 计算平均值: 遍历每个单元格，计算平均等待时间、平均速度和平均二氧化碳排放。如果没有车辆则设为0。
            for cell in cells:
                if cell['vehicles'] > 0:
                    cell['average_waiting_time'] = cell['total_waiting_time'] / cell['vehicles']
                    cell['average_speed'] = cell['total_speed'] / cell['vehicles']
                    cell['average_co2_emission'] = cell['total_co2_emission'] / cell['vehicles']
                else:
                    cell['average_waiting_time'] = 0.0
                    cell['average_speed'] = 0.0
                    cell['average_co2_emission'] = 0.0

        return edge_cells

    def get_local_tls_state(self, tls_states):
        """获得每个路口每一个时刻的信息, 获得摄像头的数据
        """
        tls_local_obs = {} # 每一个 tls 处理好的特征

        for _tls_id in self.tls_ids: # 依次处理每一个路口
            process_local_obs = []
            for _movement_index, _movement_id in enumerate(self.tls_movement_id[_tls_id]): # 处理每个交通流向: 迭代当前交通信号灯的每个交通流向，记录下其状态信息。
                occupancy = tls_states[_tls_id]['last_step_occupancy'][_movement_index]/100
                pressure = tls_states[_tls_id]['pressure_per_lane'][_movement_index]
                mean_speed = tls_states[_tls_id]['last_step_mean_speed'][_movement_index] # 获得平均速度
                direction_flags = direction_to_flags(tls_states[_tls_id]['movement_directions'][_movement_id])
                lane_numbers = tls_states[_tls_id]['movement_lane_numbers'][_movement_id]/5 # 车道数 (默认不会超过 5 个车道)
                is_now_phase = int(tls_states[_tls_id]['this_phase'][_movement_index]) # 处理其他信息: 获取流向标志、车道数（按5进行归一化）以及当前信号相位。
                # 将其添加到 obs 中
                process_local_obs.append([occupancy, pressure, mean_speed, *direction_flags, lane_numbers, is_now_phase]) # 某个 movement 对应的信息 # 记录处理后的信息: 将所有信息添加到当前流向的列表中。 7
                # process_local_obs.append([occupancy, mean_speed, *direction_flags, lane_numbers, is_now_phase])

            # 不是四岔路, 进行不全
            for _ in range(12 - len(process_local_obs)): # 记录处理后的信息: 将所有信息添加到当前流向的列表中。
                process_local_obs.append([0]*len(process_local_obs[0]))
            
            tls_local_obs[_tls_id] = process_local_obs # 存储每个路口处理好的信息

        return tls_local_obs


    # #################
    # 下面开始处理时序特征
    # #################
    def process_global_state(self, K=5):
        """根据 edge cell 的信息来计算 global info, 每个 cell 都是一个向量, 同时包含 cell 的坐标
        """
        _recent_k_data = self.edge_cells_timeseries.get_recent_k_data(K) # 参数: K 指定获取最近的K个时间片的数据
        result = {id_key: [] for _, id_data in _recent_k_data for id_key in id_data} # 获取最近数据: 从边缘单元时间序列中获取最近K个时间片的数据，并初始化结果字典。
        
        # Iterate over the input data
        for time, id_data in _recent_k_data: # 遍历时间序列数据: 对每个时间点的数据进行处理，提取边缘信息并存储。
            for edge_id, cell_data in id_data.items(): # 每个 edge 的数据
                edge_info = []
                for _cell_info in cell_data: # 某个时刻, 某个 edge 对应的 cell 的数据
                    _cell_vehicle = _cell_info['vehicles']/2
                    _cell_avg_waiting_time = _cell_info['average_waiting_time']
                    _cell_speed = _cell_info['average_speed']
                    edge_info.append([_cell_vehicle, _cell_avg_waiting_time, _cell_speed])
                result[edge_id].append(edge_info)
        
        # stack
        final_result = []
        for id_key in self.road_ids: # 整理最终结果: 将每个道路的结果收集到最终列表中。
            final_result.append(result[id_key]) # 20*[5 5 3]
        
        return np.stack(final_result) # 返回结果: 将结果转换为numpy数组并返回。

    def process_local_state(self, K=5):
        """计算局部的信息, 需要可以处理 reset 的情况, 也就是可以处理时间序列不全的时候
        """
        _recent_k_data = self.local_obs_timeseries.get_recent_k_data(K) # 获取最近数据: 从局部观察时间序列获取最近K个数据，并合并这些数据后返回。
        return merge_local_data(_recent_k_data)
    
    def process_veh_state(self, K=5):
        """获得最后 K 个时刻车辆的信息

        Args:
            K (int, optional): 去 K 个时间片, 这里车辆的信息只使用最后一个时刻的信息. Defaults to 5.
        """
        _recent_k_data_veh = self.vehicle_timeseries.get_recent_k_data(K)
        _recent_k_data_veh_padding = self.vehicle_masks_timeseries.get_recent_k_data(K)
        return merge_local_data(_recent_k_data_veh), merge_local_data(_recent_k_data_veh_padding) # 获取数据: 从车辆时间序列和车辆掩码时间序列获取最近K个数据，并合并后返回

    def process_reward(self, tls_data, vehicle_state):
        """
        Calculate the average waiting time for vehicles at all intersections.
        这里是按整个路网计算一个统一的奖励, 而不是每一个路口计算名一个奖励

        :param vehicle_state: The state of vehicles in the environment.
        :return: The negative average waiting time as the reward.
        """
        pressure = np.array([tls_data[tls_id]['pressure_per_lane'] for tls_id in self.tls_ids]) # 获取每个路口的压力信息
        
        occupancy = np.array([tls_data[tls_id]['last_step_occupancy'] for tls_id in self.tls_ids]) # 获取每个路口的占用率信息

        waiting_times = [veh['waiting_time'] for veh in vehicle_state.values()]
        
        return -occupancy.mean() * max(abs(pressure)) if len(waiting_times) > 0 else 0.0 # 返回平均等待时间作为奖励，如果没有车辆则返回0.0
    # #############
    # reset & step
    # #############
    def reset(self, seed=1): # 定义了一个名为 reset 的方法，参数 seed 用于设置随机种子，默认为 1。这个方法的作用是重置环境的状态。
        """reset env
        """
        state = self.env.reset() # 调用环境对象的 reset 方法，获取当前环境的初始状态，存储在变量 state 中。
        self.node_infos = state['node'] # 地图节点的信息
        self.lane_infos = state['lane'] # 地图车道信息
        if (self.road_ids is None) or (len(self.road_ids) == 0): # road id 可以自己输入, 或是从路网中解析 # 检查 road_ids 是否为空，如果为空，则从车道信息中提取所有的边缘 ID，并将其去重后排序，存储到 self.road_ids 中。边缘 ID 代表地图上的道路。
            self.road_ids = sorted(set([_lane['edge_id'] for _,_lane in self.lane_infos.items()])) # 获得所有的 edge id, 用于对车辆所在位置进行 one-hot
        self.tls_nodes = {_node_id:self.node_infos[_node_id]['node_coord'] for _node_id in self.tls_ids} # 找到所有信号灯对应的坐标
        self.vehicle_feature_dim = 5 + len(self.road_ids) # 车辆的特征的维度 # 计算车辆特征的维度，设定为 5 加上道路数量。这表示车辆的特征包括一些固定的属性和道路信息。

        # 更新全局最大的 max_num_cells 的个数, 同时记录每一个 edge 的 cell 个数
        _edge_cell_num = {} # 初始化一个字典 _edge_cell_num 用于存储每条道路的单元格数量，初始化一个列表 edge_cell_mask 用于存储每条道路的掩码。
        self.edge_cell_mask = []
        for _, lane_info in self.lane_infos.items(): # 遍历每条车道的信息，计算每条车道的单元格数量，并更新 _edge_cell_num 字典。如果当前车道的单元格数量超过了 max_num_cells，则更新 max_num_cells。
            _edge_id = lane_info['edge_id']
            _num_cell = lane_info['length'] // self.cell_length + 1
            _edge_cell_num[_edge_id] = _num_cell # 更新 edge_id 对应的 cell 数量
            if _num_cell > self.max_num_cells:
                self.max_num_cells = _num_cell

        # 更新 global mask
        for _road_id in self.road_ids: # 为每条道路创建掩码，掩码的前部分用 1 表示实际存在的单元格，后部分用 0 表示填充的部分。然后将每条道路的掩码添加到 edge_cell_mask 列表中
            _num_cell = _edge_cell_num[_road_id]
            _road_mask = [1]*int(_num_cell) + [0]*int(self.max_num_cells-_num_cell)
            self.edge_cell_mask.append(_road_mask)

        # Update Local Info
        tls_data = state['tls'] # 获得路口的信息
        for _tls_id, _tls_data in tls_data.items(): # 遍历每个信号灯的 ID，提取其运动 ID，并将其存储到 tls_movement_id 字典中。
            self.tls_movement_id[_tls_id] = _tls_data['movement_ids'] # 获得每个 tls 的 movement id
        local_obs = self.get_local_tls_state(tls_data) # 获取当前交通信号灯的局部状态，并将其记录到 local_obs_timeseries 中。
        self.local_obs_timeseries.add_data_point(timestamp=0, data=local_obs) # 记录局部信息

        ### 初始化non agents的策略
        for id in self.env.unwrapped.non_agent_tls_ids:
            self.rule_policies[id] = TrafficPolicy(state['tls'])
        self.junction_movement_ids = self.tls_movement_id.copy()
        for tls_id, tls_info in self.env.tsc_env.scene_objects['tls'].traffic_lights.items():
            self.junction_phase_group[tls_id] = tls_info.phase2movements.copy()          

        self.rule_actions = {key: np.int64(0) for key in self.env.unwrapped.non_agent_tls_ids}

        # Update Global Info # 从状态中提取车辆信息，并通过 get_edge_cells 方法获取每个单元格的状态，存储在 global_obs 中。
        vehicle_data = state['vehicle'] # 获得车辆的信息
        global_obs = self.get_edge_cells(vehicle_data) # 得到每一个 cell 的信息
        self.edge_cells_timeseries.add_data_point(timestamp=0, data=global_obs) # 记录全局信息 # 将全局信息记录到 edge_cells_timeseries 中。
        
        # Vehicle Local Info
        vehicle_obs, padding_masks = self.get_vehicle_obs(vehicle_data) # 调用 get_vehicle_obs 方法获取车辆的观测数据和填充掩码，并将这些数据分别记录到 vehicle_timeseries 和 vehicle_masks_timeseries 中。
        self.vehicle_timeseries.add_data_point(timestamp=0, data=vehicle_obs)
        self.vehicle_masks_timeseries.add_data_point(timestamp=0, data=padding_masks)

        # TODO, 这里需要根据不同的路网手动调整 (这里可以使用参数传入, 参数在配置文件里面, 每一个环境的参数应该是固定的) # 处理局部观测数据，创建一个字典 processed_local_obs，其中每个信号灯的状态是一个随机生成的数组，形状为 (5, 12, 7)，表示时间序列、运动数量和每个运动的特征。
        processed_local_obs = {_tls_id:np.random.randn(5,12,8) for _tls_id in self.tls_ids} # 5 是时间序列, 12 movement 数量, 7 是每个 movement 的特征
        processed_global_obs = np.random.randn(len(global_obs),5,int(self.max_num_cells),3) # len(global_obs): edge 的数量, 5 是时间序列, self.max_num_cells 是 cell 数量, 3 是每个 edge 的特征 # 处理全局观测数据，生成一个随机数组 processed_global_obs，形状为 (edge 数量, 5, 最大单元格数量, 3)，表示每条边的特征。
        processed_veh_obs = {_tls_id:np.random.randn(5,100,299) for _tls_id in self.tls_ids} # 处理车辆的观测数据和掩码，processed_veh_obs 为随机生成的数组，processed_veh_mask 为全零数组，分别对应每个信号灯的状态。
        processed_veh_mask = {_tls_id:np.zeros((5,100)) for _tls_id in self.tls_ids}

        return (processed_local_obs, processed_global_obs, np.array(self.edge_cell_mask), processed_veh_obs, processed_veh_mask) # 最后，将处理后的局部观测、全局观测、边缘掩码、车辆观测和车辆掩码以元组形式返回。
    

    def step(self, action: Dict[str, int]):
        """这里的数据处理流程为:
        => 获取每个时刻的信息
        1. self.get_local_tls_state, 获取每一个时刻信号灯的信息
        2. self.get_edge_cells, 获取每一个时刻每一个 cell 的信息
        3. self.get_vehicle_obs, 获得每一个时刻每一辆车的信息
        4. 这里会使用 self.xxx_timeseries.add_data_point, 将每个时刻的数据保存起来
        => 将每一个时刻拼接为时间序列
        1. self.process_local_state, 拼接局部信息
        2. self.process_global_state, 拼接全局信息
        3. self.process_veh_state, 拼接车辆的信息
        """
        can_perform_actions = {_tls_id:False for _tls_id in self.tls_ids}

        # 合并两个action字典
        # TODO 全为non agents时这两行应该要注销
        self.rule_actions.update(action)
        action = self.rule_actions

        # MP控制所需参数
        junction_phase_group = self.junction_phase_group
        junction_movement_ids = self.junction_movement_ids

        # 与环境交互
        while not any(can_perform_actions.values()): # 有一个信号灯要做动作, 就需要进行下发 # 只要有一个True循环就终止                        
            states, rewards, truncated, done, infos = super().step(action)
            # 1. 获得 states 中车辆和信号灯的信息
            vehicle_data = states['vehicle'] # 获得车辆的信息
            tls_data = states['tls'] # 获得路口的信息
            sim_step = int(self.env.tsc_env.sim_step) # 仿真时间

            # 2. 更新 can_perform_action
            can_perform_actions = {
                _tls_id: tls_data[_tls_id]['can_perform_action']
                for _tls_id in self.tls_ids
            } # 只要有一个 traffic signal 可以做动作, 就需要进行下发
            
            # 3. 根据原始信息提取 (1) veh obs. (2) local obs. (3) global obs
            veh_obs, padding_masks = self.get_vehicle_obs(vehicle_data) # Get Vehicle Info
            self.vehicle_timeseries.add_data_point(sim_step, veh_obs)
            self.vehicle_masks_timeseries.add_data_point(sim_step, data=padding_masks)
            local_obs = self.get_local_tls_state(tls_data) # Get Local Info
            self.local_obs_timeseries.add_data_point(sim_step, local_obs) 
            global_obs = self.get_edge_cells(vehicle_data) # Get Edge Cell
            self.edge_cells_timeseries.add_data_point(sim_step, global_obs)

        # 开始处理 state, 1. local; 2. local+global; 3. local+global+vehicle
        processed_local_obs = self.process_local_state() # 3*[5 12 7]
        processed_global_obs = self.process_global_state() # [20 5 5 3]
        processed_veh_obs, processed_veh_mask = self.process_veh_state()
        
        # 处理 reward
        #reward = self.process_reward(vehicle_data) # 计算路网的整体等待时间
        reward = self.process_reward(tls_data, vehicle_data) 
        rewards = {_tls_id:reward for _tls_id in self.tls_ids}

        # 处理 dones & truncateds
        dones = {_tls_id:done for _tls_id in self.tls_ids}
        truncateds = {_tls_id:truncated for _tls_id in self.tls_ids}

        # 处理 info
        infos = {_tls_id:{} for _tls_id in self.tls_ids}
        for _tls_id, _can_perform in can_perform_actions.items():
            infos[_tls_id]['can_perform_action'] = _can_perform

        # 处理non agents的actions
        for id in self.env.unwrapped.non_agent_tls_ids:
            # action_value = self.rule_policies[id].ft_policy(traffic_state=processed_local_obs, action_step=4, infos=infos)[id] 
            action_value = self.rule_policies[id].mp_policy(traffic_state=processed_local_obs,  junction_phase_group=junction_phase_group, junction_movement_ids=junction_movement_ids, infos=infos)[id]
            self.rule_actions[id] = np.int64(action_value)

        # Writer
        if self.filepath is not None:
            self.rewards_writer.append(float(sum(rewards.values())))
            if all(dones.values()): # 如果所有信号灯的状态都是完成的，执行以下操作：
                ep_rew = sum(self.rewards_writer)
                ep_len = len(self.rewards_writer)
                ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
                self.results_writer.write_row(ep_info)
                self.rewards_writer = list()
            
        return (processed_local_obs, processed_global_obs, np.array(self.edge_cell_mask), processed_veh_obs, processed_veh_mask), rewards, truncateds, dones, infos
    
