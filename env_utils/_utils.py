'''
@Author: WANG Maonan
@Date: 2024-04-10 15:42:31
@Description: 一些环境创建使用到的工具
@LastEditTime: 2024-05-07 21:11:55
'''
import math
import numpy as np
from collections import deque # 从collections模块导入，提供高效的双端队列。
from typing import List, Tuple, Dict # 用于类型提示。

class TimeSeriesData:
    def __init__(self, N:int=10) -> None: # 类的构造函数，初始化对象时会调用。N 是一个整数，默认为 10。
        self.N = N # 最多保存 N 个时间步骤的数据
        self.data = deque(maxlen=N) # 初始化一个双端队列 data，最大长度为 N。当超过这个长度时，最旧的数据会被自动移除。

    def add_data_point(self, timestamp:int, data): # 定义一个方法 add_data_point，用于添加数据点。timestamp 是时间戳，data 是要保存的数据。
        """记录每一个时间步的数据

        Args:
            timestamp (int): 仿真时间
            data: 需要保存的数据
        """
        # If deque is full, the oldest data will be removed automatically
        if len(self.data) == self.N:
            self.data.popleft()
        self.data.append((timestamp, data))  # Append new data as a tuple

    def get_recent_k_data(self, K: int=5): # 定义一个方法 get_recent_k_data，用于获取最近的 K 个数据点。
        """Return the last K data points

        Args:
            K (int, optional): 返回的 K 个数据. Defaults to 5.
        """
        recent_data = list(self.data)[-K:]  # Get the last K data points # 将队列转换为列表，并获取最后 K 个数据点。
        return recent_data # 返回最近的 K 个数据。
    
    def get_data_point(self, timestamp:int): # 定义一个方法 get_data_point，用于根据时间戳返回指定的数据点。
        """返回指定时间步的数据 (这个的效率不高, 但是只在绘图的时候才会用到)
        """
        for ts, data in self.data: # 遍历存储的数据，如果找到匹配的时间戳，返回对应的数据。
            if ts == timestamp:
                return data
        return None  # Return None if timestamp not found

    def get_all_data(self): # 定义一个方法 get_all_data，用于返回所有数据。
        """返回所有的数据
        """
        return list(self.data)

    def get_time_stamps(self):
        """获得所有 key, 也就是所有的时间
        """
        return [ts for ts, _ in self.data]

    def calculate_edge_attribute(self, edge_id:str, attribute:str='vehicles'): # 定义一个方法 calculate_edge_attribute，用于统计某个边（edge）的属性。edge_id 是边的标识，attribute 是要计算的属性，默认为 'vehicles'。
        """统计一个 edge 的结果, 一个 edge 包含多个 cell

        Args:
            edge_id (str): 需要计算的 edge id
            attribute (str, optional): 需要计算的属性. Defaults to 'vehicles'.

        Returns:
            _type_: _description_
        """
        attribute_timeseries = [] # 将所有时刻的属性累计起来
        for timestamp, data in self.data: # 遍历所有时间步, 这里需要按照时间顺序, 也就是按照数字大小
            _edge_attribute = [] # 统计某个时刻的属性
            for cell_info in data[edge_id]: # 遍历一个 edge 所有 cell 的数据
                _edge_attribute.append(cell_info.get(attribute, 0)) # 从单元信息中获取指定属性的值，如果不存在则使用 0，添加到边属性列表中。
            attribute_timeseries.append(_edge_attribute) # 将当前时刻的边属性添加到属性时间序列中。
        return attribute_timeseries # 返回属性时间序列。

def merge_local_data(data:Tuple[int, Dict[str, List]]): # 定义一个函数 merge_local_data，用于合并每个时刻的本地数据。参数 data 是一个元组，包含时间和每个路口的数据。
    """将每个时刻的 local data 进行合并

    Args:
        data (Tuple[int, Dict[str, List]]): 每个时刻, 每个路口的数据, 下面是一个例子:
            [
                (1, {'int1': [[1,2,3],[4,5,6]], 'int2': [[7,8,9], [1,2,3]]}),
                (2, {'int1': [[1,2,3],[4,5,6]], 'int2': [[7,8,9], [1,2,3]]}),
                ...
            ]

    Returns: 最后返回每个 tls 的数据, 输出例子如下所示:
    {
        'int1': [], # 多了一个时间维度
        'int2': ...
    }
    """
    # Initialize the result dictionary with IDs as keys and a list to hold arrays for each time step
    result = {id_key: [] for _, id_data in data for id_key in id_data} # 初始化一个字典 result，以 ID 作为键，每个键对应一个空列表，准备存储每个时间步的数据。
    
    # Iterate over the input data
    for time, id_data in data: # 遍历输入数据中的每个时间和对应的路口数据。
        for id_key, _array in id_data.items(): # 每个 id 的数据 # 遍历每个路口的 ID 和对应的数据数组。
            result[id_key].append(_array) # 将每个 ID 对应的数据数组添加到结果字典中。
    
    # stack
    for id_key in result.keys(): # 对结果字典中的每个 ID，使用 numpy 的 stack 函数将列表堆叠成一个数组。[5 12 7]*3
        result[id_key] = np.stack(result[id_key])
    
    return result # 返回合并后的结果字典。# 3*[5 100 25] 以及padding：3*[5 100]

def direction_to_flags(direction): # 定义一个函数 direction_to_flags，用于将方向字符串转换为标志列表。
    """
    Convert a direction string to a list of flags indicating the direction.

    :param direction: A string representing the direction (e.g., 's' for straight).
    :return: A list of flags for straight, left, and right.
    """
    return [
        1 if direction == 's' else 0,
        1 if direction == 'l' else 0,
        1 if direction == 'r' else 0
    ] # 根据传入的方向返回一个列表，标识直行、左转和右转的情况。

def one_hot_encode(input_list, target): # 定义一个函数 one_hot_encode，用于进行独热编码。
    # Initialize a list of zeros of length equal to the input list
    one_hot_vector = [0]*len(input_list) # 初始化一个与输入列表长度相同的全零列表。
    
    # If the target is in the input list, set the corresponding index to 1
    if target in input_list: # 如果目标在输入列表中，将对应索引位置的值设置为 1。
        one_hot_vector[input_list.index(target)] = 1
    
    return one_hot_vector # 返回独热编码后的向量。

def calculate_distance(point1:Tuple[float], point2:Tuple[float]): # 定义一个函数 calculate_distance，用于计算两点之间的距离。
    """计算两点之间的距离

    Args:
        point1 (Tuple[float]): 第一个点的坐标
        point2 (Tuple[float]): 第二个点的坐标
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) # 使用欧几里得距离公式计算并返回两点之间的距离。
