'''
@Author: WANG Maonan
@Date: 2024-04-09 22:32:16
@Description: 检查 Global Local Env 和 Vis Env
=> Global Local Env: 环境的特征是否提取正确
=> Vis Env: 是否可以正确进行可视化
@LastEditTime: 2024-04-14 18:13:58
'''
import json
import numpy as np
from typing import List

from env_utils.tsc_env import TSCEnvironment
from env_utils.global_local_wrapper import GlobalLocalInfoWrapper
from env_utils.vis_wrapper import VisWrapper

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))

def load_environment_config(env_config_path): # 该函数用于读取指定路径下的 JSON 配置文件并将其解析为 Python 字典对象。
    env_config_path = path_convert(f'./configs/env_configs/{env_config_path}') # 是配置文件的路径，调用 path_convert 将其转换为绝对路径。
    with open(env_config_path, 'r') as file:
        config = json.load(file) # 读取配置文件并使用 json.load() 将其内容加载为 Python 字典。
    return config

def make_multi_envs(
        agent_tls_ids:List[str], 
        non_agent_tls_ids:List[str], 
        sumo_cfg:str, net_file:str,
        num_seconds:int, use_gui:bool,
        road_ids: List[str],
        log_file:str, cell_length:int=20 # 每个单元格的长度，默认值为 20，用于定义路网的划分粒度
    ):
    tsc_env = TSCEnvironment(
        sumo_cfg=sumo_cfg,
        net_file=net_file, # 用于加载路网的信息
        num_seconds=num_seconds,
        agent_tls_ids=agent_tls_ids,
        non_agent_tls_ids=non_agent_tls_ids,
        tls_action_type='choose_next_phase',
        use_gui=use_gui
    )
    tsc_env = GlobalLocalInfoWrapper(tsc_env, filepath=log_file, road_ids=road_ids, cell_length=20)
    tsc_env = VisWrapper(tsc_env) # 加入绘制全局特征的功能

    return tsc_env

if __name__ == '__main__':
    sumo_cfg = path_convert("./sumo_nets/demo/env/demo.sumocfg")
    net_file = path_convert("./sumo_nets/demo/env/demo.net.xml")
    log_path = path_convert('./log/osm_berlin')
    
    env_config = load_environment_config("demo.json")
    road_ids = env_config['road_ids']

    env = make_multi_envs(
        agent_tls_ids=["E"], 
        non_agent_tls_ids=["B", "D", "F", "H"],
        sumo_cfg=sumo_cfg,
        net_file=net_file,
        num_seconds=1000,
        road_ids=road_ids,
        log_file=log_path,
        use_gui=True,
        # use_gui=False,
        cell_length=50
    )

    done = False
    state = env.reset()
    while not done:
        action = {  
            "E": np.random.randint(4)
        }
        observations, rewards, terminations, truncations, infos = env.step(action)
        # env.plot_map(timestamp=120, attributes=['total_vehicles', 'average_waiting_time', 'average_speed']) # 需要加入 vis_wrapper 之后才可以有的功能
        # env.plot_edge_attribute(edge_id='-1105574288#1', attribute='vehicles')
        # env.plot_edge_attribute(edge_id='-1105574288#1', attribute='average_waiting_time')
        # env.plot_edge_attribute(edge_id='-23755720#5', attribute='average_waiting_time')

        total_reward = sum(rewards.values())
        # if total_reward <= -100:
            # env.plot_map(timestamp=120, attributes=['total_vehicles', 'average_waiting_time', 'average_speed']) # 需要加入 vis_wrapper 之后才可以有的功能
            # env.plot_edge_attribute(edge_id='-E1', attribute='vehicles')
            # env.plot_edge_attribute(edge_id='-E2', attribute='average_waiting_time')
            # env.plot_edge_attribute(edge_id='E2', attribute='average_waiting_time')