'''
@Author: WANG Maonan
@Date: 2024-04-09 22:32:16
@Description: 检查 Global Local Env, 环境的特征是否提取正确
LastEditTime: 2024-09-17 15:48:12
'''
import json
import numpy as np
from typing import List

from env_utils.tsc_env import TSCEnvironment
from env_utils.global_local_wrapper import GlobalLocalInfoWrapper

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'), file_log_level="INFO", terminal_log_level="INFO")

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
    tsc_env = GlobalLocalInfoWrapper(tsc_env, filepath=log_file, road_ids=road_ids, cell_length=cell_length)

    return tsc_env

if __name__ == '__main__':
    # 读取实验配置文件
    env_config = load_environment_config("demo.json")

    sumo_cfg = path_convert(env_config['sumocfg'])
    net_file = path_convert(env_config['sumonet'])
    num_seconds = env_config['simulation_time']
    road_ids = env_config['road_ids']
    log_path = path_convert('./log/')
    env = make_multi_envs(
        agent_tls_ids=['E'],
        non_agent_tls_ids=['B', 'D', 'F', 'H'], # 控制 3 个路口, 都是 2 相位
        sumo_cfg=sumo_cfg,
        net_file=net_file,
        num_seconds=num_seconds,
        road_ids=road_ids,
        log_file=log_path,
        use_gui=True,
        # use_gui=False,
        cell_length=50
    )

    done = False
    eposide_reward = 0 # 累计奖励
    state = env.reset()
    while not done:
        action = {
            # 在每个仿真步骤中，生成随机的交通信号控制动作（对于每个路口，随机选择一个 0 到 3 之间的整数，表示四个相位中的一个）。
            "E": np.random.randint(4),
        }
        states, rewards, truncateds, dones, infos = env.step(action)
        (processed_local_obs, processed_global_obs, edge_cell_mask, processed_veh_obs, processed_veh_mask) = states # processed_local_obs, processed_global_obs 等是从 states 中解包出的各个环境信息。
        eposide_reward += sum(rewards.values()) # 计算累计奖励
        done = all(dones.values()) or (len(dones) == 0)
    env.close()

    print(f"累计奖励为, {eposide_reward}.")