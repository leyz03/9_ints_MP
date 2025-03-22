'''
@Author: WANG Maonan
@Date: 2024-04-14 17:56:50
@Description: 检查 petting zoo 的环境
@LastEditTime: 2024-04-17 05:17:33
'''
import json
import numpy as np
from typing import List, Dict
from pettingzoo.test import parallel_api_test

from env_utils.tsc_env import TSCEnvironment
from env_utils.global_local_wrapper import GlobalLocalInfoWrapper
from env_utils.pz_env import TSCEnvironmentPZ

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))

def load_environment_config(env_config_path): # 该函数用于读取指定路径下的 JSON 配置文件并将其解析为 Python 字典对象。
    env_config_path = path_convert(f'./configs/env_configs/{env_config_path}') # 是配置文件的路径，调用 path_convert 将其转换为绝对路径。
    with open(env_config_path, 'r') as file:
        config = json.load(file) # 读取配置文件并使用 json.load() 将其内容加载为 Python 字典。
    return config

def make_pz_envs(
        agent_tls_ids:List[str], 
        non_agent_tls_ids:List[str],
        sumo_cfg:str, net_file:str,
        log_file:str, 
        num_seconds:int, use_gui:bool,
        road_ids: List[str],
        action_space:Dict[str, int],
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
    tsc_env = TSCEnvironmentPZ(tsc_env, action_space)

    return tsc_env

if __name__ == '__main__':
    sumo_cfg = path_convert("./sumo_nets/demo/env/demo.sumocfg")
    net_file = path_convert("./sumo_nets/demo/env/demo.net.xml")
    log_path = path_convert('./log/demo')
    
    env_config = load_environment_config("demo.json")
    road_ids = env_config['road_ids']
    action_space = env_config["action_space"]

    env = make_pz_envs(
        agent_tls_ids=['E'], # 控制 3 个路口, 都是 2 相位
        non_agent_tls_ids=['B', 'D', 'F', 'H'],
        sumo_cfg=sumo_cfg,
        net_file=net_file,
        log_file=log_path,
        num_seconds=2000,
        use_gui=False,
        road_ids=road_ids,
        action_space=action_space,
    )

    # parallel_api_test(env, num_cycles=1_000_000) # 这种 agent 变化会导致无法通过 test, 但是可以在 torchrl 里面使用
    for _ in range(3): # 完整运行三次仿真, 查看是否有出错
        state, info = env.reset()
        dones = False
        while not dones:
            random_action = {_tls_id:np.random.randint(4) for _tls_id in ["E"]}
            observations, rewards, terminations, truncations, infos = env.step(random_action)
            done = all(terminations.values()) # 如果至少有一个路口没有满足终止条件，done 会被设置为 False，表示仿真继续进行。
    env.close() # 不知道为何，测试的时候不终止
