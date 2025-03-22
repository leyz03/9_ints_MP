'''
Author: Maonan Wang
Date: 2024-09-23 13:42:56
LastEditTime: 2024-10-02 17:13:56
LastEditors: Maonan Wang
Description: 测试 rule-based policy 的效果
FilePath: /Multi-TSC-6G/eval_rule_based.py
'''
import json
from typing import List
from functools import partial

from env_utils.tsc_env import TSCEnvironment
from env_utils.global_local_wrapper import GlobalLocalInfoWrapper
# from rule_based_policy import (
#     ft_policy,
#     webster_policy,
#     actuated_policy # 感应控制可视化效果很好
# )
from rule_based_policy.traffic_rule_policy import TrafficPolicy

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'), file_log_level="INFO", terminal_log_level="INFO")

def load_environment_config(env_config_path):
    env_config_path = path_convert(f'./configs/env_configs/{env_config_path}')
    with open(env_config_path, 'r') as file:
        config = json.load(file)
    return config

def make_multi_envs(
        agent_tls_ids:List[str], 
        non_agent_tls_ids:List[str], 
        sumo_cfg:str, net_file:str,
        num_seconds:int, use_gui:bool,
        road_ids: List[str],
        log_file:str, cell_length:int=20,
        **output_files
    ):
    tsc_env = TSCEnvironment(
        sumo_cfg=sumo_cfg,
        net_file=net_file, # 用于加载路网的信息
        num_seconds=num_seconds,
        agent_tls_ids=agent_tls_ids,
        non_agent_tls_ids=non_agent_tls_ids,
        # tls_action_type='choose_next_phase_syn',
        tls_action_type='choose_next_phase',
        use_gui=use_gui,
        **output_files
    )
    tsc_env = GlobalLocalInfoWrapper(tsc_env, filepath=log_file, road_ids=road_ids, cell_length=cell_length)

    return tsc_env

def combine_actions(*original_actions):
    """
    从多个字典中分别取出指定位置的键值对并组合成一个新字典
    :param original_actions: 任意数量的字典
    :return: 组合后的新字典
    """
    result = {}
    try:
        for i, action in enumerate(original_actions):
            key, value = list(action.items())[i]
            result[key] = value
        return result
    except IndexError:
        print("Error: One or more of the action dictionaries do not have enough key-value pairs.")
        return {}

if __name__ == '__main__':
   
    # 读取实验配置文件
    env_config = load_environment_config("demo.json")
    
    sumo_cfg = path_convert(env_config['sumocfg'])
    net_file = path_convert(env_config['sumonet'])
    num_seconds = env_config['simulation_time']
    road_ids = env_config['road_ids']
    log_path = path_convert('./log/')
    env = make_multi_envs(
        agent_tls_ids=['E'], # 控制 5 个路口, 都是 4 相位
        non_agent_tls_ids=['B', 'D', 'F', 'H'],
        sumo_cfg=sumo_cfg,
        net_file=net_file,
        num_seconds=num_seconds,
        road_ids=road_ids,
        log_file=log_path,
        # use_gui=True,
        use_gui=False,
        cell_length=50,
        trip_info=path_convert('./trip_info_test.xml'),
        statistic_output=path_convert('./statistic_output_test.xml'),
        summary=path_convert('./summary_test.xml')
    )

    done = False
    eposide_reward = 0 # 累计奖励
    states = env.reset()
    (processed_local_obs, processed_global_obs, edge_cell_mask, processed_veh_obs, processed_veh_mask) = states
    
    # 初始化策略
    policy_B = TrafficPolicy(processed_local_obs)
    policy_D = TrafficPolicy(processed_local_obs)
    policy_E = TrafficPolicy(processed_local_obs)
    policy_F = TrafficPolicy(processed_local_obs)
    policy_H = TrafficPolicy(processed_local_obs)
    # ft_policy
    infos = {junction_id: {'can_perform_action': False} for junction_id in processed_local_obs.keys()}

    # 获得 junction phase 的信息
    junction_phase_group = {} # phase 包含哪些 movement
    junction_movement_ids = env.tls_movement_id.copy() # phase 中体征包含的 movement 的顺序
    for tls_id, tls_info in env.env.tsc_env.scene_objects['tls'].traffic_lights.items():
        junction_phase_group[tls_id] = tls_info.phase2movements.copy()
        
    while not done:
        # rule-based policy output action
        # original_action_B = policy_B.ft_policy(traffic_state=processed_local_obs, action_step=4, infos=infos) 
        # original_action_D = policy_D.ft_policy(traffic_state=processed_local_obs, action_step=4, infos=infos) 

        # # original_action_E = policy.actuated_policy(traffic_state=processed_local_obs, 
        # #                                 junction_phase_group=junction_phase_group, 
        # #                                 junction_movement_ids=junction_movement_ids,
        # #                                 infos=infos)
        # original_action_E = policy_E.mp_policy(traffic_state=processed_local_obs, 
        #                                 junction_phase_group=junction_phase_group, 
        #                                 junction_movement_ids=junction_movement_ids,
        #                                 infos=infos) 

        # original_action_F = policy_F.ft_policy(traffic_state=processed_local_obs, action_step=4, infos=infos) 
        # original_action_H = policy_H.ft_policy(traffic_state=processed_local_obs, action_step=4, infos=infos) 

        # original_action_B = policy_B.mp_policy(traffic_state=processed_local_obs, 
        #                                 junction_phase_group=junction_phase_group, 
        #                                 junction_movement_ids=junction_movement_ids,
        #                                 infos=infos) 
        # original_action_D = policy_D.mp_policy(traffic_state=processed_local_obs, 
        #                                 junction_phase_group=junction_phase_group, 
        #                                 junction_movement_ids=junction_movement_ids,
        #                                 infos=infos) 
        # original_action_E = policy_E.mp_policy(traffic_state=processed_local_obs, 
        #                                 junction_phase_group=junction_phase_group, 
        #                                 junction_movement_ids=junction_movement_ids,
        #                                 infos=infos) 

        # original_action_F = policy_F.mp_policy(traffic_state=processed_local_obs, 
        #                                 junction_phase_group=junction_phase_group, 
        #                                 junction_movement_ids=junction_movement_ids,
        #                                 infos=infos) 
        # original_action_H = policy_H.mp_policy(traffic_state=processed_local_obs, 
        #                                 junction_phase_group=junction_phase_group, 
        #                                 junction_movement_ids=junction_movement_ids,
        #                                 infos=infos) 

        original_action_B = policy_B.ft_policy(traffic_state=processed_local_obs, action_step=4, infos=infos) 
        original_action_D = policy_D.ft_policy(traffic_state=processed_local_obs, action_step=4, infos=infos) 
        original_action_E = policy_E.ft_policy(traffic_state=processed_local_obs, action_step=4, infos=infos) 
        original_action_F = policy_F.ft_policy(traffic_state=processed_local_obs, action_step=4, infos=infos) 
        original_action_H = policy_H.ft_policy(traffic_state=processed_local_obs, action_step=4, infos=infos) 

        # 组合动作
        action = combine_actions(original_action_B, original_action_D, original_action_E, original_action_F, original_action_H) # 这里要按照顺序

        states, rewards, truncateds, dones, infos = env.step(action)
        (processed_local_obs, processed_global_obs, edge_cell_mask, processed_veh_obs, processed_veh_mask) = states
        infos = infos
        eposide_reward += sum(rewards.values()) # 计算累计奖励
        done = all(dones.values()) or (len(dones) == 0)
    env.close()

    print(f"累计奖励为, {eposide_reward}.")