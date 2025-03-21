'''
@Author: WANG Maonan
@Date: 2024-04-15 23:41:58
@Description: 加载 MAPPO 模型进行测试
LastEditTime: 2024-09-20 13:00:04
'''
import os
import json
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from train_utils.make_actor_module import policy_module
from env_utils.make_multi_tsc_env import make_multi_envs

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))

def load_environment_config(env_config_path):
    env_config_path = path_convert(f'./configs/env_configs/{env_config_path}')
    with open(env_config_path, 'r') as file:
        config = json.load(file)
    return config

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 定义实验名称
    scenario_name = "demo" # 场景名称
    model_name = "2_allcnn" # 模型的名称
    exp_config_path=path_convert(f'./configs/exp_configs/{scenario_name}/{model_name}.json')

    # 读取配置文件
    with open(exp_config_path, 'r') as file:
        exp_config = json.load(file)
    env_config = load_environment_config(exp_config['environment_name'])
    
    exp_name = exp_config["experiment_name"] # 实验名称
    sumo_cfg = path_convert(env_config["sumocfg"])
    net_file = path_convert(env_config["sumonet"])
    log_path = path_convert(f'./log/eval_{exp_name}')
    model_path = path_convert(f'./mappo_models/{exp_name}/')
    
    model_name = exp_config["model_name"]
    action_space = env_config["action_space"]
    agent_tls_ids = env_config["agent_ids"]
    non_agent_tls_ids = env_config["non_agent_ids"]
    road_ids = env_config['road_ids']
    num_seconds = env_config["simulation_time"] # 仿真时间

    # 1. Create Env
    tsc_env = make_multi_envs(
        agent_tls_ids=agent_tls_ids,
        non_agent_tls_ids=non_agent_tls_ids,
        sumo_cfg=sumo_cfg,
        net_file=net_file,
        action_space=action_space,
        num_seconds=num_seconds,
        road_ids=road_ids,
        use_gui=False,
        cell_length=100,
        log_file=log_path,
        device=device,
        trip_info=path_convert('./trip_info.xml'),
        statistic_output=path_convert('./statistic_output.xml'),
        summary=path_convert('./summary.xml')
    )

    # 2. Load Model Dict
    action_spec = tsc_env.action_spec
    action_spec = action_spec.unsqueeze(0)
    policy_gen = policy_module(model_name, action_spec, device)
    policy_gen.load_model(os.path.join(model_path, "95_actor.pkl"))
    policy = policy_gen.make_policy_module()

    # 3. Simulation with environment using the policy, ExplorationType.MODE, ExplorationType.RANDOM
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        rollouts = tsc_env.rollout(
            policy=policy,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=False,
            max_steps=2_000
        )