'''
@Author: WANG Maonan
@Date: 2024-04-15 03:58:19
@Description: 创建多智能体的环境
LastEditTime: 2024-09-17 16:27:00
'''
from typing import List, Dict
from env_utils.tsc_env import TSCEnvironment
from env_utils.global_local_wrapper import GlobalLocalInfoWrapper
from env_utils.pz_env import TSCEnvironmentPZ

from torchrl.envs import (
    ParallelEnv, 
    TransformedEnv,
    RewardSum,
    VecNorm
)
from env_utils.torchrl_pz_wrapper import PettingZooWrapper # 对原始的 torchrl 的 wrapper 进行了修改


def make_multi_envs(
        agent_tls_ids:List[str], 
        non_agent_tls_ids:List[str], 
        sumo_cfg:str, net_file:str,
        num_seconds:int, use_gui:bool,
        road_ids:List[str],
        action_space:Dict[str, int],
        cell_length:int,
        log_file:str, device:str='cpu',
        **output_files
    ):
    tsc_env = TSCEnvironment( # 创建 TSCEnvironment 实例，初始化交通信号控制环境。参数包括配置文件、网络文件、仿真时间、信号灯 ID、动作类型和图形界面选项。
        **output_files,
        sumo_cfg=sumo_cfg,
        net_file=net_file,
        num_seconds=num_seconds,
        agent_tls_ids=agent_tls_ids,
        non_agent_tls_ids=non_agent_tls_ids,
        tls_action_type='choose_next_phase',
        use_gui=use_gui,
    )
    tsc_env = GlobalLocalInfoWrapper(tsc_env, filepath=log_file, road_ids=road_ids, cell_length=cell_length) # 使用 GlobalLocalInfoWrapper 包装 tsc_env，以便在环境中使用全局和局部信息，并指定日志文件、道路 ID 和单元格长度。
    tsc_env = TSCEnvironmentPZ(tsc_env, action_space) # 将 tsc_env 再次包装成 TSCEnvironmentPZ，这是一个兼容 PettingZoo 的环境，接收定义好的动作空间。
    tsc_env = PettingZooWrapper( # 用 PettingZooWrapper 包装环境，允许在 PettingZoo 中进行交互。这里定义了代理组（agents）、是否使用分类动作、是否使用遮罩、设备类型和结束条件（所有代理都结束才算结束）。
        tsc_env, 
        group_map={'agents':agent_tls_ids, 'non_agents':non_agent_tls_ids}, # agent 可以分类, 例如不同动作空间大小
        categorical_actions=False,
        use_mask=False, # 智能体数量动态变化, 手动将 obs 和 reward 设置为 0
        device=device,
        done_on_any=False # 所有都结束才结束
    )
    tsc_env = TransformedEnv(tsc_env) # 创建一个变换环境的实例，以便后续对环境的奖励和观察进行变换。
    # print(tsc_env.reward_key) # 0 =('agents', 'reward') 1 = ('non_agents', 'reward')
    ### 原本的
    # tsc_env.append_transform(RewardSum(in_keys=[tsc_env.reward_key])) # 添加奖励变换，使得奖励总和可以在环境中计算。
    # tsc_env.append_transform(VecNorm(in_keys=[tsc_env.reward_key])) # 添加向量归一化变换，用于归一化奖励，以帮助训练过程中的稳定性。
    # 选择多个奖励键并传递给 RewardSum 和 VecNorm 变换
    tsc_env.append_transform(RewardSum(in_keys=[('agents', 'reward'), ('non_agents', 'reward')]))  # 计算代理和非代理的奖励总和
    tsc_env.append_transform(VecNorm(in_keys=[('agents', 'reward'), ('non_agents', 'reward')]))  # 对代理和非代理的奖励进行向量归一化
    # # 只对代理的奖励进行处理
    # tsc_env.append_transform(RewardSum(in_keys=[('agents', 'reward')]))  # 只计算代理奖励的总和
    # tsc_env.append_transform(VecNorm(in_keys=[('agents', 'reward')]))  # 只对代理奖励进行归一化
        
    return tsc_env                      

def make_parallel_env(
        num_envs:int,
        agent_tls_ids:List[str], 
        non_agent_tls_ids:List[str], 
        sumo_cfg:str, net_file:str,
        num_seconds:int, use_gui:bool,
        road_ids:List[str],
        action_space:Dict[str, int],
        cell_length:int,
        log_file:str,
        device:str='cpu'
    ):
    env = ParallelEnv(
        num_workers=num_envs,
        create_env_fn=make_multi_envs,
        create_env_kwargs=[{
            "agent_tls_ids": agent_tls_ids,
            "non_agent_tls_ids": non_agent_tls_ids,
            "sumo_cfg": sumo_cfg,
            "num_seconds": num_seconds,
            "net_file": net_file,
            "action_space": action_space,
            "road_ids": road_ids,
            "cell_length": cell_length,
            "use_gui" : use_gui,
            "log_file": log_file+f'/{i}',
            "device": device,
        }
        for i in range(num_envs)]
    )

    return env