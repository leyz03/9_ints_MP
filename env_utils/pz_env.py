'''
@Author: WANG Maonan
@Date: 2024-04-14 17:15:14
@Description: Petting Zoo Wrapper
=> 关于 petting zoo 环境创建, 参考 
    1. https://pettingzoo.farama.org/content/environment_creation/
    2. https://pettingzoo.farama.org/tutorials/custom_environment/3-action-masking/ (添加 action mask)
=> 由于不是每一个时刻所有 TSC 都可以做动作, 这里我们就只返回可以做动作的 TSC 的信息, 也就是 agent 的数量是一直在改变的
LastEditTime: 2024-09-17 17:15:32
'''
import functools
import numpy as np
import gymnasium as gym

from typing import Dict
from pettingzoo import ParallelEnv

# import pdb

class TSCEnvironmentPZ(ParallelEnv): # 定义了一个名为 TSCEnvironmentPZ 的类，继承自 ParallelEnv，并设置元数据，指明环境名称。
    metadata = {
        "name": "multi_agent_tsc_env",
    }
    
    def __init__(self, env, action_space):
        super().__init__()
        self.env = env
        self.render_mode = None
        
        # agent id == tls id (agent id 就是所有信号灯的 id)
        self.agents_id = self.env.unwrapped.agent_tls_ids
        self.non_agents_id = self.env.unwrapped.non_agent_tls_ids
        self.all_tsc = (self.env.unwrapped.agent_tls_ids + self.env.unwrapped.non_agent_tls_ids)

        # possible agents实际上就是所有的agents，不包括non agents
        self.possible_agents = self.all_tsc # 将所有可能的代理（交通信号灯的ID）赋值给 possible_agents，并复制到 agents。
        self.agents = self.possible_agents.copy() # 实际上这是能行动的agents

        # spaces
        self.action_spaces = { # 为每个交通信号灯创建离散的动作空间，动作数量来自于 action_space。
            _tls_id:gym.spaces.Discrete(action_space[_tls_id]) # 每一个信号灯的相位个数
            for _tls_id in self.all_tsc
        }

        self.observation_spaces = { # 为每个交通信号灯创建观察空间，包括代理ID和局部观察数据（一个形状为 (5, 12, 7) 的盒子）。
            _tls_id:gym.spaces.Dict({
                "agent_id": gym.spaces.Box(low=0, high=len(self.agents_id)), # 表明 agent id, 为了区分不同的 agent
                "local": gym.spaces.Box(
                    low=np.zeros((5,12,8)),
                    high=np.ones((5,12,8)),
                    shape=(5,12,8,)
                ),
                # "global": gym.spaces.Box(
                #     low=np.zeros((294,5,56,3)), # 3ints-(20,5,11,3)
                #     high=np.ones((294,5,56,3)),
                #     shape=(294,5,56,3,)
                # ), # 20 个 edge, 每个 edge 包含 5s 的数据, 每个 edge 有 11 个 cell, 每个 cell 有 3 个信息
                # "global_mask": gym.spaces.Box(
                #     low=np.zeros((294,56)), # 3int-(20,11)
                #     high=np.ones((294,56)),
                #     shape=(294,56)
                # ),
                # "vehicle": gym.spaces.Box(
                #     low=np.zeros((5,100,299)), # TODO, 这里车辆的 road id 也是需要修改的
                #     high=100*np.ones((5,100,299)),
                #     shape=(5,100,299)
                # ),
                # "vehicle_mask": gym.spaces.Box(
                #     low=np.zeros((5,100)),
                #     high=np.ones((5,100)),
                #     shape=(5,100)
                # ),                
            })
            for _tls_id in self.agents_id
        }

        # 现在遍历 self.non_agents_id 并为每个 non_agent_id 创建相同的结构
        for _non_agent_id in self.non_agents_id:
            self.observation_spaces[_non_agent_id] = gym.spaces.Dict({
                "non_agent_id": gym.spaces.Box(low=0, high=len(self.non_agents_id)),  # 表明 non-agent id, 为了区分不同的 non-agent
                "local": gym.spaces.Box(
                    low=np.zeros((5, 12, 8)),
                    high=np.ones((5, 12, 8)),
                    shape=(5, 12, 8,)
                ),
            })

    def reset(self, seed=None, options=None):
        """Reset the environment
        """
        # 调用底层环境的 reset 方法，获取处理后的局部观察、全局观察、边缘单元掩码、车辆观察和车辆掩码。
        processed_local_obs, processed_global_obs, edge_cell_mask, processed_veh_obs, processed_veh_mask = self.env.reset()
        agent_mask = { # 创建一个代理掩码，标记所有代理都可以执行动作。
            _tls_id:{
                'can_perform_action': True, 
            }
            for _tls_id in self.all_tsc # 这里原本是self.possible_agents
        }
        # # 遍历 non_agents_id，设置它们的 can_perform_action 为 False
        # for _tls_id in self.non_agents_id:
        #     agent_mask[_tls_id] = {
        #         'can_perform_action': False,
        #     }

        self.agents = self.possible_agents[:] # 可以做动作的 agent # 将当前可以执行动作的代理列表赋值给 agents。

        # 处理 observation
        # 只有能行动的agents才进行观察
        observations = { 
            # 处理智能体的观察数据
            **{
                _tls_id: {
                    'agent_id': _tls_index,
                    'local': processed_local_obs[_tls_id],
                    # 'global': processed_global_obs,
                    # 'global_mask': edge_cell_mask,
                    # 'vehicle': processed_veh_obs[_tls_id],
                    # 'vehicle_mask': processed_veh_mask[_tls_id]
                }
                for _tls_index, _tls_id in enumerate(self.agents_id) # 实际上是agents_act
            },
            # 处理非智能体的观察数据
            **{
                _tls_id: {
                    'non_agent_id': _tls_index,
                    'local': processed_local_obs[_tls_id],
                    # 'global': processed_global_obs,
                    # 'global_mask': edge_cell_mask,
                    # 'vehicle': processed_veh_obs[_tls_id],
                    # 'vehicle_mask': processed_veh_mask[_tls_id]
                }
                for _tls_index, _tls_id in enumerate(self.non_agents_id)
            }
        }
        # pdb.set_trace()
        return observations, agent_mask
    
    @functools.lru_cache(maxsize=None) # 定义一个使用 LRU 缓存的 observation_space 方法，用于返回指定代理的观察空间。
    def observation_space(self, agent):
        """Return the observation space for the agent.
        """
        return self.observation_spaces[agent] # 返回指定代理的观察空间。
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent): # 定义一个使用 LRU 缓存的 action_space 方法，用于返回指定代理的动作空间。
        """Return the action space for the agent.
        """
        return self.action_spaces[agent]
    
    def close(self):
        """Close the environment and stop the SUMO simulation.
        """
        self.env.close()

    def step(self, actions:Dict[str, int]):
        """Step the environment.
        """
        # pdb.set_trace()
        (processed_local_obs, processed_global_obs, edge_cell_mask, processed_veh_obs, processed_veh_mask), rewards, terminations, truncations, infos = self.env.step(actions) # 调用底层环境的 step 方法，处理动作并获取观察、奖励、终止标志、截断标志和信息。

        # 将不能做动作的 agent 设置为 0
        pz_observations = {}
        pz_rewards = {}
        
        # 处理 agents的observation
        for _tls_index, _tls_id in enumerate(self.possible_agents): # 遍历所有可能的代理，填充观察和奖励数据。
            pz_observations[_tls_id] = {
                'agent_id': _tls_index,
                'local': processed_local_obs[_tls_id],
                # 'global': processed_global_obs,
                # 'global_mask': edge_cell_mask,
                # 'vehicle': processed_veh_obs[_tls_id],
                # 'vehicle_mask': processed_veh_mask[_tls_id]
            }
            pz_rewards[_tls_id] = rewards[_tls_id]

        # 接着处理 self.non_agents 的observation
        for _tls_index, _tls_id in enumerate(self.non_agents_id):
            pz_observations[_tls_id] = {
                'non_agent_id': _tls_index,
                'local': processed_local_obs[_tls_id],
                # 'global': processed_global_obs,
                # 'global_mask': edge_cell_mask,
                # 'vehicle': processed_veh_obs[_tls_id],
                # 'vehicle_mask': processed_veh_mask[_tls_id]
            }
            pz_rewards[_tls_id] = rewards[_tls_id]

        # pdb.set_trace()
        return pz_observations, pz_rewards, terminations, truncations, infos # 返回更新后的观察、奖励、终止标志、截断标志和其他信息。