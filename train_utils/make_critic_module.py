'''
@Author: WANG Maonan
@Date: 2024-04-15 16:09:29
@Description: 创建 Critic Network
1. 中心化的 critic, 也就是使用全局的信息, 最后输出是 1
2. 共享权重, 所有的 agent 的 critic 使用一样的权重
@LastEditTime: 2024-04-25 16:48:54
'''
import torch
import importlib

from torchrl.modules import (
    ValueOperator, # 从torchrl.modules模块导入ValueOperator类。这个类可能用于处理与值函数相关的操作，在强化学习中，值函数是评估状态或状态-动作对的重要组件。
)

def load_critic_model(model_name): # 该函数的输入参数，是一个字符串，表示要加载的模型名称。
    module = importlib.import_module(f'train_utils.{model_name}') # 动态加载名为 train_utils.{model_name} 的模块。这个模块通常存放在项目的 train_utils 目录下。
    CriticNetwork = getattr(module, 'CriticNetwork') # 从加载的模块中获取 CriticNetwork 类。CriticNetwork 是一个用于创建和管理 critic 网络的类。
    return CriticNetwork
    
class critic_module():
    def __init__(self, model_name, device) -> None:
        CriticNetwork = load_critic_model(model_name) # 调用之前定义的 load_critic_model 函数，根据传入的 model_name 加载对应的 CriticNetwork 类。
        self.critic_net = CriticNetwork().to(device)

    def make_critic_module(self):
        value_module = ValueOperator(
            module=self.critic_net, # 这意味着 ValueOperator 将使用 critic_net 网络来计算值函数。
            in_keys=[("agents", "observation"), ("non_agents", "observation")],
        )
        return value_module
    
    def save_model(self, model_path):
        torch.save(self.critic_net.state_dict(), model_path)
    
    def load_model(self, model_path):
        model_dicts = torch.load(model_path)
        self.critic_net.load_state_dict(model_dicts)