'''
@Author: WANG Maonan
@Date: 2023-10-30 23:15:18
@Description: 创建 Actor Module
1. 不是中心化, 即每个 agent 根据自己的 obs 进行决策
2. 模型的权重是共享的, 因为 agent 是相同的类型, 所以只有一个 actor 的权重即可
@LastEditTime: 2024-04-25 17:16:49
'''
import torch
import importlib
from loguru import logger
from tensordict.nn import TensorDictModule

from torchrl.modules import ( # 从 torchrl 库中导入 OneHotCategorical 和 ProbabilisticActor。这两个模块用于定义概率分布和策略网络。
    OneHotCategorical,
    ProbabilisticActor,
)

def load_actor_model(model_name): # 定义一个函数 load_actor_model，用于动态加载指定的 Actor 模型：
    module = importlib.import_module(f'train_utils.{model_name}') # importlib.import_module：根据 model_name 动态导入相应模块。
    ActorNetwork = getattr(module, 'ActorNetwork') # getattr(module, 'ActorNetwork')：从导入的模块中获取 ActorNetwork 类。
    return ActorNetwork


class policy_module(): # 定义一个名为 policy_module 的类，表示策略模块。
    def __init__(self, model_name, action_spec, device) -> None:
        ActorNetwork = load_actor_model(model_name) # 调用 load_actor_model 函数，加载指定名称的 Actor 网络类，并将其赋值给 ActorNetwork。
        self.action_spec = action_spec # 将 action_spec 保存为类的一个属性，以便后续使用。
        self.actor_net = ActorNetwork(action_size=action_spec[0]['agents']['action'].shape[-1]).to(device) # 创建 Actor 网络的实例 actor_net，传入动作空间的大小（action_spec.shape[-1]），并将模型移动到指定的设备上。        
        self.device = device
        logger.info(f'RL: Actor Model:\n {self.actor_net}') # 使用 logger 记录 Actor 模型的信息，以便于调试和日志跟踪。
    
    def make_policy_module(self): # 定义一个方法 make_policy_module，用于创建策略模块。
       
        policy_module_agent = TensorDictModule( # 创建一个 TensorDictModule 实例 policy_module，用于将 Actor 网络与输入输出键进行关联：
            self.actor_net, # self.actor_net：使用的 Actor 网络。
            in_keys=[("agents", "observation"), ("non_agents", "observation")], # in_keys：输入键，表示输入的观测来自 agents 字典中的 observation。
            out_keys=[("agents", "logits")], # out_keys：输出键，表示输出的 logits 存储在 agents 字典中的 logits。      
        )

        policy_agent = ProbabilisticActor( # 创建一个 ProbabilisticActor 实例 policy，用于定义策略：
            module=policy_module_agent, # module=policy_module：使用刚刚创建的 policy_module。
            spec=self.action_spec[0]['agents']['action'], # spec=self.action_spec：传入动作规范。
            in_keys={ # 指定输入的 logits 来源。
                "logits":("agents", "logits"),
                # "mask":("agents", "action_mask") # 这里需要传入一个 mask, 根据 agent mask 自己进行计算
            }, 
            # in_keys=[("agents", "logits"), ("non_agents", "logits")],# 使用列表来定义输入键
            out_keys=[("agents", "action")], # env.action_key # out_keys：输出的动作存储在 agents 字典中的 action。
            distribution_class=OneHotCategorical, # distribution_class=OneHotCategorical：指定分布类为 OneHotCategorical，用于处理离散动作。
            return_log_prob=True, # return_log_prob=True：返回动作的对数概率。
        )

        return policy_agent
    
    def save_model(self, model_path):
        torch.save(self.actor_net.state_dict(), model_path)
    
    def load_model(self, model_path):
        model_dicts = torch.load(model_path, map_location=self.device) # 使用 torch.load 加载模型参数，并根据指定的设备将其映射到相应的位置。
        self.actor_net.load_state_dict(model_dicts)