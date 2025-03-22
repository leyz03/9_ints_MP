'''
@Author: WANG Maonan
@Date: 2024-04-25 16:27:56
@Description: Critic Network (根据全局信息做出判断)
LastEditTime: 2024-09-17 16:32:38
'''
from torch import nn
import torch.nn.functional as F

class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.conv = nn.Conv2d(in_channels=5, out_channels=32, kernel_size=(1, 8))
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Linear(in_features=32, out_features=1)

    def forward(self, local_global_x, x2): # 定义前向传播方法，接受一个包含局部和全局信息的输入字典 local_global_x。
        # print(local_global_x.keys()) # torch.Size([1, 400, 1]) # _StringKeys(dict_keys(['agent_id', 'local']))
        x_local = local_global_x['local'] # torch.Size([1, 400, 1, 5, 12, 8])
        # x_global = local_global_x['global']
        env_batch_nagents = list(x_local.shape[:-3]) # 包含 n_envs, batchsize 和 n_agents # 获取输入数据的形状信息，包含环境数量、批次大小和代理数量。
        timeseries, movement, feature_num = x_local.shape[-3:]
        x_local = x_local.view(-1, timeseries, movement, feature_num) # (batch_size, agent_num, 5, movement, movement_feature) # 调整输入的形状，将其变为 (batch_size, agent_num, 5, movement, movement_feature)。

        x_local = self.conv(x_local) # (batch_size*agent_num, 5, 12, 8) --> (batch_size*agent_num, 32, 12, 1)
        x_local = F.relu(x_local) # torch.Size([400, 32, 12, 1])
        x_local = x_local.squeeze(-1) # (batch_size*agent_num, 32, 12)
        x_local = self.pool(x_local) # (batch_size*agent_num, 32, 1)
        x_local = x_local.squeeze(-1) # (batch_size*agent_num, 32)
        x_local = self.fc(x_local) # (batch_size*agent_num, 1)

        return x_local.view(*env_batch_nagents, -1) # 将输出重新调整形状为包含环境、批次和代理的形式，返回最终结果。