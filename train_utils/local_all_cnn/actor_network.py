'''
@Author: WANG Maonan
@Date: 2024-04-25 16:39:36
@Description: Actor Network 
LastEditTime: 2024-09-17 16:15:26
'''
from torch import nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, action_size):
        super(ActorNetwork, self).__init__()
        self.conv = nn.Conv2d(in_channels=5, out_channels=32, kernel_size=(1, 8)) # 创建一个二维卷积层conv，输入通道数为5，输出通道数为32，卷积核大小为(1, 7)。
        self.pool = nn.AdaptiveAvgPool1d(output_size=1) # 输出维度为(batch_size, 32, 1, 1) # 创建一个自适应平均池化层pool，将输入的最后一维输出到大小为1
        self.fc = nn.Linear(in_features=32, out_features=action_size) # 创建一个全连接层fc，输入特征数为32，输出特征数为action_size，用于生成动作的概率分布。

    def forward(self, x1, x2):
        x = x1["local"] # 从输入字典中提取"local"键对应的值。 # torch.Size([1, 1, 5, 12, 8])
        env_batch_nagents = list(x.shape[:-3]) # 包含 n_envs, batchsize 和 n_agents # 获取输入张量的形状，除了最后三个维度，组成一个列表env_batch_nagents，表示环境数量、批大小和代理数量。
        timeseries, movement, feature_num = x.shape[-3:] # 获取输入张量的最后三个维度，分别赋值给timeseries、movement和feature_num。

        x = x.view(-1, timeseries, movement, feature_num) # (batch_size*agent_num, 5, 12, 8) # 重新调整输入张量的形状，将其展平为(batch_size*agent_num, 5, 12, 8)。

        x = self.conv(x) # (batch_size*agent_num, 5, 12, 8) --> (batch_size*agent_num, 32, 12, 1), 分析 x[2,0].round(decimals=2) # 通过卷积层处理输入x，输出形状变为(batch_size*agent_num, 32, 12, 1)。
        x = F.tanh(x) # 对卷积输出应用双曲正切激活函数tanh，引入非线性。
        x = x.squeeze(-1) # (batch_size*agent_num, 32, 12) # 压缩最后一个维度，去掉大小为1的维度，输出形状为(batch_size*agent_num, 32, 12)。
        x = self.pool(x) # (batch_size*agent_num, 32, 1) # 应用自适应平均池化层，输出形状为(batch_size*agent_num, 32, 1)。
        x = x.squeeze(-1) # (batch_size*agent_num, 32) # 再次压缩最后一个维度，输出形状变为(batch_size*agent_num, 32)。
        x = F.tanh(x) # 再次应用tanh激活函数，进一步处理数据。
        x = self.fc(x) # (batch_size*agent_num, 3), 3 是动作个数 # 通过全连接层处理，输出形状为(batch_size*agent_num, 3)，其中3是动作的数量。
        return x.view(*env_batch_nagents, -1) # 最后，将输出调整为原来的环境、批大小和代理数量的形状，并返回结果。