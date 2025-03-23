'''
@Author: WANG Maonan
@Date: 2024-04-25 16:27:48
@Description: Actor Network
LastEditTime: 2024-09-17 17:50:17
'''
import torch
from torch import nn
import torch.nn.functional as F
# import pdb

class ActorNetwork(nn.Module):
    def __init__(self, action_size):
        super(ActorNetwork, self).__init__()
        self.agent_embedding = nn.Embedding(5, 256) # 只考虑E的local observation
        
        self.fc1 = nn.Linear(in_features=60, out_features=256)  # 5*12=60
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=action_size)
        self.layer_norm1 = nn.LayerNorm(256)
        self.layer_norm2 = nn.LayerNorm(128)
        self.layer_norm3 = nn.LayerNorm(64)

    def forward(self, x1, x2):
        # print(x1) # E交叉口 torch.Size([1, 1, 5, 12, 8]
        # print(x2) # 其他交叉口 torch.Size([1, 4, 5, 12, 8]
        # pdb.set_trace()
        x_agent_id = x1['agent_id']  # torch.Size([1, 5, 1])或torch.Size([1, 1, 1])# (n_envs, batchsize, n_agents, id_embedding)

        x_local = x1['local']  # torch.Size([1, 1, 5, 12, 8]) # (n_envs, batchsize, n_agents, timeseries, n_movements, n_features)

        # # 对 non_agents 进行特殊处理
        # non_agents_local = x2['local']  # 形状是 [1, 4, 5, 12, 8]
        # # 对于 non_agents，直接输出一个固定的 logits (1维动作空间，值可以是0)
        # non_agents_logits = torch.zeros(non_agents_local.shape[1], 1).to(x2['local'].device) # torch.Size([4, 1])
        # env_batch_n_non_agents = list(non_agents_local.shape[:-3]) # [1,4]
        # # 将 non_agents_logits 扩展为 [1, 4, 1] 形状
        # non_agents_logits = non_agents_logits.view(*env_batch_n_non_agents, -1)  # [1, 4, 1]

        env_batch_nagents = list(x_local.shape[:-3])  # [1 5]或[1 1] # 包含 n_envs, batchsize 和 n_agents
        timeseries, movement, feature_num = x_local.shape[-3:] # 5 12 8

        x_local = x_local[..., 0].view(-1, timeseries * movement)  # torch.Size([5, 60]) 或 torch.Size([1, 60]) # 只获得占有率
        x_agent_embedding = self.agent_embedding(x_agent_id.view(-1).long()) # torch.Size([5, 256]) 或 torch.Size([1, 256])

        # First layer
        out = self.fc1(x_local) # torch.Size([5, 256]) 或 torch.Size([1, 256])
        out = self.layer_norm1(out + x_agent_embedding) # torch.Size([5, 256]) 或 torch.Size([1, 256])
        out = F.relu(out) # torch.Size([5, 256])或torch.Size([1, 256])

        # Second layer
        out = self.fc2(out) # torch.Size([5, 128])
        out = self.layer_norm2(out) # torch.Size([5, 128])或 torch.Size([1, 128])
        out = F.relu(out) # torch.Size([5, 128])或torch.Size([1, 128])

        # Third layer
        out = self.fc3(out) # torch.Size([5, 64])或torch.Size([1, 64])
        out = self.layer_norm3(out) # torch.Size([5, 64])或torch.Size([1, 64])
        out = F.relu(out) # torch.Size([5, 64])或torch.Size([1, 64])

        # Output layer
        out = self.fc4(out) # torch.Size([5, 4])或torch.Size([1, 4])
        
        # 将输出的 out 重新调整形状，使得它具有与输入相同的环境批量和代理数量维度。通过 view 函数将输出形状变为 [1, 5, 4]，即每个代理对应的 4 维动作空间（action size）。
        output = out.view(*env_batch_nagents, -1) # torch.Size([1, 1, 4])
        # print(out.view(*env_batch_nagents, -1)) # tensor([[[ 0.2212, -0.6701,  0.6197,  0.2078]]])
        # # Prepare the output as a dictionary
        # output = {
        #     ("agents", "logits"): out.view(*env_batch_nagents, -1),  # torch.Size([1, 5, 4]) or [1, 1, 4]
        #     ("non_agents", "logits"): non_agents_logits  # torch.Size([1, 4, 1])
        # }              
        return output 