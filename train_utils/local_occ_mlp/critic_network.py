'''
@Author: WANG Maonan
@Date: 2024-04-25 16:27:56
@Description: Critic Network
LastEditTime: 2024-09-17 17:50:45
'''
from torch import nn
import torch.nn.functional as F

class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.agent_embedding = nn.Embedding(5, 256)
        self.fc1 = nn.Linear(in_features=60, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=1)
        self.layer_norm1 = nn.LayerNorm(256)
        self.layer_norm2 = nn.LayerNorm(128)
        self.layer_norm3 = nn.LayerNorm(64)

    def forward(self, x1, x2):
        x_agent_id = x1['agent_id']  # (n_envs, batchsize, n_agents, id_embedding)

        x_local = x1['local']  # torch.Size([1, 400, 1, 5, 12, 8]) # (n_envs, batchsize, n_agents, timeseries, n_movements, n_features)
        
        env_batch_nagents = list(x_local.shape[:-3])  # 包含 n_envs, batchsize 和 n_agents
        timeseries, movement, feature_num = x_local.shape[-3:]

        x_local = x_local[..., 0].view(-1, timeseries * movement)  # 只获得占有率
        x_agent_embedding = self.agent_embedding(x_agent_id.view(-1).long())

        # First layer
        out = self.fc1(x_local)
        out = self.layer_norm1(out + x_agent_embedding)
        out = F.relu(out)

        # Second layer
        out = self.fc2(out)
        out = self.layer_norm2(out)
        out = F.relu(out)

        # Third layer
        out = self.fc3(out)
        out = self.layer_norm3(out)
        out = F.relu(out)

        # Output layer
        out = self.fc4(out)

        return out.view(*env_batch_nagents, -1)