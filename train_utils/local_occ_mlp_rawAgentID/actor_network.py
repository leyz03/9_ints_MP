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
        # Process agent data (x1)
        x_agent_id = x1['agent_id']  # torch.Size([1, 5, 1])或torch.Size([1, 1, 1])
        x_local = x1['local']  # torch.Size([1, 1, 5, 12, 8])
        
        # Process non_agent data (x2)
        non_agents_local = x2['local']  # 形状是 [1, 4, 5, 12, 8]
        
        # Create fixed logits for non_agents (zeros) - FIX HERE
        # First get the proper batch dimensions
        env_batch_n_non_agents = list(non_agents_local.shape[:-3])  # [1,4]
        # Create tensor with proper size directly
        non_agents_logits = torch.zeros(*env_batch_n_non_agents, 1).to(x_local.device)  # [1, 4, 1]
        
        # Continue with agent processing
        env_batch_nagents = list(x_local.shape[:-3])  # [1 5]或[1 1]
        timeseries, movement, feature_num = x_local.shape[-3:]  # 5 12 8
        
        x_local = x_local[..., 0].view(-1, timeseries * movement)  # [5, 60] or [1, 60]
        x_agent_embedding = self.agent_embedding(x_agent_id.view(-1).long())
        
        # Model layers (unchanged)
        out = self.fc1(x_local)
        out = self.layer_norm1(out + x_agent_embedding)
        out = F.relu(out)
        
        out = self.fc2(out)
        out = self.layer_norm2(out)
        out = F.relu(out)
        
        out = self.fc3(out)
        out = self.layer_norm3(out)
        out = F.relu(out)
        
        out = self.fc4(out)
        
        # Return both agent and non_agent outputs in a dictionary
        output = {
            ("agents", "logits"): out.view(*env_batch_nagents, -1),  # [1, 5, 4] or [1, 1, 4]
            ("non_agents", "logits"): non_agents_logits  # [1, 4, 1]
        }
        
        return output