'''
@Author: WANG Maonan
@Date: 2023-10-30 23:42:12
@Description: Train & Test Log Module
@LastEditTime: 2023-10-31 21:21:56
'''
import torch

from tensordict import TensorDictBase
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.record.loggers import Logger


def log_training(
    logger: Logger,
    training_td: TensorDictBase,
    sampling_td: TensorDictBase,
    sampling_time: float,
    training_time: float,
    total_time: float,
    iteration: int,
    current_frames: int, # 当前帧数和总帧数。
    total_frames: int,
    step: int,
):
    if ("next", "agents", "reward") not in sampling_td.keys(True, True): # 检查sampling_td字典中是否存在("next", "agents", "reward")这个键，如果不存在，则创建它。
        sampling_td.set( # sampling_td.get(("next", "reward"))获取下一个时间步的奖励。
            ("next", "agents", "reward"),
            sampling_td.get(("next", "reward"))
            .expand(sampling_td.get("agents").shape) # 使用expand方法扩展奖励的形状，使其与agents的形状匹配。
            .unsqueeze(-1), # 在最后一个维度上增加一个维度，以便与预期的数据形状一致。
        )
    if ("next", "agents", "episode_reward") not in sampling_td.keys(True, True): # 与前面的代码类似，检查并设置episode_reward（即每个回合的奖励）。
        sampling_td.set( # 如果episode_reward键不存在，创建并设置它。episode_reward表示每个代理在当前回合中的奖励。
            ("next", "agents", "episode_reward"),
            sampling_td.get(("next", "episode_reward"))
            .expand(sampling_td.get("agents").shape)
            .unsqueeze(-1),
        )

    to_log = { # 创建一个字典to_log，将training_td字典中的每个值取均值并转换为Python原生类型（.item()）。这是为了减少数据的维度，使其便于记录。
        f"train/learner/{key}": value.mean().item()
        for key, value in training_td.items() # 这些键值对的形式是train/learner/{key}，例如可能是train/learner/actor_loss等。
    } # 这里创建一个to_log字典，其中包含训练数据中的所有值。每个值都会求均值（mean()），并将结果转换为Python的原生类型（.item()），这是为了方便后续记录。

    if "info" in sampling_td.get("agents").keys(): # 如果sampling_td中的agents部分包含"info"字段，获取这些信息并将其添加到to_log字典中。info可能包含代理的其他辅助信息（例如状态值、动作等）。
        to_log.update(
            {
                f"train/info/{key}": value.mean().item()
                for key, value in sampling_td.get(("agents", "info")).items()
            }
        )

    reward = sampling_td.get(("next", "agents", "reward")).mean(-2)  # Mean over agents # 获取下一个时间步的奖励并计算所有代理的奖励均值。mean(-2)表示对所有代理进行平均，假设奖励数据的维度中包含多个代理。
    done = sampling_td.get(("next", "done")) # done表示一个回合是否结束。如果done为True，则当前回合结束。
    if done.ndim > reward.ndim: # 如果done的维度大于reward的维度，调整done的维度，使其与奖励数据的维度一致
        done = done[..., 0, :]  # Remove expanded agent dim
    
    episode_reward = sampling_td.get(("next", "agents", "episode_reward")).mean(-2)[done] # 获取每个代理的回合奖励并计算所有代理的回合奖励均值。done标志用于选择已结束的回合。
    to_log.update( # 将一些额外的统计数据添加到日志中，如最小奖励、最大奖励、平均奖励，训练时间、采样时间、迭代时间等。
        {
            "train/reward/reward_min": reward.min().item(),
            "train/reward/reward_mean": reward.mean().item(),
            "train/reward/reward_max": reward.max().item(),
            "train/reward/episode_reward_min": episode_reward.min().item() if episode_reward.numel() > 0 else float('nan'),
            "train/reward/episode_reward_mean": episode_reward.mean().item() if episode_reward.numel() > 0 else float('nan'),
            "train/reward/episode_reward_max": episode_reward.max().item() if episode_reward.numel() > 0 else float('nan'),
            "train/sampling_time": sampling_time,
            "train/training_time": training_time,
            "train/iteration_time": training_time + sampling_time,
            "train/total_time": total_time,
            "train/training_iteration": iteration,
            "train/current_frames": current_frames,
            "train/total_frames": total_frames,
        }
    )

    for key, value in to_log.items(): # 将to_log字典中的每个数据记录到日志中，step表示当前的训练步骤。
        logger.log_scalar(key, value, step=step)

    return to_log


def log_evaluation(
    logger: Logger, # 日志记录器。
    rollouts: TensorDictBase, # 包含评估数据的张量字典。
    env_test: VmasEnv,
    evaluation_time: float,
    step: int, # 当前评估步骤。
):
    rollouts = list(rollouts.unbind(0)) # 将rollouts展开成一个列表，每个元素是一个独立的轨迹（TensorDict）。
    for k, r in enumerate(rollouts): # 遍历每条轨迹，计算done标志，并找到第一个done为True的位置（表示一个回合结束）。然后将轨迹截断，保留到第一个回合结束的时间步。
        next_done = r.get(("next", "done")).sum(
            tuple(range(r.batch_dims, r.get(("next", "done")).ndim)),
            dtype=torch.bool,
        )
        done_index = next_done.nonzero(as_tuple=True)[0][
            0
        ]  # First done index for this traj
        rollouts[k] = r[: done_index + 1]

    rewards = [td.get(("next", "agents", "reward")).sum(0).mean() for td in rollouts] # 计算每条轨迹的总奖励，并对所有轨迹的奖励取平均。
    to_log = { # 用于记录评估指标的日志。这个字典将包含一些与轨迹相关的统计数据。
        "eval/episode_reward_min": min(rewards), # 记录所有轨迹奖励中的最小值。
        "eval/episode_reward_max": max(rewards), # 记录所有轨迹奖励中的最大值。
        "eval/episode_reward_mean": sum(rewards) / len(rollouts), # 计算并记录所有轨迹奖励的平均值。这里通过计算 rewards 列表的总和并除以轨迹数量来得到平均奖励。
        "eval/episode_len_mean": sum([td.batch_size[0] for td in rollouts]) # 记录每条轨迹的平均长度。td.batch_size[0] 是每条轨迹的长度，假设每条轨迹的 batch_size 是一个列表或元组，因此 [0] 取第一个元素，即轨迹的长度。通过对所有轨迹长度求和并除以轨迹数，得到平均轨迹长度。
        / len(rollouts),
        "eval/evaluation_time": evaluation_time, # 记录评估的时间（假设 evaluation_time 是在程序其他地方计算的时间，单位可能是秒）。
    }

    for key, value in to_log.items(): # 解释：这行代码用于将上面字典 to_log 中的每个指标记录到日志中。 # to_log.items() 遍历字典中的每个键值对。
        logger.log_scalar(key, value, step=step) # logger.log_scalar(key, value, step=step) 记录每个指标的值到日志中，其中 key 是指标的名称（例如 "eval/episode_reward_min"），value 是该指标的值（例如最小奖励），step 是当前训练的步骤（通常用于跟踪训练过程中的指标变化）。

    return sum(rewards) / len(rollouts) # 这行代码返回了所有轨迹的平均奖励。sum(rewards) 对 rewards 列表中的奖励求和，len(rollouts) 计算轨迹的数量，二者相除得到所有轨迹的平均奖励，并将这个值作为函数的返回值。