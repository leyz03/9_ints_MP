'''
@Author: WANG Maonan
@Date: 2023-10-29 22:46:25
@Description: 使用 MAPPO 算法进行训练, nohup python train_mappo.py > train_log.out 2>&1 &
LastEditTime: 2024-09-20 12:59:32
'''
import os
import json
import tqdm
import time
import torch

from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.record.loggers import generate_exp_name, get_logger

# 环境相关
from env_utils.make_multi_tsc_env import make_parallel_env

# actor & critic
from train_utils.make_log import log_training
from train_utils.make_actor_module import policy_module
from train_utils.make_critic_module import critic_module

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'), file_log_level="INFO", terminal_log_level="WARNING")


def load_environment_config(env_config_path):
    env_config_path = path_convert(f'./configs/env_configs/{env_config_path}')
    with open(env_config_path, 'r') as file:
        config = json.load(file)
    return config

def train(exp_config_path:str):  # noqa: F821
    # 读取实验配置文件
    with open(exp_config_path, 'r') as file:
        exp_config = json.load(file)
    env_config = load_environment_config(exp_config['environment_name'])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # 超参数设置
    num_envs = 8 # 同时开启的环境个数
    n_iters = 500 # 控制训练的时间长度

    exp_name = exp_config["experiment_name"]
    model_name = exp_config["model_name"]

    action_space = env_config["action_space"]
    road_ids = env_config['road_ids']
    agent_tls_ids = env_config["agent_ids"]
    non_agent_tls_ids = env_config["non_agent_ids"]
    num_seconds = env_config["simulation_time"] # 仿真时间
    frames_per_batch = num_envs*env_config["simulation_steps"]*2 # 差不多是 2 轮游戏 # 每批次的帧数
    memory_size = frames_per_batch # 回放缓冲区的大小
    collector_total_frames = frames_per_batch*n_iters
    minibatch_size = num_envs*(env_config["simulation_steps"]//2) # multi-agent 这个参数稍微大一些, 至少包含一半的数据
    num_epochs = 10 # optimization steps per batch of data collected, 10-15 即可

    # Create Env
    sumo_cfg = path_convert(env_config["sumocfg"])
    net_file = path_convert(env_config["sumonet"])
    log_path = path_convert(f'./log/{exp_name}/')
    model_path = path_convert(f'./mappo_models/{exp_name}/') # 模型存储的路径

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # 训练使用的环境
    env = make_parallel_env(
        num_envs=num_envs,
        sumo_cfg=sumo_cfg,
        net_file=net_file,
        num_seconds=num_seconds,
        agent_tls_ids=agent_tls_ids,
        non_agent_tls_ids=non_agent_tls_ids,
        action_space=action_space,
        road_ids=road_ids,
        cell_length=100,
        use_gui=False,
        log_file=log_path,
        device=device
    )

    # #################
    # Policy and Critic
    # #################
    # 根据环境的动作规格创建策略模块（policy）和价值网络模块（value_module）。
    action_spec = env.action_spec 
    policy_gen = policy_module(model_name, action_spec, device)
    policy_agent = policy_gen.make_policy_module()
    
    value_gen = critic_module(model_name, device)
    value_module = value_gen.make_critic_module()
    
    # Data Collector
    collector_agent = SyncDataCollector( # 用于从环境中收集数据。这里设置了每次收集的数据量和总的帧数。
        env,
        policy_agent,
        device=device,
        storing_device=device,
        frames_per_batch=frames_per_batch,
        total_frames=collector_total_frames,
    )

    # Reply Buffer
    replay_buffer = TensorDictReplayBuffer( # 用于存储和采样数据的回放缓冲区。使用懒加载存储，并设定每次从缓冲区中采样的数据量。
        storage=LazyTensorStorage(memory_size, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,
    )

    # Loss
    loss_module = ClipPPOLoss( # 使用ClipPPOLoss来定义PPO算法的损失函数，设置clip_epsilon来防止策略更新过大，entropy_coef用于调整策略的熵（即探索与利用的权衡）。
        actor=policy_agent,
        critic=value_module,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        normalize_advantage=True,
    )

    loss_module.set_keys(
        reward=('agents', 'reward'),
        action=('agents', 'action'),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )
    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=0.9, lmbda=0.9
    )
    optim = torch.optim.Adam(loss_module.parameters(), 5e-5) # 定义Adam优化器，用来优化损失函数的参数。

    # Create Logger
    # 创建日志记录器，用于记录训练过程中的指标数据。
    exp_name = generate_exp_name("MAPPO", "TSC")
    logger = get_logger(
        'tensorboard', 
        logger_name="mappo_tensorboard", 
        experiment_name=exp_name
    )

    # Start Training
    # 使用tqdm库创建一个进度条，记录训练过程中已收集的帧数。
    pbar = tqdm.tqdm(total=collector_total_frames)
    total_time = 0
    total_frames = 0 # 当前训练的 frames
    sampling_start = time.time()

    # 每次从数据收集器中获取新的数据，并计算采样时间。使用no_grad()进行无梯度计算。
    for i, tensordict_data in enumerate(collector_agent):
        pbar.update(tensordict_data.numel())

        sampling_time = time.time() - sampling_start

        with torch.no_grad():
            loss_module.value_estimator(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )
        
        # 更新已收集的总帧数，并将数据扩展到回放缓冲区中。
        current_frames = tensordict_data.numel()
        total_frames += current_frames
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        # 执行训练循环，包括梯度计算、反向传播、优化更新等操作。
        training_tds = []
        training_start = time.time()
        for _ in range(num_epochs): # optimization steps per batch of data collected
            for _ in range(frames_per_batch // minibatch_size):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                training_tds.append(loss_vals.detach())

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), max_norm=40
                )
                training_tds[-1].set("grad_norm", total_norm.mean())

                optim.step()
                optim.zero_grad()
        # 更新策略权重，并记录当前的训练日志。
        collector_agent.update_policy_weights_()

        training_time = time.time() - training_start
        iteration_time = sampling_time + training_time
        total_time += iteration_time
        training_tds = torch.stack(training_tds)

        # More logs
        log_training(
            logger,
            training_tds,
            tensordict_data,
            sampling_time,
            training_time,
            total_time,
            i,
            current_frames,
            total_frames,
            step=i,
        )

        # 保存模型
        # 每5个迭代保存一次策略网络（actor）和价值网络（critic）的模型。
        if i % 5 == 0:
            policy_gen.save_model(os.path.join(model_path,f'actor.pkl'))
            value_gen.save_model(os.path.join(model_path,f'critic.pkl'))

        sampling_start = time.time() # 重置采样时间的开始时间。

if __name__ == "__main__": # 主函数，通过不同的场景和模型名称来调用train函数进行训练。
    scenario_names = ["demo"] # 场景名称
    model_names = [
        "1_occmlp", "2_allcnn"
    ] # 模型的名称
    for scenario_name in scenario_names:
        for model_name in model_names:
            train(exp_config_path=path_convert(f'./configs/exp_configs/{scenario_name}/{model_name}.json'))

#     model_names = [
#         "1_occmlp", "2_allcnn", "3_occmlp_noLaynorm", 
#         "4_occmlp_noAgentID", "5_occmlp_raw", "6_occmlp_rawLaynorm", 
#         "7_occmlp_noLaynorm_noAgentID", "8_occmlp_rawAgentID"
#     ] # 模型的名称

