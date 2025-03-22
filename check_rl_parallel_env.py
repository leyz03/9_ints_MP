'''
@Author: WANG Maonan
@Date: 2023-10-30 23:01:03
@Description: 检查同时开启多个仿真环境
@LastEditTime: 2024-05-06 21:32:30
'''
import json
from loguru import logger

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
from env_utils.make_multi_tsc_env import make_parallel_env

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'), file_log_level="INFO", terminal_log_level="WARNING") # 设置终端输出的日志级别为 WARNING，表示在终端显示警告级别及更严重的日志（包括警告、错误等）

def load_environment_config(env_config_path): # 该函数用于读取指定路径下的 JSON 配置文件并将其解析为 Python 字典对象。
    env_config_path = path_convert(f'./configs/env_configs/{env_config_path}') # 是配置文件的路径，调用 path_convert 将其转换为绝对路径。
    with open(env_config_path, 'r') as file:
        config = json.load(file) # 读取配置文件并使用 json.load() 将其内容加载为 Python 字典。
    return config

if __name__ == '__main__':
    sumo_cfg = path_convert("./sumo_nets/demo/env/demo.sumocfg")
    net_file = path_convert("./sumo_nets/demo/env/demo.net.xml")
    log_path = path_convert('./log/demo')
    env_config = load_environment_config("demo.json")
    road_ids = env_config['road_ids']
    action_space = {
      "E": 4,
      "B": 1,
      "D": 1,
      "F": 1,
      "H": 1
    }
    tsc_env = make_parallel_env( # 函数会创建一个并行仿真环境对象 tsc_env，并根据上述配置来启动多个仿真环境
        num_envs=6,
        sumo_cfg=sumo_cfg,
        net_file=net_file,
        num_seconds=1300,
        agent_tls_ids=['E'],
        non_agent_tls_ids=['B', 'D', 'F', 'H'],
        road_ids=road_ids,
        cell_length=50,
        action_space=action_space,
        use_gui=False,
        log_file=log_path
    )
    rollouts = tsc_env.rollout(1_000, break_when_any_done=False) # 这一行调用 tsc_env 对象的 rollout 方法，开始执行仿真并生成轨迹数据。具体参数如下：1_000：表示执行 1000 次仿真步骤。break_when_any_done=False：设置为 False 表示即使某个环境中的仿真结束，也继续执行其他环境中的仿真，直到 1000 步仿真都执行完
    for r in rollouts:
        logger.info(f'RL: {r}') # 这一段代码遍历 rollouts 中的每个结果 r，并通过 logger.info 记录每个结果的日志。f'RL: {r}' 会把每个仿真结果 r 输出为日志，日志级别为 INFO，并且记录的信息包括每次仿真步骤的详细内容。