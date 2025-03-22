'''
@Author: WANG Maonan
@Date: 2023-09-01 13:45:26
@Description: 给场景生成路网
@LastEditTime: 2024-04-15 23:48:04
'''
import numpy as np
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
from tshub.sumo_tools.generate_routes import generate_route

# 初始化日志
current_file_path = get_abs_path(__file__)
set_logger(current_file_path('./'))

# 开启仿真 --> 指定 net 文件
sumo_net = current_file_path("./env/demo.net.xml")

# 指定要生成的路口 id 和探测器保存的位置
generate_route(
    sumo_net=sumo_net,
    interval=[5,5,5,5], 
    edge_flow_per_minute={
        'A3': [np.random.randint(5, 10) for _ in range(4)],
        'C3': [np.random.randint(5, 10) for _ in range(4)],

        'W7': [np.random.randint(5, 10) for _ in range(4)],
        'E3': [np.random.randint(5, 10) for _ in range(4)],

        'G3': [np.random.randint(5, 10) for _ in range(4)],
        'I1': [np.random.randint(5, 10) for _ in range(4)],
        
        'A1': [np.random.randint(0, 5) for _ in range(4)],
        'G2': [np.random.randint(0, 5) for _ in range(4)],

        'N1': [np.random.randint(0, 5) for _ in range(4)],
        'S5': [np.random.randint(0, 5) for _ in range(4)],

        'C1': [np.random.randint(0, 5) for _ in range(4)],
        'I3': [np.random.randint(0, 5) for _ in range(4)],
    }, # 每分钟每个 edge 有多少车
    edge_turndef={
        'A3__N7': [0.5]*4,
        'N8__A4': [0.5]*4,
        'N7__N4': [0.5]*4,
        'N3__N8': [0.5]*4,
        'N4__C4': [0.5]*4,
        'C3__N3': [0.5]*4,

        'W7__W4': [0.5]*4,
        'W3__W8': [0.5]*4,
        'W4__E7': [0.5]*4,
        'E8__W3': [0.5]*4,
        'E7__E4': [0.5]*4,
        'E3__E8': [0.5]*4,

        'G3__S7': [0.5]*4,
        'S8__G4': [0.5]*4,
        'S7__S4': [0.5]*4,
        'S3__S8': [0.5]*4,
        'S4__I2': [0.5]*4,
        'I1__S3': [0.5]*4,

        'A1__W1': [0.5]*4,
        'W2__A2': [0.5]*4,
        'W1__W6': [0.5]*4,
        'W5__W2': [0.5]*4,
        'W6__G2': [0.5]*4,
        'G1__W5': [0.5]*4,

        'N1__N6': [0.5]*4,
        'N5__N2': [0.5]*4,
        'N6__S1': [0.5]*4,
        'S2__N5': [0.5]*4,
        'S1__S6': [0.5]*4,
        'S5__S2': [0.5]*4,

        'C1__E1': [0.5]*4,
        'E2__C2': [0.5]*4,
        'E1__E6': [0.5]*4,
        'E5__E2': [0.5]*4,
        'E6__I4': [0.5]*4,
        'I3__E5': [0.5]*4,
    },
    veh_type={
        'background_1': {'color':'26, 188, 156', 'probability':0.3},
        'background_2': {'color':'155, 89, 182', 'speed':15, 'probability':0.7},
    },
    output_trip=current_file_path('./testflow.trip.xml'),
    output_turndef=current_file_path('./testflow.turndefs.xml'),
    output_route=current_file_path('./demo.rou.xml'),
    interpolate_flow=False,
    interpolate_turndef=False,
)