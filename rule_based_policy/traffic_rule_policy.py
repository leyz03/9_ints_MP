import numpy as np

class TrafficPolicy:
    def __init__(self, traffic_state):
        self.phase_indices = {}
        self.call_counts = {}
        for i, _ in enumerate(traffic_state.keys()):
            phase_index_key = f'PHASE_INDEX_{i}'
            call_count_key = f'CALL_COUNT_{i}'
            self.phase_indices[phase_index_key] = 0
            self.call_counts[call_count_key] = 0

    def ft_policy(self, traffic_state, infos, action_step=2, *args, **kwargs):
        for index, sub_dict in enumerate(infos.values()):
            result = sub_dict.get('can_perform_action', False)
            if result:
                phase_index_key = f'PHASE_INDEX_{index}'
                call_count_key = f'CALL_COUNT_{index}'
                self.call_counts[call_count_key] += 1
                if self.call_counts[call_count_key] >= action_step:
                    self.phase_indices[phase_index_key] = (self.phase_indices[phase_index_key] + 1) % 4
                    self.call_counts[call_count_key] = 0

        action = {junction_id: self.phase_indices[f'PHASE_INDEX_{index}'] for index, junction_id in
                  enumerate(traffic_state.keys())}
        return action

    def actuated_policy(self, traffic_state, junction_phase_group, junction_movement_ids, infos, *args, **kwargs):
        # 首先计算每一个 phase 的最大占有率
        junction_phase_max_occ = {junction_id: {} for junction_id in traffic_state.keys()}
        for junction_id, phase_group in junction_phase_group.items():
            for phase_id, phase_movements in phase_group.items():
                if infos[junction_id]['can_perform_action']:
                    max_occ = 0
                    for movement_id in phase_movements:
                        _movement_index = junction_movement_ids[junction_id].index(movement_id)
                        max_occ = max(max_occ, traffic_state[junction_id][-1][_movement_index][0])
                    junction_phase_max_occ[junction_id][phase_id] = max_occ
                else:
                    junction_phase_max_occ[junction_id][phase_id] = 0

        # 选择最堵车的 phase 作为 index
        action = {}
        for key, sub_dict in junction_phase_max_occ.items():
            max_key = max(sub_dict, key=sub_dict.get)
            action[key] = max_key

        return action

    def webster_policy(self, traffic_state, junction_phase_group, junction_movement_ids, infos, max_occ_threshold: float = 0.3, *args, **kwargs):
        # 计算当前每个 junction 所在的 phase
        current_junction_phase = {junction_id: 0 for junction_id in traffic_state.keys()}  # 记录当前路口的 phase index
        for junction_id, junction_info in traffic_state.items():
            this_phase = junction_info[-1][:, -1]  # 这个路口此时哪些 movement 可以同行
            this_phase_movement_id = np.where(this_phase == 1)[0]  # 当前 phase 对应的 movement id
            this_phase_movement = [junction_movement_ids[junction_id][i] for i in this_phase_movement_id]  # 当前 phase 对应的 movement
            for phase_index, phase_movements in junction_phase_group[junction_id].items():
                if set(phase_movements) == set(this_phase_movement):
                    current_junction_phase[junction_id] = phase_index
                    break

        # 计算每一个 phase 的最大占有率
        junction_phase_max_occ = {junction_id: {} for junction_id in traffic_state.keys()}
        for junction_id, phase_group in junction_phase_group.items():
            for phase_id, phase_movements in phase_group.items():
                if infos[junction_id]['can_perform_action']:
                    max_occ = 0
                    for movement_id in phase_movements:
                        _movement_index = junction_movement_ids[junction_id].index(movement_id)
                        max_occ = max(max_occ, traffic_state[junction_id][-1][_movement_index][0])
                    
                    junction_phase_max_occ[junction_id][phase_id] = max_occ
                else:
                    junction_phase_max_occ[junction_id][phase_id] = 0

        # 选择最堵车的 phase 作为 index
        action = {}
        for junction_id, current_phase_index in current_junction_phase.items():
            if junction_phase_max_occ[junction_id][current_phase_index] > max_occ_threshold:
                action[junction_id] = current_phase_index
            else:
                action[junction_id] = (current_phase_index + 1) % 4

        return action

    def mp_policy(self, traffic_state, junction_phase_group, junction_movement_ids, infos, *args, **kwargs):
        # 首先计算每一个 phase 的mp之和
        junction_phase_max_occ = {junction_id: {} for junction_id in traffic_state.keys()}
            
        for junction_id, phase_group in junction_phase_group.items():
            for phase_id, phase_movements in phase_group.items():
                if infos[junction_id]['can_perform_action']:
                    max_press = 0
                    for movement_id in phase_movements:
                        _movement_index = junction_movement_ids[junction_id].index(movement_id)
                        max_press += traffic_state[junction_id][-1][_movement_index][1]
                    junction_phase_max_occ[junction_id][phase_id] = max_press
                else:
                    junction_phase_max_occ[junction_id][phase_id] = 0

        # 选择最堵车的 phase 作为 index
        action = {}
        for key, sub_dict in junction_phase_max_occ.items():
            max_key = max(sub_dict, key=sub_dict.get)
            action[key] = max_key

        return action