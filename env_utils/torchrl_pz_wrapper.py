# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations # 这行代码用于导入未来的注解特性，允许在函数中使用尚未定义的类型。

import copy
import importlib
import warnings
from typing import Dict, List, Tuple, Union

import packaging
import torch
from tensordict import TensorDictBase

import numpy as np

from torchrl.data.tensor_specs import (
    # CompositeSpec, # 复合张量规格。
    Composite,
    # DiscreteTensorSpec, # 离散张量规格。
    Categorical,
    OneHotDiscreteTensorSpec, # 一热编码的离散张量规格。
    # UnboundedContinuousTensorSpec, # 无界连续张量规格。
    Unbounded
)
from torchrl.envs.common import _EnvWrapper # 从torchrl库导入环境包装器类_EnvWrapper，用于包装环境。
from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform, set_gym_backend # _gym_to_torchrl_spec_transform: 将gym环境的规格转换为torchrl规格。# set_gym_backend: 设置gym的后端。
from torchrl.envs.utils import _classproperty, check_marl_grouping, MarlGroupMapType # _classproperty: 用于定义类属性的装饰器。# check_marl_grouping: 检查多智能体强化学习分组的函数。# MarlGroupMapType: 多智能体强化学习分组映射类型。

_has_pettingzoo = importlib.util.find_spec("pettingzoo") is not None # 检查pettingzoo库是否已安装，如果安装，则_has_pettingzoo为True，否则为False。


def _get_envs(): # 定义一个名为_get_envs的函数，用于获取环境列表。
    if not _has_pettingzoo:
        raise ImportError("PettingZoo is not installed in your virtual environment.")
    try:
        from pettingzoo.utils.all_modules import all_environments # 尝试从pettingzoo的工具模块导入all_environments，这是一个包含所有环境的字典。
    except ModuleNotFoundError as err: # 如果导入失败，捕获ModuleNotFoundError异常并执行以下代码。
        warnings.warn( # 发出警告，说明无法加载所有模块，并打印错误信息，随后将尝试单独加载模块。
            f"PettingZoo failed to load all modules with error message {err}, trying to load individual modules."
        )
        all_environments = _load_available_envs() # 调用_load_available_envs函数来加载可用的环境。

    return list(all_environments.keys()) # 返回所有环境的名称列表。


def _load_available_envs() -> Dict: # 定义一个名为_load_available_envs的函数，返回一个字典，包含所有可用的环境。
    all_environments = {}
    try:
        from pettingzoo.mpe.all_modules import mpe_environments # 尝试从pettingzoo.mpe模块导入mpe_environments，这是MPE（多智能体环境）相关的环境。

        all_environments.update(mpe_environments) # 将MPE环境更新到all_environments字典中。
    except ModuleNotFoundError as err: # 如果导入失败，发出警告并打印错误信息。
        warnings.warn(f"MPE environments failed to load with error message {err}.")
    try:
        from pettingzoo.sisl.all_modules import sisl_environments # 同样，尝试导入SISL（单智能体环境）相关的环境。

        all_environments.update(sisl_environments)
    except ModuleNotFoundError as err:
        warnings.warn(f"SISL environments failed to load with error message {err}.")
    try:
        from pettingzoo.classic.all_modules import classic_environments # 尝试导入经典环境相关的环境。

        all_environments.update(classic_environments)
    except ModuleNotFoundError as err:
        warnings.warn(f"Classic environments failed to load with error message {err}.")
    try:
        from pettingzoo.atari.all_modules import atari_environments # 尝试导入Atari环境相关的环境。

        all_environments.update(atari_environments)
    except ModuleNotFoundError as err:
        warnings.warn(f"Atari environments failed to load with error message {err}.")
    try:
        from pettingzoo.butterfly.all_modules import butterfly_environments # 尝试导入Butterfly环境相关的环境。

        all_environments.update(butterfly_environments)
    except ModuleNotFoundError as err:
        warnings.warn(
            f"Butterfly environments failed to load with error message {err}."
        )
    return all_environments


class PettingZooWrapper(_EnvWrapper): # 定义一个名为 PettingZooWrapper 的类，它继承自 _EnvWrapper。这个类将为 PettingZoo 环境提供一个通用的封装器。
    """PettingZoo environment wrapper.

    To install petting zoo follow the guide `here <https://github.com/Farama-Foundation/PettingZoo#installation>__`.

    This class is a general torchrl wrapper for all PettingZoo environments.
    It is able to wrap both ``pettingzoo.AECEnv`` and ``pettingzoo.ParallelEnv``. # 该类能够封装两种类型的环境：AECEnv（行动-环境-行动）和 ParallelEnv（并行环境）。

    Let's see how more in details:

    In wrapped ``pettingzoo.ParallelEnv`` all agents will step at each environment step. # 在包装的 ParallelEnv 中，所有代理（agents）会在每一步环境中同时行动。
    If the number of agents during the task varies, please set ``use_mask=True``. # 如果在任务中代理的数量会变化，建议将 use_mask 设置为 True。
    ``"mask"`` will be provided
    as an output in each group and should be used to mask out dead agents. # 将会在每个组的输出中提供一个 "mask"，用于屏蔽死去的代理。
    The environment will be reset as soon as one agent is done (unless ``done_on_any`` is ``False``). # 当任一代理完成任务时，环境会被重置（除非 done_on_any 设置为 False）。

    In wrapped ``pettingzoo.AECEnv``, at each step only one agent will act. # 在包装的 AECEnv 中，每一步只有一个代理会进行行动。
    For this reason, it is compulsory to set ``use_mask=True`` for this type of environment. # 因此，对于这种类型的环境，必须将 use_mask 设置为 True。
    ``"mask"`` will be provided as an output for each group and can be used to mask out non-acting agents. # 每个组的输出中将提供 "mask"，用于屏蔽未行动的代理。
    The environment will be reset only when all agents are done (unless ``done_on_any`` is ``True``). # 环境仅在所有代理完成时重置（除非 done_on_any 设置为 True）。

    If there are any unavailable actions for an agent, # 如果代理有任何不可用的动作，
    the environment will also automatically update the mask of its ``action_spec`` and output an ``"action_mask"`` # 环境会自动更新其 action_spec 的掩码，并输出一个 "action_mask"，
    for each group to reflect the latest available actions. This should be passed to a masked distribution during
    training. # 对于每个组，反映最新可用动作。在训练时，应将其传递给掩码分布。

    As a feature of torchrl multiagent, you are able to control the grouping of agents in your environment. # 作为 TorchRL 多代理的一个特性，您可以控制环境中代理的分组。
    You can group agents together (stacking their tensors) to leverage vectorization when passing them through the same # 您可以将代理分组（堆叠它们的张量），以在将它们传递到同一神经网络时利用向量化。
    neural network. You can split agents in different groups where they are heterogenous or should be processed by # 您可以将代理拆分为不同的组，以便它们是异构的或应由不同的神经网络处理。
    different neural networks. To group, you just need to pass a ``group_map`` at env constructiuon time. # 要进行分组，只需在环境构造时传递一个 group_map。

    By default, agents in pettingzoo will be grouped by name. # 默认情况下，PettingZoo 中的代理将按名称分组。
    For example, with agents ``["agent_0","agent_1","agent_2","adversary_0"]``, the tensordicts will look like:

        >>> print(env.rand_action(env.reset())) # 这行代码示例展示了如何重置环境并随机选择动作。
        TensorDict(
            fields={
                agent: TensorDict(
                    fields={ # fields 字典包含每个代理的张量信息。
                        action: Tensor(shape=torch.Size([3, 9]), device=cpu, dtype=torch.int64, is_shared=False),
                        action_mask: Tensor(shape=torch.Size([3, 9]), device=cpu, dtype=torch.bool, is_shared=False),
                        done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([3, 3, 3, 2]), device=cpu, dtype=torch.int8, is_shared=False),
                        terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([3]))},
                adversary: TensorDict( # 对于对手（adversary），也会定义一个 TensorDict。
                    fields={
                        action: Tensor(shape=torch.Size([1, 9]), device=cpu, dtype=torch.int64, is_shared=False),
                        action_mask: Tensor(shape=torch.Size([1, 9]), device=cpu, dtype=torch.bool, is_shared=False),
                        done: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([1, 3, 3, 2]), device=cpu, dtype=torch.int8, is_shared=False),
                        terminated: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([1]))},
            batch_size=torch.Size([]))
        >>> print(env.group_map) # 这行代码示例展示了如何打印环境的代理分组映射。
        {"agent": ["agent_0", "agent_1", "agent_2"], "adversary": ["adversary_0"]} # 输出的代理分组映射示例。

    Otherwise, a group map can be specified or selected from some premade options. # Group Map: 指定或选择一个分组映射，这可以帮助管理多个代理的状态和动作。通过使用预先定义的选项，可以简化配置过程。
    See :class:`torchrl.envs.utils.MarlGroupMapType` for more info. # MarlGroupMapType.ONE_GROUP_PER_AGENT: 这是一个具体的选项，表示每个代理都有自己的数据结构（称为tensor dict），这类似于PettingZoo库中的并行API。这意味着每个代理都可以独立处理其输入和输出，保持数据的隔离。
    For example, you can provide ``MarlGroupMapType.ONE_GROUP_PER_AGENT``, telling that each agent should
    have its own tensordict (similar to the pettingzoo parallel API).

    Grouping is useful for leveraging vectorisation among agents whose data goes through the same
    neural network. # 向量化（Vectorization）: 在训练过程中，将多个代理的数据集中到一个神经网络中处理，可以提高效率。通过分组，多个代理的状态可以同时传递给网络，这样可以加速训练过程。

    Args:
        env (``pettingzoo.utils.env.ParallelEnv`` or ``pettingzoo.utils.env.AECEnv``): the pettingzoo environment to wrap.
        return_state (bool, optional): whether to return the global state from pettingzoo
            (not available in all environments). Defaults to ``False``.
        group_map (MarlGroupMapType or Dict[str, List[str]]], optional): how to group agents in tensordicts for
            input/output. By default, agents will be grouped by their name. Otherwise, a group map can be specified
            or selected from some premade options. See :class:`torchrl.envs.utils.MarlGroupMapType` for more info.
        use_mask (bool, optional): whether the environment should output a ``"mask"``. This is compulsory in
            wrapped ``pettingzoo.AECEnv`` to mask out non-acting agents and should be also used
            for ``pettingzoo.ParallelEnv`` when the number of agents can vary. Defaults to ``False``.
        categorical_actions (bool, optional): if the enviornments actions are discrete, whether to transform
            them to categorical or one-hot.
        seed (int, optional): the seed. Defaults to ``None``.
        done_on_any (bool, optional): whether the environment's done keys are set by aggregating the agent keys
            using ``any()`` (when ``True``) or ``all()`` (when ``False``). Default (``None``) is to use ``any()`` for
            parallel environments and ``all()`` for AEC ones.

    Examples:
        >>> # Parallel env
        >>> from torchrl.envs.libs.pettingzoo import PettingZooWrapper
        >>> from pettingzoo.butterfly import pistonball_v6
        >>> kwargs = {"n_pistons": 21, "continuous": True}
        >>> env = PettingZooWrapper(
        ...     env=pistonball_v6.parallel_env(**kwargs),
        ...     return_state=True,
        ...     group_map=None, # Use default for parallel (all pistons grouped together)
        ... )
        >>> print(env.group_map)
        ... {'piston': ['piston_0', 'piston_1', ..., 'piston_20']}
        >>> env.rollout(10)
        >>> # AEC env
        >>> from pettingzoo.classic import tictactoe_v3
        >>> from torchrl.envs.libs.pettingzoo import PettingZooWrapper
        >>> from torchrl.envs.utils import MarlGroupMapType
        >>> env = PettingZooWrapper(
        ...     env=tictactoe_v3.env(),
        ...     use_mask=True, # Must use it since one player plays at a time
        ...     group_map=None # # Use default for AEC (one group per player)
        ... )
        >>> print(env.group_map)
        ... {'player_1': ['player_1'], 'player_2': ['player_2']}
        >>> env.rollout(10)
    """

    git_url = "https://github.com/Farama-Foundation/PettingZoo"
    libname = "pettingzoo"

    @_classproperty
    def available_envs(cls): # 定义了一个类属性available_envs，用来返回可用的环境列表。@_classproperty是一个装饰器，表示这是一个类属性而不是实例属性。
        if not _has_pettingzoo:
            return []
        return list(_get_envs()) # 如果_has_pettingzoo为False（表示PettingZoo库未安装），则返回一个空列表；否则，调用_get_envs()函数获取可用环境的列表。
 
    def __init__( # 构造函数，用于初始化PettingZooWrapper对象。它接受多个参数：
        self,
        env: Union[ # env: 要包装的PettingZoo环境，可以是并行环境或AEC环境。
            "pettingzoo.utils.env.ParallelEnv",  # noqa: F821 
            "pettingzoo.utils.env.AECEnv",  # noqa: F821
        ] = None,
        return_state: bool = False, # 是否返回全局状态。
        group_map: MarlGroupMapType | Dict[str, List[str]] | None = None, # 如何在输入/输出中对智能体进行分组
        use_mask: bool = False, # 是否输出“mask”，用于在非动作智能体中进行掩蔽
        categorical_actions: bool = True, # 如果动作是离散的，是否将其转换为分类或独热编码。
        seed: int | None = None,
        done_on_any: bool | None = None, # done_on_any: 环境的结束条件是按any()还是all()聚合。
        **kwargs,
    ):
        if env is not None: # 如果env不为None，将其加入到kwargs字典中，方便后续传递给父类。
            kwargs["env"] = env

        self.group_map = group_map
        self.return_state = return_state
        self.seed = seed
        self.use_mask = use_mask
        self.categorical_actions = categorical_actions
        self.done_on_any = done_on_any

        super().__init__(**kwargs, allow_done_after_reset=True) # 调用父类的构造函数，传入kwargs，并设置allow_done_after_reset为True，允许在重置后继续完成操作。

    def _get_default_group_map(self, agent_names: List[str]): # 定义一个私有方法_get_default_group_map，用于获取默认的分组映射。
        # This function performs the default grouping in pettingzoo
        if not self.parallel: # 检查当前环境是否为并行环境。
            # In AEC envs we will have one group per agent by default
            group_map = MarlGroupMapType.ONE_GROUP_PER_AGENT.get_group_map(agent_names) # 如果不是并行环境，则为每个智能体创建一个组，使用MarlGroupMapType.ONE_GROUP_PER_AGENT获取默认的分组映射。
        else:
            # In parallel envs, by default
            # Agents with names "str_int" will be grouped in group name "str"
            group_map = {} # 如果是并行环境，初始化一个空的分组映射字典。
            for agent_name in agent_names: # 遍历每个智能体的名称。
                # See if the agent follows the convention "name_int"
                follows_convention = True # 初始化一个标志，表示智能体名称是否遵循命名约定，并将智能体名称按下划线分割为列表。
                agent_name_split = agent_name.split("_")
                if len(agent_name_split) == 1: # 如果分割后的列表长度为1，则说明不遵循约定，标记为False。
                    follows_convention = False
                try: # 尝试将名称最后一部分转换为整数，如果失败，标记为不遵循约定。
                    int(agent_name_split[-1])
                except ValueError:
                    follows_convention = False

                # If not, just put it in a single group
                if not follows_convention: # 如果不遵循约定，将智能体自身作为唯一组放入分组映射中。
                    group_map[agent_name] = [agent_name]
                # Otherwise, group it with other agents that follow the same convention
                else: # 如果遵循约定，将名称的前面部分连接成组名。
                    group_name = "_".join(agent_name_split[:-1])
                    if group_name in group_map: # 检查该组名是否已经存在于分组映射中。如果存在，将智能体添加到该组；否则，创建新组。
                        group_map[group_name].append(agent_name)
                    else:
                        group_map[group_name] = [agent_name]

        return group_map # 返回最终的分组映射。

    @property
    def lib(self): # 这段代码定义了一个属性方法 lib，它导入 pettingzoo 库并返回这个库的引用。这个方法可以通过类的实例直接访问。
        import pettingzoo

        return pettingzoo

    def _build_env( # 定义了一个名为 _build_env 的方法，接收一个参数 env，该参数可以是 ParallelEnv 或 AECEnv 类型的对象。Union 用于表明这个参数可以是多种类型。
        self,
        env: Union[
            "pettingzoo.utils.env.ParallelEnv",  # noqa: F821
            "pettingzoo.utils.env.AECEnv",  # noqa: F821
        ],
    ):
        import pettingzoo # 再次导入 pettingzoo 库，以便在方法内部使用。

        if packaging.version.parse(pettingzoo.__version__).base_version != "1.24.3": # 检查当前安装的 pettingzoo 库的版本是否为 1.24.3。如果不是，则发出一个警告，提示用户如果遇到兼容性问题可以在 TorchRL 的 GitHub 上报告。
            warnings.warn(
                "PettingZoo in TorchRL is tested using version == 1.24.3 , "
                "If you are using a different version and are experiencing compatibility issues,"
                "please raise an issue in the TorchRL github."
            )

        self.parallel = isinstance(env, pettingzoo.utils.env.ParallelEnv) # 判断传入的 env 是否是 ParallelEnv 类型，并将结果赋值给实例变量 self.parallel。
        if not self.parallel and not self.use_mask: # 如果 env 不是并行环境（即为 AECEnv），并且 use_mask 没有被设置为 True，则抛出一个错误，提示需要设置 use_mask=True。
            raise ValueError("For AEC environments you need to set use_mask=True")
        if len(self.batch_size): # 检查 batch_size 的长度，如果不为零，则抛出一个运行时错误，表明 PettingZoo 不支持自定义的 batch_size。
            raise RuntimeError(
                f"PettingZoo does not support custom batch_size {self.batch_size}."
            )

        return env

    @set_gym_backend("gymnasium")
    def _make_specs( # 定义 _make_specs 方法，并使用装饰器 @set_gym_backend("gymnasium")。这个方法也接收一个环境对象 env。
        self,
        env: Union[
            "pettingzoo.utils.env.ParallelEnv",  # noqa: F821
            "pettingzoo.utils.env.AECEnv",  # noqa: F821
        ],
    ) -> None:
        # Set default for done on any or all
        if self.done_on_any is None: # 如果 done_on_any 还没有被设置，则将其赋值为 self.parallel 的值。
            self.done_on_any = self.parallel

        # Create and check group map
        if self.group_map is None: # 如果 group_map 还未被定义，则调用 _get_default_group_map 方法，并传入 possible_agents 以获取默认的组映射。
            self.group_map = self._get_default_group_map(self.possible_agents)
        elif isinstance(self.group_map, MarlGroupMapType): # 如果 group_map 是 MarlGroupMapType 类型的实例，则调用其 get_group_map 方法，传入 possible_agents 以获取具体的组映射。
            self.group_map = self.group_map.get_group_map(self.possible_agents)
        check_marl_grouping(self.group_map, self.possible_agents) # 调用 check_marl_grouping 方法，验证当前的组映射是否与可能的代理（agents）匹配。
        self.has_action_mask = {group: False for group in self.group_map.keys()} # 创建一个字典 has_action_mask，其键为 group_map 中的每个组，值都初始化为 False。

        # action_spec = CompositeSpec() # 创建四个 CompositeSpec 对象：action_spec、observation_spec、reward_spec 和 done_spec。done_spec 特别定义了三个子项（done、terminated 和 truncated），每个子项都是 DiscreteTensorSpec 类型，表示状态的离散值，并指定了形状、数据类型和设备。
        action_spec = Composite()
        observation_spec = Composite()
        reward_spec = Composite()
        done_spec = Composite(
            {
                "done": Categorical(
                    n=2,
                    shape=torch.Size((1,)),
                    dtype=torch.bool,
                    device=self.device,
                ),
                "terminated": Categorical(
                    n=2,
                    shape=torch.Size((1,)),
                    dtype=torch.bool,
                    device=self.device,
                ),
                "truncated": Categorical(
                    n=2,
                    shape=torch.Size((1,)),
                    dtype=torch.bool,
                    device=self.device,
                ),
            },
        )
        for group, agents in self.group_map.items(): # 遍历 group_map 中的每个组及其对应的代理，调用 _make_group_specs 方法以创建组的观察、行动、奖励和完成规格。
            (
                group_observation_spec,
                group_action_spec,
                group_reward_spec,
                group_done_spec,
            ) = self._make_group_specs(group_name=group, agent_names=agents)
            action_spec[group] = group_action_spec # 将每个组的规格分别赋值给相应的 action_spec、observation_spec、reward_spec 和 done_spec。
            observation_spec[group] = group_observation_spec
            reward_spec[group] = group_reward_spec
            done_spec[group] = group_done_spec

        self.action_spec = action_spec
        self.observation_spec = observation_spec
        self.reward_spec = reward_spec
        self.done_spec = done_spec # 最后，将创建的规格赋值给实例的属性，以便后续使用。

    def _make_group_specs(self, group_name: str, agent_names: List[str]):
        n_agents = len(agent_names) # 计算代理的数量，即 agent_names 列表的长度，并将其存储在 n_agents 变量中。
        action_specs = [] # 初始化两个空列表：action_specs 用于存储每个代理的动作规范，observation_specs 用于存储每个代理的观察规范。
        observation_specs = []
        for agent in agent_names:
            action_specs.append( # 对于每个代理，调用 self.action_space(agent) 获取该代理的动作空间，并使用 _gym_to_torchrl_spec_transform 方法将其转换为 TorchRL 规范，然后将其以字典形式存入 CompositeSpec 中，最后将这个规范添加到 action_specs 列表中。
                Composite(
                    {
                        "action": _gym_to_torchrl_spec_transform(
                            self.action_space(agent),
                            remap_state_to_observation=False,
                            categorical_action_encoding=self.categorical_actions,
                            device=self.device,
                        )
                    },
                )
            )
            observation_specs.append( # 同样地，对于每个代理，调用 self.observation_space(agent) 获取观察空间，使用 _gym_to_torchrl_spec_transform 转换，并以字典形式存入 CompositeSpec 中，最后将其添加到 observation_specs 列表中。
                Composite(
                    {
                        "observation": _gym_to_torchrl_spec_transform(
                            self.observation_space(agent),
                            remap_state_to_observation=False,
                            device=self.device,
                        )
                    }
                )
            )
        group_action_spec = torch.stack(action_specs, dim=0) # 将 action_specs 列表中的所有动作规范沿第0维（即新创建的维度）堆叠成一个张量，并将结果存储在 group_action_spec 中。
        group_observation_spec = torch.stack(observation_specs, dim=0) # 同样地，将 observation_specs 列表中的所有观察规范沿第0维堆叠，结果存储在 group_observation_spec 中。

        # Sometimes the observation spec contains an action mask.
        # Or sometimes the info spec contains an action mask.
        # We uniform this by removing it from both places and optionally set it in a standard location.
        group_observation_inner_spec = group_observation_spec["observation"]
        if ( # 检查 group_observation_inner_spec 是否是 CompositeSpec 类型，并且它的键中是否包含 "action_mask"。
            isinstance(group_observation_inner_spec, Composite)
            and "action_mask" in group_observation_inner_spec.keys()
        ):
            self.has_action_mask[group_name] = True # TODO, 这里用到了 action mask # 如果条件为真，则将 self.has_action_mask 字典中与 group_name 相关的值设为 True，表示该组使用了动作掩码。
            del group_observation_inner_spec["action_mask"]
            group_observation_spec["action_mask"] = Categorical( # 将一个新的动作掩码添加到 group_observation_spec 中。这个掩码是一个 DiscreteTensorSpec，其形状根据是否使用分类动作（self.categorical_actions）进行不同处理。
                n=2,
                shape=group_action_spec["action"].shape
                if not self.categorical_actions
                else (
                    *group_action_spec["action"].shape,
                    group_action_spec["action"].space.n,
                ),
                dtype=torch.bool,
                device=self.device,
            )

        if self.use_mask: # 检查是否启用掩码功能。
            group_observation_spec["mask"] = Categorical( # 如果启用掩码，将一个新的掩码添加到 group_observation_spec 中，形状为 (n_agents,)，类型为布尔型。
                n=2,
                shape=torch.Size((n_agents,)),
                dtype=torch.bool,
                device=self.device,
            )

        group_reward_spec = Composite( # 创建一个 CompositeSpec 用于定义奖励规范，其中包含一个名为 "reward" 的无界连续张量，形状为 (n_agents, 1)，并将其存储在 group_reward_spec 中。
            {
                "reward": Unbounded(
                    shape=torch.Size((n_agents, 1)),
                    device=self.device,
                    dtype=torch.float32,
                )
            },
            shape=torch.Size((n_agents,)),
        )
        group_done_spec = Composite( # 创建一个 CompositeSpec 用于定义完成状态规范，包含三个布尔型的离散张量："done"、"terminated" 和 "truncated"，形状均为 (n_agents, 1)，将其存储在 group_done_spec 中
            {
                "done": Categorical(
                    n=2,
                    shape=torch.Size((n_agents, 1)),
                    dtype=torch.bool,
                    device=self.device,
                ),
                "terminated": Categorical(
                    n=2,
                    shape=torch.Size((n_agents, 1)),
                    dtype=torch.bool,
                    device=self.device,
                ),
                "truncated": Categorical(
                    n=2,
                    shape=torch.Size((n_agents, 1)),
                    dtype=torch.bool,
                    device=self.device,
                ),
            },
            shape=torch.Size((n_agents,)),
        )
        return (
            group_observation_spec,
            group_action_spec,
            group_reward_spec,
            group_done_spec,
        )

    def _check_kwargs(self, kwargs: Dict): # 这是一个检查传入参数的私有方法，kwargs 是一个字典，用于接收环境相关的参数。
        import pettingzoo

        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"] # 从 kwargs 中提取 env 变量。
        if not isinstance( # 检查 env 的类型是否是 ParallelEnv 或 AECEnv 中的任意一种。如果不是，则抛出 TypeError。
            env, (pettingzoo.utils.env.ParallelEnv, pettingzoo.utils.env.AECEnv)
        ):
            raise TypeError("env is not of type expected.")

    def _init_env(self):
        # Add info # 根据 self.parallel 的值决定使用并行环境还是 AEC（交替执行环境），并调用相应的重置方法，返回观察和信息字典。
        if self.parallel:
            _, info_dict = self._reset_parallel(seed=self.seed)
        else:
            _, info_dict = self._reset_aec(seed=self.seed)

        for group, agents in self.group_map.items(): # 遍历 self.group_map 中的每个组和对应的智能体。
            info_specs = []
            for agent in agents:
                info_specs.append( # 将信息规范的列表堆叠成一个张量。
                    Composite(
                        {
                            "info": Composite(
                                {
                                    key: Unbounded(
                                        shape=torch.as_tensor(value).shape,
                                        device=self.device,
                                    )
                                    for key, value in info_dict[agent].items()
                                }
                            )
                        },
                        device=self.device,
                    )
                )
            info_specs = torch.stack(info_specs, dim=0)
            if ("info", "action_mask") in info_specs.keys(True, True): # 检查信息规范中是否包含 "action_mask"。
                if not self.has_action_mask[group]: # 如果该组没有动作掩码，则创建一个，并将其添加到观察规范中。
                    self.has_action_mask[group] = True
                    group_action_spec = self.input_spec[
                        "full_action_spec", group, "action"
                    ]
                    self.observation_spec[group]["action_mask"] = Categorical(
                        n=2,
                        shape=group_action_spec.shape
                        if not self.categorical_actions
                        else (*group_action_spec.shape, group_action_spec.space.n),
                        dtype=torch.bool,
                        device=self.device,
                    )
                group_inner_info_spec = info_specs["info"] # 获取组的内部信息规范并删除动作掩码。
                del group_inner_info_spec["action_mask"]

            if len(info_specs["info"].keys()): # 如果信息规范有键，则更新该组的观察规范。
                self.observation_spec[group].update(info_specs)

        if self.return_state: # 检查是否需要返回状态。
            try: # 尝试将状态空间转换为 Torch RL 规范，如果失败则使用当前状态创建 UnboundedContinuousTensorSpec。
                state_spec = _gym_to_torchrl_spec_transform(
                    self.state_space,
                    remap_state_to_observation=False,
                    device=self.device,
                )
            except AttributeError:
                state_example = torch.as_tensor(self.state(), device=self.device)
                state_spec = Unbounded(
                    shape=state_example.shape,
                    dtype=state_example.dtype,
                    device=self.device,
                )
            self.observation_spec["state"] = state_spec # 将状态规范添加到观察规范中。

        # Caching # 创建重置输出的缓存。
        self.cached_reset_output_zero = self.observation_spec.zero()
        self.cached_reset_output_zero.update(self.output_spec["full_done_spec"].zero())

        self.cached_step_output_zero = self.observation_spec.zero()
        self.cached_step_output_zero.update(self.output_spec["full_reward_spec"].zero())
        self.cached_step_output_zero.update(self.output_spec["full_done_spec"].zero())

    def _set_seed(self, seed: int):
        self.seed = seed
        self.reset(seed=self.seed)

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        if tensordict is not None: # 如果传入的 tensordict 不为空，检查 _reset 的信号。
            _reset = tensordict.get("_reset", None)
            if _reset is not None and not _reset.all():
                raise RuntimeError(
                    f"An attempt to call {type(self)}._reset was made when no "
                    f"reset signal could be found. Expected '_reset' entry to "
                    f"be `tensor(True)` or `None` but got `{_reset}`."
                )
        if self.parallel: # 根据环境类型重置并获取观察和信息字典。
            # This resets when any is done
            observation_dict, info_dict = self._reset_parallel(**kwargs)
        else:
            # This resets when all are done
            observation_dict, info_dict = self._reset_aec(**kwargs)

        # We start with zeroed data and fill in the data for alive agents
        tensordict_out = self.cached_reset_output_zero.clone() # 克隆缓存的重置输出。
        # Update the "mask" for non-acting agents
        self._update_agent_mask(tensordict_out) # 更新智能体的掩码。
        # Update the "action_mask" for non-available actions
        observation_dict, info_dict = self._update_action_mask( # 更新动作掩码。
            tensordict_out, observation_dict, info_dict
        )

        # Now we get the data (obs and info)
        for group, agent_names in self.group_map.items(): # 遍历组和智能体名称，获取各组的观察和信息。
            group_observation = tensordict_out.get((group, "observation"))
            group_info = tensordict_out.get((group, "info"), None)

            for index, agent in enumerate(agent_names): # 将观察和信息更新到 tensordict_out 中。
                group_observation[index] = self.observation_spec[group, "observation"][
                    index
                ].encode(observation_dict[agent])
                if group_info is not None:
                    agent_info_dict = info_dict[agent]
                    for agent_info, value in agent_info_dict.items():
                        group_info.get(agent_info)[index] = torch.as_tensor(
                            value, device=self.device
                        )

        return tensordict_out

    def _reset_aec(self, **kwargs) -> Tuple[Dict, Dict]: # 定义了一个名为 _reset_aec 的方法，接收任意关键字参数 **kwargs，返回一个元组，包含两个字典。
        self._env.reset(**kwargs) # 调用环境的 reset 方法，重置环境的状态，传入关键字参数。

        observation_dict = { # 创建一个字典 observation_dict，其中每个代理（agent）的观察信息来自环境的 observe 方法。
            agent: self._env.observe(agent) for agent in self.possible_agents
        }
        info_dict = self._env.infos # 从环境中获取信息字典 info_dict，包含环境的额外信息。
        return observation_dict, info_dict

    def _reset_parallel(self, **kwargs) -> Tuple[Dict, Dict]: # 定义了 _reset_parallel 方法，也接收关键字参数，返回两个字典。
        return self._env.reset(**kwargs)

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase: # 定义了 _step 方法，接收一个 tensordict 参数，类型为 TensorDictBase，返回一个 TensorDictBase 对象。
        if self.parallel: # 检查是否以并行模式运行。
            (
                observation_dict,
                rewards_dict,
                terminations_dict,
                truncations_dict,
                info_dict,
            ) = self._step_parallel(tensordict) # 如果是并行模式，调用 _step_parallel 方法，获取观察、奖励、终止、截断和信息字典。
        else: # 否则，调用 _step_aec 方法，获取相同的字典。
            (
                observation_dict,
                rewards_dict,
                terminations_dict,
                truncations_dict,
                info_dict,
            ) = self._step_aec(tensordict)

        # We start with zeroed data and fill in the data for alive agents (初始为全 0, 只有 alive agent 才会进行更新)
        tensordict_out = self.cached_step_output_zero.clone() # 克隆一个零值的 tensordict_out，用于存储输出数据。
        # Update the "mask" for non-acting agents, 直接更新 tensordict_out 中的 mask
        self._update_agent_mask(tensordict_out) # 更新代理的掩码，标记哪些代理是“活”的。
        # Update the "action_mask" for non-available actions
        observation_dict, info_dict = self._update_action_mask( # 更新可用的动作掩码，并返回更新后的观察字典和信息字典。
            tensordict_out, observation_dict, info_dict
        ) # 这里 observation_dict 里面就是 agent 的信息

        # Now we get the data
        for group, agent_names in self.group_map.items(): # 遍历代理组和代理名称
            group_observation = tensordict_out.get((group, "observation"))
            group_reward = tensordict_out.get((group, "reward"))
            group_done = tensordict_out.get((group, "done"))
            group_terminated = tensordict_out.get((group, "terminated"))
            group_truncated = tensordict_out.get((group, "truncated"))
            group_info = tensordict_out.get((group, "info"), None) # 获取每个组的观察、奖励、完成、终止、截断和信息数据。

            for index, agent in enumerate(agent_names): # 遍历每个代理在组中的索引和名称。
                if agent in observation_dict:  # Live agents # 如果代理在观察字典中，表示该代理是活的。
                    group_observation[index] = self.observation_spec[ # 对该代理的观察进行编码并存储在组观察中。
                        group, "observation"
                    ][index].encode(observation_dict[agent])
                    group_reward[index] = torch.tensor( # 将该代理的奖励转换为张量并存储在组奖励中。
                        rewards_dict[agent],
                        device=self.device,
                        dtype=torch.float32,
                    )
                    group_done[index] = torch.tensor( # 根据该代理的终止或截断状态更新完成标志。
                        terminations_dict[agent] or truncations_dict[agent],
                        device=self.device,
                        dtype=torch.bool,
                    )
                    group_truncated[index] = torch.tensor(
                        truncations_dict[agent],
                        device=self.device,
                        dtype=torch.bool,
                    )
                    group_terminated[index] = torch.tensor(
                        terminations_dict[agent],
                        device=self.device,
                        dtype=torch.bool,
                    )

                    if group_info is not None: # 如果组信息不为空，获取代理的附加信息。
                        agent_info_dict = info_dict[agent]
                        for agent_info, value in agent_info_dict.items(): # 遍历代理信息字典中的每个信息项。
                            group_info.get(agent_info)[index] = torch.tensor( # 将信息值转换为张量并存储在组信息中。
                                value, device=self.device
                            )

                elif self.use_mask:
                    if agent in self.agents: # 如果该代理在所有代理列表中。
                        raise ValueError( # 抛出错误，表示找不到该代理。
                            f"Dead agent {agent} not found in step observation but still available in {self.agents}"
                        )
                    # Dead agent
                    terminated = ( # 确定该代理的终止、截断和完成状态。
                        terminations_dict[agent] if agent in terminations_dict else True
                    )
                    truncated = (
                        truncations_dict[agent] if agent in truncations_dict else True
                    )
                    done = terminated or truncated
                    group_done[index] = done
                    group_terminated[index] = terminated
                    group_truncated[index] = truncated

                else: # 如果没有使用掩码且找到死代理，抛出错误。
                    # Dead agent, if we are not masking it out, this is not allowed
                    raise ValueError(
                        "Dead agents found in the environment,"
                        " you need to set use_mask=True to allow this."
                    )

        # set done values
        done, terminated, truncated = self._aggregate_done( # 调用 _aggregate_done 方法，聚合完成、终止和截断状态。
            tensordict_out, use_any=self.done_on_any
        )

        tensordict_out.set("done", done) # 在输出的 tensordict 中设置完成、终止和截断的状态。
        tensordict_out.set("terminated", terminated)
        tensordict_out.set("truncated", truncated)
        return tensordict_out

    def _aggregate_done(self, tensordict_out, use_any): # 定义了一个名为 _aggregate_done 的方法，接受两个参数：tensordict_out（一个张量字典）和 use_any（一个布尔值）。
        done = False if use_any else True # 根据 use_any 的值初始化 done、truncated 和 terminated 三个变量。如果 use_any 为 True，则它们初始化为 False；否则，初始化为 True。 
        truncated = False if use_any else True
        terminated = False if use_any else True
        for key in self.done_keys:
            if isinstance(key, tuple):  # Only look at group keys # 检查当前键是否为元组（用于分组键）
                if use_any:
                    if key[-1] == "done": # 如果键的最后一个元素是 "done"，则使用 or 逻辑更新 done 的值，查看对应的张量是否有任何一个为 True。
                        done = done | tensordict_out.get(key).any()
                    if key[-1] == "terminated":
                        terminated = terminated | tensordict_out.get(key).any()
                    if key[-1] == "truncated":
                        truncated = truncated | tensordict_out.get(key).any()
                    if done and terminated and truncated: # 如果 done、terminated 和 truncated 都为 True，则退出循环，因为所有值都已经被确定。
                        # no need to proceed further, all values are flipped
                        break
                else: # 如果 use_any 为 False，则进入此条件块。
                    if key[-1] == "done":
                        done = done & tensordict_out.get(key).all() # 检查 "done" 状态，使用 and 逻辑更新 done 的值，确保所有对应的张量都为 True。
                    if key[-1] == "terminated":
                        terminated = terminated & tensordict_out.get(key).all()
                    if key[-1] == "truncated":
                        truncated = truncated & tensordict_out.get(key).all()
                    if not done and not terminated and not truncated: # 如果 done、terminated 和 truncated 都为 False，则退出循环。
                        # no need to proceed further, all values are flipped
                        break
        return (
            torch.tensor([done], device=self.device),
            torch.tensor([terminated], device=self.device),
            torch.tensor([truncated], device=self.device),
        )

    def _step_parallel( # 定义 _step_parallel 方法，接受一个张量字典 tensordict，返回五个字典。
        self,
        tensordict: TensorDictBase,
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        action_dict = {} # 初始化一个空字典 action_dict，用于存储动作。
        for group, agents in self.group_map.items(): # 遍历 self.group_map 中的每个组及其对应的代理。
            if group == "non_agents":
                break
            else:    
                group_action = tensordict.get((group, "action")) # 从 tensordict 中获取当前组的动作。
                group_action_np = self.input_spec[  # 将获取的动作转换为 NumPy 数组。
                    "full_action_spec", group, "action"
                ].to_numpy(group_action)
                for index, agent in enumerate(agents):
                    action_dict[agent] = group_action_np[index]

        return self._env.step(action_dict) # 这里跳转到pz_env.py

    def _step_aec( # 定义 _step_aec 方法，接受一个张量字典 tensordict，返回五个字典。
        self,
        tensordict: TensorDictBase,
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        for group, agents in self.group_map.items(): # 遍历 self.group_map 中的每个组及其对应的代理。
            if self.agent_selection in agents: # 检查当前选择的代理是否在该组中。
                agent_index = agents.index(self._env.agent_selection) # 获取当前选择代理在组中的索引。
                group_action = tensordict.get((group, "action")) # 获取当前组的动作。
                group_action_np = self.input_spec[ # 将该动作转换为 NumPy 数组。
                    "full_action_spec", group, "action"
                ].to_numpy(group_action)
                action = group_action_np[agent_index] # 提取当前选择代理的具体动作。
                break # 退出循环，因为已经找到了对应的代理和动作。

        self._env.step(action) # 执行当前选择代理的动作。
        terminations_dict = self._env.terminations # 获取环境的终止、截断、信息和奖励字典。
        truncations_dict = self._env.truncations
        info_dict = self._env.infos
        rewards_dict = self._env.rewards
        observation_dict = { # 创建一个字典，记录所有可能代理的观察信息。
            agent: self._env.observe(agent) for agent in self.possible_agents
        }
        return ( # 最后，返回包含观察、奖励、终止、截断和信息的五个字典。
            observation_dict,
            rewards_dict,
            terminations_dict,
            truncations_dict,
            info_dict,
        )

    def _update_action_mask(self, td, observation_dict, info_dict): # 这个方法用于更新动作掩码。td是一个包含当前状态信息的字典，observation_dict和info_dict分别包含代理的观察信息和其他信息
        # Since we remove the action_mask keys we need to copy the data
        observation_dict = copy.deepcopy(observation_dict) # 这里使用deepcopy深拷贝observation_dict和info_dict，以确保对原始数据的更改不会影响到外部变量。
        info_dict = copy.deepcopy(info_dict)
        # In AEC only one agent acts, in parallel env self.agents contains the agents alive
        agents_acting = self.agents if self.parallel else [self.agent_selection] # 在并行环境中，agents_acting会包含所有正在行动的代理；否则，只包含当前选择的代理。

        for group, agents in self.group_map.items(): # 遍历所有的代理组和其中的代理。self.group_map是一个映射，定义了代理的分组。
            if self.has_action_mask[group]: # 检查当前组是否有动作掩码。
                group_mask = td.get((group, "action_mask")) # 获取当前组的动作掩码，并将其增加True，目的是确保该掩码的基本有效性。
                group_mask += True
                for index, agent in enumerate(agents): # 遍历当前组中的每个代理及其索引。
                    agent_obs = observation_dict[agent] # 从观察字典和信息字典中获取当前代理的观察信息和其他信息。
                    agent_info = info_dict[agent]
                    if isinstance(agent_obs, Dict) and "action_mask" in agent_obs: # 检查代理的观察信息是否为字典类型，并且是否包含“action_mask”键。
                        if agent in agents_acting: # 如果当前代理在活动代理中，更新group_mask的相应位置为代理的动作掩码（转换为torch.tensor格式）。
                            group_mask[index] = torch.tensor(
                                agent_obs["action_mask"],
                                device=self.device,
                                dtype=torch.bool,
                            )
                        del agent_obs["action_mask"] # 删除代理观察信息中的“action_mask”键，以避免重复使用。
                    elif isinstance(agent_info, Dict) and "action_mask" in agent_info: # 类似地，检查代理信息字典中的“action_mask”键。
                        if agent in agents_acting: # 如果当前代理在活动代理中，更新group_mask的相应位置为代理的信息中的动作掩码。
                            group_mask[index] = torch.tensor(
                                agent_info["action_mask"],
                                device=self.device,
                                dtype=torch.bool,
                            )
                        del agent_info["action_mask"] # 这里会删除 info 里面的 action mask

                group_action_spec = self.input_spec["full_action_spec", group, "action"] # 获取当前组的完整动作规格。
                if isinstance( # 检查动作规格是否是离散型的张量规格。
                    group_action_spec, (Categorical, OneHotDiscreteTensorSpec)
                ):
                    # We update the mask for available actions
                    group_action_spec.update_mask(group_mask.clone()) # 更新当前组的动作规格掩码，使用group_mask的克隆版本，以保持原始掩码不变。

        return observation_dict, info_dict # 最后返回更新后的观察字典和信息字典。

    def _update_agent_mask(self, td): # 这个方法用于更新代理的掩码。
        """这里 group_mask 会更新 tensordict_out["agents"]["mask"] 里面的内容
        """
        if self.use_mask: # 检查是否使用掩码功能。
            # In AEC only one agent acts, in parallel env self.agents contains the agents alive
            agents_acting = self.agents if self.parallel else [self.agent_selection] # 与之前类似，确定正在行动的代理。
            for group, agents in self.group_map.items():
                group_mask = td.get((group, "mask")) # 获取当前组的掩码并初始化。
                group_mask += True

                # We now add dead agents to the mask
                for index, agent in enumerate(agents): # 如果当前代理不在活动代理列表中，则将其在group_mask中的相应位置设为False。
                    if agent not in agents_acting:
                        group_mask[index] = False # 将不在 agents_acting 的 agent 的 group_mask 设置为 False

    def close(self) -> None:
        self._env.close()