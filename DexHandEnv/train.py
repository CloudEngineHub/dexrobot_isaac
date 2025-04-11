# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import hydra

from omegaconf import DictConfig, OmegaConf
from omegaconf import DictConfig, OmegaConf


def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

    train_cfg = config_dict["params"]["config"]

    train_cfg["device"] = cfg.rl_device

    train_cfg["population_based_training"] = cfg.pbt.enabled
    train_cfg["pbt_idx"] = cfg.pbt.policy_idx if cfg.pbt.enabled else None

    # Make sure we don't accidentally overwrite the experiment name with None
    if train_cfg.get("full_experiment_name"):
        print(f"Preserving full_experiment_name: {train_cfg['full_experiment_name']}")
    else:
        # As a fallback, if it wasn't set earlier
        print("Warning: full_experiment_name was not set properly before preprocessing")

    print(f"Using rl_device: {cfg.rl_device}")
    print(f"Using sim_device: {cfg.sim_device}")
    print(train_cfg)

    try:
        model_size_multiplier = config_dict["params"]["network"]["mlp"][
            "model_size_multiplier"
        ]
        if model_size_multiplier != 1:
            units = config_dict["params"]["network"]["mlp"]["units"]
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(
                f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}'
            )
    except KeyError:
        pass

    return config_dict


@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    import os
    import subprocess
    from datetime import datetime

    # Generate a centralized timestamp for all parts of the code
    # This ensures consistency across all places that use the timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # noinspection PyUnresolvedReferences
    import isaacgym
    from DexHandEnv.pbt.pbt import PbtAlgoObserver, initial_pbt_check
    from DexHandEnv.utils.rlgames_utils import multi_gpu_get_rank
    from hydra.utils import to_absolute_path
    from DexHandEnv.tasks import isaacgym_task_map
    import gym
    from DexHandEnv.utils.reformat import omegaconf_to_dict, print_dict
    from DexHandEnv.utils.utils import set_np_formatting, set_seed

    if cfg.pbt.enabled:
        initial_pbt_check(cfg)

    from DexHandEnv.utils.rlgames_utils import (
        RLGPUEnv,
        RLGPUAlgoObserver,
        MultiObserver,
        ComplexObsRLGPUEnv,
    )
    from DexHandEnv.utils.wandb_utils import WandbAlgoObserver
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from DexHandEnv.learning import amp_continuous
    from DexHandEnv.learning import amp_players
    from DexHandEnv.learning import amp_models
    from DexHandEnv.learning import amp_network_builder
    import DexHandEnv

    # Use the centralized timestamp
    run_name = f"{cfg.wandb_name}_{timestamp}"

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(
        cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank
    )

    def create_isaacgym_env(**kwargs):
        envs = DexHandEnv.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{run_name}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs

    env_configurations.register(
        "rlgpu",
        {
            "vecenv_type": "RLGPU",
            "env_creator": lambda **kwargs: create_isaacgym_env(**kwargs),
        },
    )

    ige_env_cls = isaacgym_task_map[cfg.task_name]
    dict_cls = (
        ige_env_cls.dict_obs_cls
        if hasattr(ige_env_cls, "dict_obs_cls") and ige_env_cls.dict_obs_cls
        else False
    )

    if dict_cls:

        obs_spec = {}
        actor_net_cfg = cfg.train.params.network
        obs_spec["obs"] = {
            "names": list(actor_net_cfg.inputs.keys()),
            "concat": not actor_net_cfg.name == "complex_net",
            "space_name": "observation_space",
        }
        if "central_value_config" in cfg.train.params.config:
            critic_net_cfg = cfg.train.params.config.central_value_config.network
            obs_spec["states"] = {
                "names": list(critic_net_cfg.inputs.keys()),
                "concat": not critic_net_cfg.name == "complex_net",
                "space_name": "state_space",
            }

        vecenv.register(
            "RLGPU",
            lambda config_name, num_actors, **kwargs: ComplexObsRLGPUEnv(
                config_name, num_actors, obs_spec, **kwargs
            ),
        )
    else:

        vecenv.register(
            "RLGPU",
            lambda config_name, num_actors, **kwargs: RLGPUEnv(
                config_name, num_actors, **kwargs
            ),
        )

    rlg_config_dict = omegaconf_to_dict(cfg.train)

    # Set the full_experiment_name directly in the config dict
    experiment_name = cfg.train.params.config.name + f"_{timestamp}"
    rlg_config_dict["params"]["config"]["full_experiment_name"] = experiment_name

    # Print confirmation that we've set the experiment name
    print(f"Setting full_experiment_name to: {experiment_name}")
    print(f"Config full_experiment_name: {rlg_config_dict['params']['config']['full_experiment_name']}")

    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)

    observers = [RLGPUAlgoObserver()]

    if cfg.pbt.enabled:
        pbt_observer = PbtAlgoObserver(cfg)
        observers.append(pbt_observer)

    if cfg.wandb_activate:
        cfg.seed += global_rank
        if global_rank == 0:
            # initialize wandb only once per multi-gpu run
            wandb_observer = WandbAlgoObserver(cfg)
            observers.append(wandb_observer)

    # register new AMP network builder and agent
    def build_runner(algo_observer):
        # Make timestamp available to agent initialization
        from DexHandEnv.learning.common_agent import set_global_timestamp
        set_global_timestamp(timestamp)

        runner = Runner(algo_observer)
        runner.algo_factory.register_builder(
            "amp_continuous", lambda **kwargs: amp_continuous.AMPAgent(**kwargs)
        )
        runner.player_factory.register_builder(
            "amp_continuous", lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs)
        )
        model_builder.register_model(
            "continuous_amp",
            lambda network, **kwargs: amp_models.ModelAMPContinuous(network),
        )
        model_builder.register_network(
            "amp", lambda **kwargs: amp_network_builder.AMPBuilder()
        )

        return runner

    # convert CLI arguments into dictionary
    # create runner and set the settings
    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict)
    runner.reset()

    # dump config dict
    if not cfg.test:
        experiment_dir = os.path.join(
            "runs",
            f"{cfg.train.params.config.name}_{timestamp}"
        )

        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

        # Save git commit hash
        try:
            git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
            with open(os.path.join(experiment_dir, "git_hash.txt"), "w") as f:
                f.write(git_hash)

            # Save git diff
            git_diff = subprocess.check_output(["git", "diff"]).decode("ascii")
            with open(os.path.join(experiment_dir, "git_diff.patch"), "w") as f:
                f.write(git_diff)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: Could not save git information: {e}")

    runner.run(
        {
            "train": not cfg.test,
            "play": cfg.test,
            "checkpoint": cfg.checkpoint,
            "sigma": cfg.sigma if cfg.sigma != "" else None,
        }
    )


if __name__ == "__main__":
    launch_rlg_hydra()
