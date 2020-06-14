import random
from typing import Optional
from enum import Enum

import stable_baselines3.td3 as td3
import stable_baselines3.sac as sac
import typer
from stable_baselines3.common.noise import NormalActionNoise
from torch import nn
import numpy as np

from tennis.environment_wrappers import UnityMultiAgentEnvironmentWrapper
from tennis.definitions import ROOT_DIR

EXPERIMENTS_DIR = ROOT_DIR / 'experiments'


class RLAlgorithm(str, Enum):
    td3 = 'td3'
    sac = 'sac'


algorithm_and_policy = {
    RLAlgorithm.td3: (td3.TD3, td3.MlpPolicy),
    RLAlgorithm.sac: (sac.SAC, sac.MlpPolicy)
}


def train(
        experiment_name: str = typer.Option(...),
        total_timesteps: int = int(5e5),
        env_seed: int = random.randint(0, int(1e6)),
        environment_port: int = 6005,
        device: str = 'cuda',
        gamma: float = 0.98,
        learning_rate: float = 7.3e-4,
        layers_comma_sep: str = '400,300',
        eval_freq: int = 100000,
        n_eval_episodes: int = 5,
        rl_algorithm: RLAlgorithm = RLAlgorithm.sac,
        batch_size: int = 256,
        buffer_size: int = 300000,
        gradient_steps: int = 64,
        learning_starts: int = 10000,
        sac_tau: float = 0.02,
        sac_train_freq: int = 64,
        td3_noise_type: Optional[str] = None,
        td3_noise_std: Optional[float] = None
):
    """Train two agent in the tennis environment. Training is using single agent algorithms to train both agents with
    the union of their observations.

    Args:
        experiment_name: the name of the experiment which will be used to create a directory under 'experiments' and
            store there all training artifacts along with the final and best models
        total_timesteps: the number of timestamps to run till stopping training
        env_seed: a seed for the environment random initialization - if not set, defaults to random
        environment_port: this is the port used by the unity environment to communicate with the C# backend. One needs
            to set different ports to different environments which run in parallel.
        device: the device used to train the model, can be 'cpu' or 'cuda:x'
        gamma: the discount rate applied to future actions
        learning_rate: the learning rate used by the policy and value network optimizer
        layers_comma_sep: a sequence of layer width for the networks as a comma-separated list
        eval_freq: the number of steps after which a validation round will take place. Whenever there is an improvement,
            the best model will be saved under the 'eval' directory in the experiment. Available only for the single
            agent environment.
        n_eval_episodes: number of episodes run during evaluation, available only for the single agent environment
        rl_algorithm: the algorithm used to train an agent
        batch_size: the batch size used during training
    """
    experiment_path = EXPERIMENTS_DIR / experiment_name
    model_path = experiment_path / 'model'
    eval_path = experiment_path / 'eval'
    tensorboard_log_path = experiment_path / 'tensorboard_logs'
    for path in [experiment_path, eval_path, tensorboard_log_path]:
        path.mkdir(exist_ok=True, parents=True)

    environment_parameters = dict(
        seed=env_seed,
        no_graphics=True,
        train_mode=True,
        environment_port=environment_port)

    env = UnityMultiAgentEnvironmentWrapper(**environment_parameters)

    algorithm_class, policy = algorithm_and_policy[rl_algorithm]

    layers = [int(layer_width) for layer_width in layers_comma_sep.split(',')]

    policy_kwargs = remove_none_entries(dict(
        activation_fn=nn.ReLU,
        net_arch=layers))

    if rl_algorithm == RLAlgorithm.sac:
        algorithm_specific_parameters = dict(
            buffer_size=buffer_size,
            tau=sac_tau,
            train_freq=sac_train_freq,
            gradient_steps=gradient_steps,
            learning_starts=learning_starts
        )
    elif rl_algorithm == RLAlgorithm.td3:
        action_shape = (env.num_envs, env.action_space.shape[0])
        action_noise = (
            NormalActionNoise(
                np.zeros(action_shape, dtype=np.float32),
                td3_noise_std * np.ones(action_shape, dtype=np.float32))
            if td3_noise_type == 'normal'
            else None)
        algorithm_specific_parameters = remove_none_entries(dict(
            buffer_size=buffer_size,
            gradient_steps=gradient_steps,
            learning_starts=learning_starts,
            action_noise=action_noise))
    else:
        raise ValueError(f'Unknown algorithm: {rl_algorithm}')

    model = algorithm_class(
        policy,
        env,
        verbose=1,
        tensorboard_log=str(tensorboard_log_path),
        device=device,
        gamma=gamma,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        **remove_none_entries(algorithm_specific_parameters))

    model.learn(
        total_timesteps=total_timesteps,
        eval_env=env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        eval_log_path=str(eval_path))

    model.save(str(model_path))


def remove_none_entries(d):
    return {k: v for k, v in list(d.items()) if v is not None}


if __name__ == '__main__':
    typer.run(train)
