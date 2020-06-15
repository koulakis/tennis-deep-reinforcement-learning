from typing import Any, Tuple, Dict, Optional, List, Union, Sequence
import random
from enum import Enum

from gym.spaces import Box
import gym
from unityagents import UnityEnvironment, BrainInfo, BrainParameters
import numpy as np

from tennis.definitions import ROOT_DIR


AGENT_ENVIRONMENT_PATH = str(ROOT_DIR / 'unity_tennis_environment/Tennis.x86_64')


class Player(str, Enum):
    first = 'first'
    second = 'second'


class UnityEnvironmentWrapperToGym(gym.Env):
    def __init__(
            self,
            *args,
            train_mode: bool = True,
            seed: Optional[int] = None,
            port: Optional[int] = None,
            **kwargs):
        """A wrapper class which translates the given Unity environment to a gym environment. It is setup to work with
        the single agent reacher environment.

        Args:
            *args: arguments which are directly passed to the Unity environment. This is supposed to make the
                the initialization of the wrapper very similar to the initialization of the Unity environment.
            train_mode: toggle to set the unity environment to train mode
            seed: sets the seed of the environment - if not given, a random seed will be used
            port: port of the environment, used to be able to run multiple environment concurrently
        """
        self.train_mode = train_mode
        self.unity_env, self.brain_name, self.brain = _setup_unity_environment(
            *args,
            path=AGENT_ENVIRONMENT_PATH,
            port=port,
            seed=seed,
            **kwargs)

        self.action_space, self.observation_space, self.reward_range = _environment_specs(self.brain)

        self.num_envs = 1
        self.episode_step = 0
        self.episode_rewards = np.zeros(2)

    def step(self, action) -> Tuple[Any, float, bool, Dict[str, Any]]:
        brain_info = self.unity_env.step(action)[self.brain_name]
        state, reward, done, rewards = self._parse_brain_info(brain_info)

        self.episode_rewards += rewards
        self.episode_step += 1

        info = (
            dict(episode=dict(
                r=max(self.episode_rewards),
                l=self.episode_step))
            if done else dict())

        return state, reward, done, info

    def reset(self) -> np.ndarray:
        brain_info = self.unity_env.reset(train_mode=self.train_mode)[self.brain_name]
        self.episode_step = 0
        self.episode_rewards = np.zeros(2)

        return self._parse_brain_info(brain_info)[0]

    def render(self, mode='human') -> None:
        pass

    def close(self):
        self.unity_env.close()

    @staticmethod
    def _parse_brain_info(info: BrainInfo) -> Tuple[np.ndarray, float, bool, np.ndarray]:
        """Extract the state, reward and done information from an environment brain."""
        observations = info.vector_observations.flatten()
        rewards = info.rewards
        done = any(info.local_done)

        return observations, np.array(rewards).mean(), done, rewards


def _setup_unity_environment(
        *args,
        path: str,
        port: Optional[int],
        seed: Optional[int],
        **kwargs
) -> Tuple[UnityEnvironment, str, BrainParameters]:
    """Setup a Unity environment and return it and its brain."""
    kwargs['file_name'] = path
    kwargs['seed'] = random.randint(0, int(1e6)) if not seed else seed
    if port:
        kwargs['base_port'] = port

    unity_env = UnityEnvironment(*args, **kwargs)
    brain_name = unity_env.brain_names[0]
    brain = unity_env.brains[brain_name]

    return unity_env, brain_name, brain


def _environment_specs(brain: BrainParameters) -> Tuple[Box, Box, Tuple[float, float]]:
    """Extract the action space, observation space and reward range info from an environment brain. Here the two agent
    action and observation spaces are stacked together in order for them to be treated as one agent."""
    action_space_size = 2 * brain.vector_action_space_size
    observation_space_size = 2 * 24  # brain.vector_observation_space_size
    action_space = Box(
        low=np.array(action_space_size * [-1.0]),
        high=np.array(action_space_size * [1.0]))
    observation_space = Box(
        low=np.array(observation_space_size * [-float('inf')]),
        high=np.array(observation_space_size * [float('inf')]))
    reward_range = (-float('inf'), float('inf'))

    return action_space, observation_space, reward_range
