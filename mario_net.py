import cv2
import numpy as np

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import gymnasium as gym

from gymnasium.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

class ResizeEnv(gym.ObservationWrapper):
    def __init__(self, env, size):
        gym.ObservationWrapper.__init__(self, env)
        (oldh, oldw, oldc) = env.observation_space.shape
        newshape = (size, size, oldc)
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=newshape, dtype=np.uint8)

    def observation(self, frame):
        height, width, _ = self.observation_space.shape
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if frame.ndim == 2:
            frame = frame[:,:,None]
        return frame

class CustomRewardAndDoneEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(CustomRewardAndDoneEnv, self).__init__(env)
        self.current_score = 0
        self.current_x = 0
        self.stay_x_count = 0
        self.max_x = 0

    def reset(self, **kwargs):
        self.current_score = 0
        self.current_x = 0
        self.stay_x_count = 0
        self.max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, truncated, terminated, info = self.env.step(action)
        x_pos = info['x_pos']
        if x_pos > self.max_x:
            self.stay_x_count = 0
            reward += x_pos - self.max_x
        else:
            reward -= 0.1
            self.stay_x_count += 1
            if self.stay_x_count > 50 * 10:
                reward -= 100
                truncated = True
        if info['flag_get']:
            reward += 500
            truncated = True
            print('GOAL')
        if info['life'] < 2:
            reward -= 500
            truncated = True
        self.current_score = info['score']
        self.current_x = x_pos
        self.max_x = max(self.max_x, self.current_x)
        return state, reward / 10., truncated, terminated, info


class MarioNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim):
        super(MarioNet, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

def create_mario_env(stage_name, movement, skip_frame_count = 0, is_color = False):
    env = gym_super_mario_bros.make(stage_name)
    env0 = env
    env = JoypadSpace(env, movement)
    env = CustomRewardAndDoneEnv(env)
    if skip_frame_count > 1:
        env = SkipFrame(env, skip=skip_frame_count)
    if not is_color:
        env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeEnv(env, size=84)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    return env, env0
