import argparse
import os
from pathlib import Path
import shutil
import time

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback
import sb3_contrib
import torch as th

from mario_net import create_mario_env, MarioNet

# Suppress DeprecationWarning:
#   WARN: The environment SuperMarioBros-1-1-v0 is out of date. You should consider upgrading to version `v3`.
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

save_dir = Path('./model')
reward_log_path = (save_dir / 'reward_log.csv')

DASH_MOVEMENT = [
    ['right', 'B'],
    ['right', 'A', 'B'],
]

# Test Param
EPISODE_NUMBERS = 20
MAX_TIMESTEP_TEST = 1000

# Model Param
CHECK_FREQ_NUMB = 10000
TOTAL_TIMESTEP_NUMB = 5000000
# LEARNING_RATE = 0.0001
# GAE = 1.0
# ENT_COEF = 0.01
# N_STEPS = 512
# GAMMA = 0.9
# BATCH_SIZE = 64
# N_EPOCHS = 10


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, base_epoch, total_timestep_numb, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.base_epoch = base_epoch
        self.total_timestep_numb = total_timestep_numb

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0 and self.n_calls > 0:
            n_calls = self.n_calls + self.base_epoch
            model_path = (self.save_path / f'best_model_{n_calls}')
            self.model.save(model_path)

            total_reward = [0] * EPISODE_NUMBERS
            total_time = [0] * EPISODE_NUMBERS
            best_reward = 0

            for i in range(EPISODE_NUMBERS):
                state = env.reset()  # reset for each new trial
                done = False
                total_reward[i] = 0
                total_time[i] = 0
                while not done and total_time[i] < MAX_TIMESTEP_TEST:
                    action, _ = self.model.predict(state)
                    state, reward, done, info = env.step(action)
                    total_reward[i] += reward[0]
                    total_time[i] += 1

                if total_reward[i] > best_reward:
                    best_reward = total_reward[i]
                    target_epoch = n_calls

                state = env.reset()  # reset for each new trial

            average_reward = sum(total_reward) / EPISODE_NUMBERS
            print('time steps:', n_calls, '/', self.total_timestep_numb)
            print('average reward:{:.2f}, average time:{:.2f}, best_reward:{:.2f}'
                  .format(average_reward, sum(total_time) / EPISODE_NUMBERS, best_reward))

            with open(reward_log_path, 'a') as f:
                print('{},{:.5f},{:.5f}'.format(n_calls, average_reward, best_reward), file=f)

        return True


def training(sb3_class, model_path=None, base_epoch=0, total_timesteps=None, **kwargs):
    policy_kwargs = dict(
        features_extractor_class=MarioNet,
        features_extractor_kwargs=dict(features_dim=512),
    )
    kwargs['policy_kwargs'] = policy_kwargs | (kwargs['policy_kwargs'] if 'policy_kwargs' in kwargs else {})

    callback = TrainAndLoggingCallback(check_freq=CHECK_FREQ_NUMB, save_path=save_dir,
                                       base_epoch=base_epoch, total_timestep_numb=total_timesteps)

    if model_path is None:
        model = sb3_class('CnnPolicy', env, verbose=0, tensorboard_log=save_dir, **kwargs)
    else:
        model = sb3_class.load(model_path, env=env, verbose=0, tensorboard_log=save_dir, **kwargs)
        callback.base_epoch = base_epoch

    model.learn(total_timesteps=total_timesteps - base_epoch, callback=callback)

def save_png(screen, filename):
    im = Image.fromarray(screen)
    im.save(filename, 'PNG')

def replay(sb3_class, model_path=None, skip_frame_count=4, seed=None, render_mode=None):
    if model_path is None:
        reward_log = pd.read_csv(reward_log_path.absolute(), index_col='timesteps')
        # target_epoch = reward_log.index[len(reward_log.index) - 1]  # 最後のエポックを選択する場合
        target_epoch = reward_log['reward'].idxmax()  # 平均報酬が最大のエポックを選択する場合
        # target_epoch = reward_log['best_reward'].idxmax()  # 最高報酬が最大のエポックを選択する場合

        reward = reward_log.loc[target_epoch]['reward']
        best_reward = reward_log.loc[target_epoch]['best_reward']
        print(f'target epoch:{target_epoch}, reward:{reward}, best:{best_reward}')

        model_path = save_dir / f'best_model_{target_epoch}.zip'

    if seed:
        th.manual_seed(seed)

    print(model_path)
    model = sb3_class.load(model_path, env=env)

    png_dir = 'frames'
    frame_index = 0
    if render_mode == 'png':
        shutil.rmtree(png_dir, ignore_errors=True)
        os.makedirs(png_dir, exist_ok=True)
        screen = env0.unwrapped.screen

    state = env.reset()
    done = False
    plays = 0
    wins = 0
    while plays < 100:
        if render_mode == 'png':
            save_png(screen, '{}/frame{:06}.png'.format(png_dir, frame_index))
            frame_index += 1
        else:
            env0.render()
            time.sleep(1.0 / 50 * (skip_frame_count if skip_frame_count > 0 else 1))

        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
        if done:
            if render_mode:
                return

            state = env.reset()
            if info[0]["flag_get"]:
                wins += 1
            plays += 1

            if seed:
                th.manual_seed(seed)
    print("Model win rate: " + str(wins) + "%")

def find_sb3_class(sb3_algo):
    try:
        sb3_class = getattr(stable_baselines3, sb3_algo)
        return sb3_class
    except AttributeError:
        try:
            sb3_class = getattr(sb3_contrib, sb3_algo)
            return sb3_class
        except AttributeError:
            print(f'Cannot find algorithm: {args.sb3_algo}')
            exit(1)

def parse_literal(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def detect_last_epoch(reward_log_path):
    reward_log = pd.read_csv(reward_log_path, index_col='timesteps')
    nrow = reward_log.shape[0]
    target_epoch = int(reward_log.index[nrow - 1])
    return target_epoch

if __name__ == '__main__':
    class ParamProcessor(argparse.Action):
        def __call__(self, parser, namespace, values, option_strings=None):
            param_dict = getattr(namespace, self.dest, [])
            if param_dict is None:
                param_dict = {}
            k, v = values.split("=")
            param_dict[k] = parse_literal(v)
            setattr(namespace, self.dest, param_dict)

    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('--sb3_algo', default='PPO', help='StableBaseline3 RL algorithm i.e. A2C, DQN, PPO, QRDQN, SAC, TD3, HER, etc.')
    parser.add_argument('--movement', default='simple', help='simple, complex or dash')
    parser.add_argument('--world', help='world', type=int, default=1)
    parser.add_argument('--stage', help='stage', type=int, default=1)
    parser.add_argument('--replay', help='Replay mode', nargs='?', default=False)
    parser.add_argument('--render-mode', help='Rendering mode for replay (e.g. png)', type=str)
    parser.add_argument('--seed', help='Random seed for replay mode', type=int, default=None)
    parser.add_argument('--plot', help='Plot mode', nargs='?', default=False)
    parser.add_argument('--continue', help='Continue training', action='store_true')
    parser.add_argument('--total-timesteps', help='Total timesteps', type=int, default=TOTAL_TIMESTEP_NUMB)
    parser.add_argument('--skip-frame', help='Skip frame count (0=disabled, default: 4)', type=int, default=4)
    parser.add_argument('--color', help='Color', action='store_true')
    parser.add_argument("--param", help='Parameter for sb3_algo', action=ParamProcessor)
    args = parser.parse_args()

    env_name = f'SuperMarioBros-{args.world}-{args.stage}-v0'
    movements = dict(simple=SIMPLE_MOVEMENT, complex=COMPLEX_MOVEMENT, dash=DASH_MOVEMENT)
    env, env0 = create_mario_env(env_name, movements[args.movement],
                                skip_frame_count=args.skip_frame,
                                is_color=args.color)

    print(f'RL Algorithm: {args.sb3_algo}')
    sb3_class = find_sb3_class(args.sb3_algo)

    if args.replay != False:
        replay(sb3_class, args.replay, args.skip_frame, args.seed, args.render_mode)
    elif args.plot != False:
        log_path = args.plot if args.plot else reward_log_path
        reward_log = pd.read_csv(log_path, index_col='timesteps')
        reward_log.plot()
        plt.show()
    else:
        kDefaultParams = dict(
            DQN = dict(
                buffer_size = 100000,
            ),
            QRDQN = dict(
                buffer_size = 100000,
                policy_kwargs = dict(
                    optimizer_class = th.optim.Adam,
                    optimizer_kwargs = dict(eps=0.01 / 32),  # Proposed in the QR-DQN paper where `batch_size = 32`
                ),
            ),
            A2C = dict(
                learning_rate = 0.0001,
                policy_kwargs = dict(
                    optimizer_class = th.optim.RMSprop,
                    optimizer_kwargs = dict(alpha=0.99, eps=1e-5, weight_decay=0),  # eps=rms_prop_eps
                ),
            ),
            TRPO = dict(
                learning_rate = 3e-4,
            ),
        )

        kwargs = args.param if args.param else {}
        kwargs = kDefaultParams.get(args.sb3_algo, {}) | kwargs

        if getattr(args, 'continue'):
            os.makedirs(save_dir, exist_ok=True)
            epoch = detect_last_epoch(reward_log_path)
            model_path = save_dir / f'best_model_{epoch}'
            kwargs['model_path'] = model_path
            kwargs['base_epoch'] = epoch
        else:
            shutil.rmtree(save_dir, ignore_errors=True)
            os.makedirs(save_dir)
            with open(reward_log_path, 'w') as f:
                print('timesteps,reward,best_reward', file=f)

        training(sb3_class, total_timesteps=args.total_timesteps,
                 **kwargs)
