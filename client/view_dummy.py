import logging

from os import path
from typing import Optional, Tuple

import numpy as np
import gym

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.base_class import BaseAlgorithm
from rl_zoo3.utils import get_model_path, get_saved_hyperparams

from client.env.smallsize import SmallSizeEnv
from registration import register_custom_env

#
# Setting up logger
#

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d/%m/%y %H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

#
# Configuring parameters
#

env_name = "SmallSizeLeague-v1"
algo_name = "ppo"

n_eval_episodes = 20

#
# Running the script
#

logger.info("Visualização de um agente")

register_custom_env(id="SmallSizeLeague",
                    entry_point="env.smallsize:SmallSizeEnv")

name_prefix, model_path, log_path = get_model_path(
    exp_id=1,
    folder="./logs",
    algo=algo_name,
    env_name=env_name,
    load_best=False,
    load_last_checkpoint=True
)

stats_path = path.join(log_path, env_name)
hyperparams, stats_path = get_saved_hyperparams(stats_path)

class DummyWrapper(BaseAlgorithm):
    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        self.action_space = action_space
        self.observation_space = observation_space

    def _setup_model(self) -> None:
        pass

    def learn(self, total_timesteps, callback, log_interval, tb_log_name, eval_env = None, eval_freq = -1, n_eval_episodes = 5, eval_log_path = None, reset_num_timesteps = True, progress_bar = False):
        pass

    def predict(self, observation: np.ndarray, state: Optional[Tuple[np.ndarray, ...]] = None, episode_start: Optional[np.ndarray] = None, deterministic: bool = False) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        num_envs = observation.shape[0]

        act = np.zeros(shape=(num_envs, self.action_space.shape[0]))
        obs = np.zeros(shape=(num_envs, self.observation_space.shape[0]))

        for i in range(num_envs):
            act[i] = self.action_space.sample()
            obs[i] = self.observation_space.sample()

        return act, obs

if __name__ == '__main__':
    env = SmallSizeEnv(view_mode=True, id=99)
    check_env(env)

    model = DummyWrapper(env.action_space, env.observation_space)

    rewards, _ = evaluate_policy(
        model, env, return_episode_rewards=True, n_eval_episodes=n_eval_episodes)
    rewards = np.array(rewards)

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    logger.info(f"   reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()
