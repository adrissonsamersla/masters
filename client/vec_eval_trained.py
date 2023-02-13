import logging

from os import path
from multiprocessing import freeze_support

import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from rl_zoo3.utils import ALGOS, get_model_path, get_saved_hyperparams

from client.env.smallsize import SmallSizeEnv
from registration import register_custom_env

#
# Setting up logger
#

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d/%m/%y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

#
# Configuring parameters
#

env_name = "SmallSizeLeague-v2"
algo_name = "ppo"

exp_id = 1

n_envs = 10
n_eval_episodes = 500

#
# Running the script
#

logger.info("Avaliação de um agente")

register_custom_env(id="SmallSizeLeague",
                    entry_point="env.smallsize:SmallSizeEnv")

name_prefix, model_path, log_path = get_model_path(
    exp_id=exp_id,
    folder="./logs",
    algo=algo_name,
    env_name=env_name,
    load_best=True,
    load_last_checkpoint=False
)

stats_path = path.join(log_path, env_name)
hyperparams, stats_path = get_saved_hyperparams(stats_path)

if __name__ == '__main__':
    freeze_support()

    env = make_vec_env(SmallSizeEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    model = ALGOS[algo_name].load(
        model_path, env=None, custom_objects=custom_objects)

    logger.info("model loaded")

    env.reset()

    rewards, _ = evaluate_policy(
        model, env, return_episode_rewards=True, n_eval_episodes=n_eval_episodes)
    rewards = np.array(rewards)

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    logger.info(f"   reward = {mean_reward:.5f} +/- {std_reward:.5f}")

    logger.info(f"   saving the array of episodic rewards to eval.npz")
    np.savez('executions/eval.npz', rewards=rewards)

    env.close()
