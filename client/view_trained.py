import logging

from os import path

import numpy as np

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from rl_zoo3.utils import ALGOS, get_model_path, get_saved_hyperparams

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

env_name = "SmallSizeLeague-v2"
algo_name = "ppo"

exp_id = 1
n_eval_episodes=20

#
# Running the script
#

logger.info("Visualização de um agente")

register_custom_env(id="SmallSizeLeague",
                    entry_point="env.smallsize:SmallSizeEnv")

name_prefix, model_path, log_path = get_model_path(
    exp_id=exp_id,
    folder="./logs",
    algo=algo_name,
    env_name=env_name,
    load_best=False,
    load_last_checkpoint=True
)

stats_path = path.join(log_path, env_name)
hyperparams, stats_path = get_saved_hyperparams(stats_path)

if __name__ == '__main__':
    env = SmallSizeEnv(view_mode=True, id=99)
    check_env(env)

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    model = ALGOS[algo_name].load(
        model_path, env=env, custom_objects=custom_objects)

    rewards, _ = evaluate_policy(
        model, env, return_episode_rewards=True, n_eval_episodes=n_eval_episodes)
    rewards = np.array(rewards)

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    logger.info(f"   reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()
