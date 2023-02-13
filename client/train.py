import logging

from rl_zoo3.train import train

from registration import register_custom_env

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d/%m/%y %H:%M:%S',
                    level=logging.WARNING)

register_custom_env(id="SmallSizeLeague",
                    entry_point="env.smallsize:SmallSizeEnv")

if __name__ == "__main__":
    train()
