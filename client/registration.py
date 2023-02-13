import logging
logger = logging.getLogger(__name__)

import gym.envs
from gym.envs.registration import registry

# Current environment versions for registration
current_versions = [1, 2]


def register_custom_env(id: str, entry_point: str):
    for env in registry.env_specs:
        if id in env:
            logger.warn(f"Removing {env} from registry")
            del registry.env_specs[env]

    for current_version in current_versions:
        env_id = f"{id}-v{current_version}"
        logger.info(f"Registering {env_id} from registry")
        gym.envs.register(
            id=env_id,
            entry_point=entry_point,
        )
