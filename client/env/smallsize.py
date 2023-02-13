from . import config_utils as utils
from .codegen import smallsize_pb2 as smallsize

import zmq
import gym.utils
import gym
import os
import time
import logging

import subprocess as sp
import multiprocessing as mp
import numpy as np

# Raises error when dealing with NaNs and infs
np.seterr(all='raise')


class SmallSizeEnv(gym.Env, gym.utils.EzPickle):
    """
    Small Size League (SSL) environment.
    Thin wrapper to the c++ agents that define the task.
    """

    metadata = {'render.modes': ['human']}
    context = zmq.Context(1)

    def __init__(self, view_mode=False, id=-1):
        super(SmallSizeEnv, self).__init__()

        # Current process id
        if id == -1:
            proc_id = mp.current_process()._identity
            if len(proc_id):
                id = mp.current_process()._identity[0]
            else:
                id = 0

        # Set up logger
        self.logger = logging.getLogger(f'SmallsizeEnv[id={id}]')
        self.logger.debug(f"id = {id}")

        # Current working directory
        cwd = os.getcwd()
        self.logger.debug(f"cwd={cwd}")

        # Set up itaSim config
        config_path = os.path.join(cwd, "./config/")
        utils.parse_xml(config_path, id)

        # Start the agent and the simulator (background processes)
        self.logger.info("Starting the agent and simulator processes...")

        self.contr_proc = self._controlled_agent_job(id, cwd, view_mode)

        # Start connection with the agent (c++)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://127.0.0.1:" + str(5000 + id))

        # Ask for the setup (observation and action spaces)
        req = smallsize.WrapperRequest()
        req.type = smallsize.SetupEnvRequestType
        req.setup_env.CopyFrom(smallsize.SetupEnvRequest())
        self.socket.send(req.SerializeToString())

        self.logger.info("Waiting for SetupEnvResponse...")

        res = smallsize.WrapperResponse()
        res.ParseFromString(self.socket.recv())
        assert res.type == smallsize.SetupEnvResponseType

        setup = res.setup_env
        self.logger.info(f"Number states= {setup.num_state_dim}")
        self.logger.info(f"Number action= {setup.num_action_dim}")
        self.logger.info(f"Action bounds= {setup.action_bound}")

        # Build observation and action spaces for the Gym Environment
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(setup.num_state_dim, ),
            dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=-np.array(setup.action_bound, dtype=np.float32),
            high=np.array(setup.action_bound, dtype=np.float32),
            dtype=np.float32)

    def reset(self):
        self.logger.info("Called reset => Sending EpisodeRequest")

        req = smallsize.WrapperRequest()
        req.type = smallsize.EpisodeRequestType
        req.episode.CopyFrom(smallsize.EpisodeRequest())
        self.socket.send(req.SerializeToString())

        res = smallsize.WrapperResponse()
        res.ParseFromString(self.socket.recv())
        assert res.type == smallsize.EpisodeResponseType

        response = res.episode
        return np.array(response.state.observation, dtype=np.float32)

    def step(self, action):
        self.logger.debug("Called step  => Sending SimulationRequest")
        self.logger.debug("               action: {}".format(action))

        content = smallsize.Action()
        content.action.extend(action)

        sim = smallsize.SimulationRequest()
        sim.action.CopyFrom(content)

        req = smallsize.WrapperRequest()
        req.type = smallsize.SimulationRequestType
        req.simulation_env.CopyFrom(sim)
        self.socket.send(req.SerializeToString())

        res = smallsize.WrapperResponse()
        res.ParseFromString(self.socket.recv())
        assert res.type == smallsize.SimulationResponseType

        response = res.simulation_env
        self.logger.debug("               observation: {}".format(response.state.observation))
        self.logger.debug("               reward: {}".format(response.reward))
        self.logger.debug("               done?: {}".format(response.done))

        return np.array(response.state.observation, dtype=np.float32), response.reward, response.done, {}

    def render(self, mode='human'):
        return

    def close(self):
        self.logger.debug("Called close => Sending CloseRequest")

        # Close connection
        req = smallsize.WrapperRequest()
        req.type = smallsize.CloseRequestType
        req.close.CopyFrom(smallsize.CloseRequest())
        self.socket.send(req.SerializeToString())
        time.sleep(10.0)

        # kill background processes
        self.contr_proc.terminate()

        # closing subprocess logs file
        self.log_file.close()

        return

    def _controlled_agent_job(self, id: str, cwd: str, view_mode: bool):
        self.logger.info(
            f"Executing subprocess {id} on {cwd} with view_mode={view_mode}")

        env = os.environ
        env["LOG_LEVEL"] = "WARNING"
        # env["LOG_LEVEL"] = "INFO"
        env["HEADLESS_FLAG"] = "-H" if not view_mode else ""

        self.log_file = open(f"./executions/debug_agent_target_{id}.txt", 'w')

        proc = sp.Popen(args=f"./binaries/rl_runner {id}",
                        stdout=self.log_file, stderr=sp.STDOUT,
                        cwd=cwd, env=env, shell=True)
        self.logger.info("Agent process started")

        try:
            return_code = proc.wait(timeout=10.0)
            self.logger.error(
                f"Controlled agent process early termination: return code = {return_code}")
        except sp.TimeoutExpired:
            # Timeout => still running, as expected
            return proc
