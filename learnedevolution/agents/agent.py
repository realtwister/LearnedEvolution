from ..utils.parse_config import ParseConfig;

class Agent(ParseConfig):
    def __init__(self):
        pass;

    def reset(self):
        pass;

    def act(self, observation, deterministic):
        raise NotImplementedError;

    def observe_reward(self, reward):
        pass;
