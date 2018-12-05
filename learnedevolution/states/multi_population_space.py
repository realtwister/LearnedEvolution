import numpy as np;
from collections import deque;
from gym.spaces import Box;

from .state import State;
from ..utils.parse_config import config_factory

class MultiPopulationSpace(State):
    def __init__(self,
        single_state,
        number_of_states = 2,
        epsilon = 1e-20):
        self.single_state = single_state
        self.epsilon = epsilon
        self.number_of_states = number_of_states

    def _reset(self):
        self._populations = deque(maxlen = self.number_of_states);

    def append_population(self, population):
        self._populations.appendleft(population);
    def _encode(self, population):
        self.append_population(population);
        total_state = [];
        for i in range(self.number_of_states):
            if i < len(self._populations):
                current_population = self._populations[i]
            else:
                current_population = None
            total_state.append(self.single_state(current_population, self._populations[0]));
        total_state = np.stack(total_state);
        if np.any(np.isnan(total_state)):
            print("state is NaN");
        if np.any(np.isinf(total_state)):
            print("state is Inf");
        return total_state.flatten();

    def _decode(self, action):
        if len(self._populations) == 0:
            return action;
        return self.single_state.invert(action,self._populations[0]);


    @property
    def state_space(self):
        single_state_size= self.single_state.state_space['shape'][0];
        return dict(type='float', shape=(self.number_of_states*single_state_size,));

    @property
    def gym_state_space(self):
        single_state_size= self.single_state.state_space['shape'][0];
        return Box(high = 10, low  = -10,shape=(self.number_of_states*single_state_size,));

    @property
    def action_space(self):
        return dict(type='float', shape=(self.single_state.action_space['shape'][0],));

    @property
    def gym_action_space(self):
        return Box(high = 100, low= -100, shape=(self.single_state.action_space['shape'][0],));

    @classmethod
    def _get_kwargs(cls, config, key = ""):
        cls._config_required(
            'single_state',
            'number_of_states',
            'epsilon'
        )
        cls._config_defaults(
            single_state = dict(
                type = "TranslationScaleInvariant"
            ),
            number_of_states = 2,
            epsilon = 1e-20
        )
        kwargs = super()._get_kwargs(config, key = key);

        from .single_states import single_state_classes

        kwargs['single_state'] = config_factory(
            single_state_classes,
            config,
            key+'.single_state')
        return kwargs
