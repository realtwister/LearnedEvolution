from .utils.parse_config import ParseConfig
from .utils.signals import TimedCallback
from .utils import git_hash
import numpy as np;
import os;
import pickle

class Evaluator(ParseConfig):
    default_topic = "evaluator"
    def __init__(self,
        algorithm,
        problem_suite,
        restoredir,
        logdir,
        step,
        N_episodes,
        name,
        seed,
        summarizer):
        self.algorithm = algorithm
        self.problem_suite = problem_suite
        self.restoredir = restoredir
        self.logdir = logdir
        self.name = name
        self.N_episodes = N_episodes
        self.summarizer = summarizer
        self.step = step

        self.seed(seed)

        if self.restoredir is not None:
            self.restore();

        self.setup_logdir();
        self.i = 0;

    def seed(self, seed = None):
        if seed is None:
            seed = np.round(np.random.rand()*1000).astype(int);
        self.seed = seed;
        self.algorithm.seed(seed);
        self.problem_suite.seed(seed+1);

    def _setup_histories(self):
        self.timed_callback = TimedCallback(event = 'after_generate_population',
                                  sender = self.algorithm,
                                  fns = self._save_population)
        self.timed_callback.connect()
        self.histories = [[]]

    def _save_population(self, *args, **kwargs):
        self.histories[-1] +=[self.summarizer(self.algorithm._population_obj,self.algorithm, self.problem)]

    def _finish_history(self):
        self.histories += [[]];

    def _finish_histories(self):
        self.histories = self.histories[:-1]
        self.timed_callback.disconnect()

    def run(self):
        self._setup_histories();
        for i, problem in enumerate(self.problem_suite.iter(self.N_episodes)):
            self.problem = problem
            self.algorithm.maximize(problem.fitness);
            self._finish_history();
            self.i +=1;
            print(self.i)
        self._finish_histories();
        if self.step is not None:
            path = os.path.join(self.logdir,self.name,self.step+'.pkl')
        else:
            path = os.path.join(self.logdir, self.name+'.pkl')
        with open(path, 'wb') as logfile:
            pickle.dump(self.histories, logfile);
        return self.histories;

    def restore(self):
        assert os.path.exists(self.restoredir), "Restore path does not exist"
        self.algorithm.restore(self.restoredir);

    def setup_logdir(self):
        if not os.path.exists(os.path.join(self.logdir,self.name)):
            os.makedirs(os.path.join(self.logdir,self.name));

    def copy_config_file(self, config_file, name="config.py"):
        if self.step is None:
            new_config = os.path.join(self.logdir, self.name+"."+name)
        else:
            new_config = os.path.join(self.logdir, self.name, self.step+"."+name);
        with open(config_file, 'r') as original: data = original.read()
        with open(new_config, 'w') as modified: modified.write("# GIT HASH: "+git_hash()+"\n" + data)

    @classmethod
    def from_config_file(cls, config_file, replace=None, overwrite_config_file= None):
        obj = super().from_config_file(config_file, replace, overwrite_config_file)
        if overwrite_config_file:
            obj.copy_config_file(overwrite_config_file, "overwrite_config.py")
        return obj


    @classmethod
    def _get_kwargs(cls, config, key = "evaluator"):
        cls._config_required(
            "algorithm",
            "problem_suite",
            "restoredir",
            "step",
            "logdir",
            "N_episodes",
            "name",
            "seed",
            "summarizer"
        )
        cls._config_defaults(
        restoredir = None,
            step = None,
            seed = 1001,
            N_episodes = 100,
            summarizer = lambda pop: dict(mean_fitness=np.mean(pop.fitness), mean = pop.mean, covariance = pop.covariance),
            name="evaluation"
        )

        kwargs = super()._get_kwargs(config, key)

        from learnedevolution.algorithm import Algorithm

        import tensorflow as tf
        tf.set_random_seed(kwargs['seed'])

        kwargs['algorithm'] = Algorithm.from_config(config, kwargs['algorithm']['key'])

        from learnedevolution.problems import ProblemSuite
        kwargs['problem_suite'] = ProblemSuite.from_config(config, kwargs['problem_suite']['key'])

        return kwargs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", help="log_dir of benchmark")
    parser.add_argument("step", help ="step to evaluate")
    parser.add_argument("--config_file", help = "evaluator config")

    args = parser.parse_args()

    replace = dict(
        logdir = args.log_dir,
        step = args.step,
        restoredir = os.path.join(args.log_dir,"saves",args.step)
    )

    evaluator = Evaluator.from_config_file(os.path.join(args.log_dir,'config.py'), replace=replace, overwrite_config_file = args.config_file)
    evaluator.run()
