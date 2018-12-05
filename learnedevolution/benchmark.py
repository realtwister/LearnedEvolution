from .utils.parse_config import ParseConfig, config_factory
from .utils import git_hash
import os
import time

class Benchmark(ParseConfig):
    default_topic = "benchmark"
    def __init__(self,
        algorithm,
        problem_suite,
        should_save,
        logdir,
        seed,
        restoredir,
        N_episodes
        ):
        self.should_save = should_save
        self.logdir = logdir
        self.restoredir = restoredir
        self.N_episodes = N_episodes
        self.algorithm = algorithm
        self.problem_suite = problem_suite

        self.seed(seed);

        if self.restoredir is not None:
            self.restore();

        self.setup_logdir()

        self.i = 0
        self.status = "IDLE"

    def seed(self, seed = None):
        self.seed = seed;
        self.algorithm.seed(seed);
        self.problem_suite.seed(seed+1);

    def run(self):
        self.start = (self.i, time.clock())
        for i, problem in enumerate(self.problem_suite.iter(self.N_episodes)):
            if self.savedir is not None and self.should_save(i+1):
                self.status = "SAVING"
                self.print_status();
                self.algorithm.save(os.path.join(self.savedir,str(i+1)))
                self.status = "RUNNING"
                self.print_status();
            if self.i % 10 == 0:
                self.print_status();

            self.algorithm.maximize(problem.fitness);
            self.i +=1;

    def print_status(self):
        print(" "*70, end='\r');
        speed = (self.i-self.start[0])/(time.clock()-self.start[1])
        print(" ",self.i,":",self.status, end="");
        print(" speed:",speed, "it/s", end = "");
        print("\r", end = "")


    def setup_logdir(self):
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir);
        self.savedir = os.path.join(self.logdir,"saves")
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir);

    def copy_config_file(self, config_file, name="config.py"):
        new_config = os.path.join(self.logdir, name);
        with open(config_file, 'r') as original: data = original.read()
        with open(new_config, 'w') as modified: modified.write("# GIT HASH: "+git_hash()+"\n" + data)


    def restore(self):
        assert os.path.exists(self.restoredir), "Restore path does not exist"
        self.algorithm.restore(self.restoredir);

    @classmethod
    def from_config_file(cls, config_file, replace=None, overwrite_config_file= None):
        obj = super().from_config_file(config_file, replace, overwrite_config_file)
        obj.copy_config_file(config_file)
        if overwrite_config_file:
            obj.copy_config_file(overwrite_config_file, "overwrite_config.py")
        return obj

    @classmethod
    def _get_kwargs(cls, config, key = "benchmark"):
        cls._config_required(
            'should_save',
            'logdir',
            'restoredir',
            'N_episodes',
            'seed',
            'algorithm',
            'problem_suite'
        );

        cls._config_defaults(
            should_save = lambda x: x%1000==0,
            logdir = None,
            restoredir = None,
            N_episodes = -1,
            seed = None,
        )
        kwargs = super()._get_kwargs(config, key)

        from .algorithm import Algorithm

        kwargs['algorithm'] = Algorithm.from_config(config, kwargs['algorithm']['key'])

        from .problems import ProblemSuite
        kwargs['problem_suite'] = ProblemSuite.from_config(config, kwargs['problem_suite']['key'])

        return kwargs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help = "path to the config file")
    parser.add_argument("log_dir", help = "directory to log to")

    args = parser.parse_args()


    replace = dict(
        logdir = args.log_dir
    )

    benchmark = Benchmark.from_config_file(args.config_file, replace=replace)
    benchmark.run()
