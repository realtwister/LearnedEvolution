from .utils.parse_config import ParseConfig, config_factory
import os

class Benchmark(ParseConfig):
    def __init__(self,
        algorithm,
        problem_suite,
        should_save,
        savedir,
        seed,
        restoredir,
        N_episodes
        ):
        self.should_save = should_save
        self.savedir = savedir
        self.restoredir = restoredir
        self.N_episodes = N_episodes
        self.algorithm = algorithm
        self.problem_suite = problem_suite

        self.seed(seed);

        if self.restoredir is not None:
            self.restore();

        self.setup_savedir()

        self.i = 0
        self.status = "IDLE"

    def seed(self, seed = None):
        self.seed = seed;
        self.algorithm.seed(seed);
        self.problem_suite.seed(seed+1);

    def run(self):
        for i, problem in enumerate(self.problem_suite.iter(self.N_episodes)):
            if self.savedir is not None and self.should_save(i):
                self.status = "SAVING"
                self.print_status();
                self.algorithm.save(os.path.join(self.savedir,str(i)))
                self.status = "RUNNING"
                self.print_status();
            if self.i % 10 == 0:
                self.print_status();

            self.algorithm.maximize(problem.fitness);
            self.i +=1;

    def print_status(self):
        print(" "*120, end='\r');
        print(" ",self.i,":",self.status, end="\r");


    def setup_savedir(self):
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir);

    def restore(self):
        assert os.path.exists(self.restoredir), "Restore path does not exist"
        self.algorithm.restore(self.restoredir);




    @classmethod
    def _get_kwargs(cls, config, key = ""):
        cls._config_required(
            'should_save',
            'savedir',
            'restoredir',
            'N_episodes',
            'seed',
            'algorithm',
            'problem_suite'
        );

        cls._config_defaults(
            should_save = lambda x: x%1000==0,
            savedir = None,
            restoredir = None,
            N_episodes = -1,
            seed = None,
        )
        kwargs = super()._get_kwargs(config, key)

        from learnedevolution.algorithm import Algorithm

        kwargs['algorithm'] = Algorithm.from_config(config, kwargs['algorithm']['key'])

        from learnedevolution.problems import ProblemSuite
        kwargs['problem_suite'] = ProblemSuite.from_config(config, kwargs['problem_suite']['key'])

        return kwargs
