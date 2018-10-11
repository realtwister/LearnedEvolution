import os;
import importlib.util;
import shutil
from tqdm import tqdm;
from learnedevolution.utils import git_hash;


class Benchmark:
    def __init__(self, config_file, logdir, queue=None, progress=False):

        # Initialize everything in config
        config = self._config_is_valid(config_file);
        self._copy_config(config_file, logdir);
        self._p = self._check_parameters(config.parameters);
        self._algorithm = config.initiate_algorithm(logdir);
        self._train_suite, self._test_suite = config.initiate_problem_generator();
        loggers = config.initiate_logging(logdir);
        self._logdir = logdir;
        self._logger = dict(
            algorithm = loggers[0],
            problem = loggers[1]
        )
        self._queue = queue;
        self._progress = progress;

        self._seed();

    @staticmethod
    def _config_is_valid(config_file):
        assert os.path.isfile(config_file);
        spec = importlib.util.spec_from_file_location("config_file", config_file)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        assert hasattr(config_module, 'BenchmarkConfig');
        return config_module.BenchmarkConfig();

    @staticmethod
    def _copy_config(config_file,logdir):
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        new_config = os.path.join(logdir, 'config.py');
        with open(config_file, 'r') as original: data = original.read()
        with open(new_config, 'w') as modified: modified.write("# GIT HASH: "+git_hash()+"\n" + data)



    @staticmethod
    def _check_parameters(parameters):
        necessary = [
            'N_test','N_epoch','N_train',
            'seed_test', 'seed_train'
        ];
        missing = [];
        for n in necessary:
            if n not in parameters:
                missing +=[n];
        if len(missing) >0:
            raise ValueError("Missing parameters {} in config file".format(",".join(missing)));
        return parameters;

    def _pack_iter(self, iter, **kwargs):
        if self._progress:
            return tqdm(iter, **kwargs);
        else:
            return iter;


    def _seed(self):
        self._test_suite.seed(self._p['seed_test']);

        self._train_suite.seed(self._p['seed_train']);
        self._algorithm.seed(self._p['seed_train']);

    def _reset_test(self):
        self._test_suite.seed(self._p['seed_test']);

    def train(self, steps):
        self._algorithm.set_target_attr('learning', True)
        for problem in self._pack_iter(self._train_suite.iter(steps), total=steps, desc="Training", leave=False):
            self._algorithm.maximize(problem.fitness)

    def test(self, steps):
        self._algorithm.set_target_attr('learning', False);
        self._reset_test();
        for i, problem in self._pack_iter(enumerate(self._test_suite.iter(steps)), total=steps, desc="Testing", leave=False):
            if self._algorithm._steps< 2:
                self._logger['problem'].add_current('problem', i);
            self._algorithm._steps -=1;
            self._logger['algorithm'].record(suffix="BENCHMARK.deterministic."+str(i));
            mean, covariance = self._algorithm.maximize(problem.fitness, deterministic=True);

            self._algorithm._steps -=1;
            self._logger['algorithm'].record(suffix="BENCHMARK.explorative."+str(i));
            mean, covariance = self._algorithm.maximize(problem.fitness);

    def run(self):
        print("Logging to: {}".format(self._logdir))
        self.test(self._p['N_test']);
        for i in self._pack_iter(range(self._p['N_epoch']), desc="Epochs", leave=False):
            self.train(self._p['N_train']);
            self.test(self._p['N_test']);
            if self._queue is not None:
                self._queue.put(float(i+1)/self._p['N_epoch']);

    def close(self):
        self._algorithm.close();

    @property
    def test_suite(self):
        self._reset_test();
        return self._test_suite;

    @property
    def train_suite(self):
        return self._train_suite;











if __name__ == "__main__":
    name = input("Please enter benchmark name:\n");
    b = Benchmark("./config.py", "/tmp/thesis/single_benchmarks/"+name, progress=True);
    b.run();
