import argparse;
import os
import re;
import numpy as np;

from itertools import product;
import fileinput;
from shutil import copyfile
from multiprocessing import Pool, Manager
from progressbar import ProgressBar;
import time;
import itertools as it;

from .benchmark import Benchmark;

VAR_REGEX = re.compile("<<VARIABLE:[a-zA-Z0-9_.,{} ()\+\"\'\[\]\-]+>>");
SPACE_REGEX = re.compile("{{[a-zA-Z0-9_., ()\"\'\[\]\+\-]+}}");
SPEC_VAR_REGEX = lambda var: "<<VARIABLE:"+var+"(|{{[a-zA-Z0-9_., \"\'{}()\[\]\+\-]+}})>>";

def confirm(question, should_confirm, before_fn = None):
    if not should_confirm:
        if before_fn is not None:
            before_fn();
        return True;
    while True:
        if before_fn is not None:
            before_fn();
        answer = input(question +" (y/n) ");
        if answer.lower() in ["y", "yes"]:
            return True;
        if answer.lower() in ["n", "no"]:
            return False;

def search_variable(line):
    found_vars = [];
    for var_found in VAR_REGEX.finditer(line):
        var = var_found.group()[11:-2];
        space_found = SPACE_REGEX.search(var);
        if space_found:
            var = var[:space_found.start()]
            space = space_found.group()[2:-2];
            found_vars.append((var,space));
        else:
            found_vars.append((var, None))
    return found_vars;

def find_variables_in_file(file_path):
    variables = dict();
    with open(file_path) as f:
        for line in f:
            for var, space in search_variable(line):
                if space is not None:
                    if var in variables and variables[var] is not None:
                        print("Space for variable {} defined multiple times".format(var));
                    else:

                        variables[var] = create_space(var, space);
                elif var not in variables:
                    variables[var] = None;
    for var in variables:
        while variables[var] is None:
            variables[var] = create_space(var ,input("Space for {} not specified. Please specify:".format(var)));
    return variables;

def create_space(var, space):
    try:
        res = eval(space);
    except Exception:
        print("Could not parse {} as a space".format(space));
        return None;

    if not hasattr(res, '__iter__'):
        print("Input \"{}\" is not iterable".format(space));
        return None;

    try:
        res = np.array(res);
    except Exception:
        print("Input {} not compatible with np.array".format(space));
        return None;
    return res;

def fill_config(config_file, keys, values):
    for line in fileinput.input(config_file, inplace = 1):
        for var, val in zip(keys, values):
            line = re.sub(SPEC_VAR_REGEX(var), str(val), line);
        print(line, end="");



def setup_run(template_file, run_dir, keys, values):
    os.mkdir(run_dir);
    config_file = os.path.join(run_dir, 'config.py');
    copyfile(template_file, config_file);
    fill_config(config_file, keys, values);
    Benchmark._config_is_valid(config_file);

def run_single(args):
    (run_dir, q)=args;
    import os;
    from wurlitzer import pipes
    from io import StringIO
    import sys;

    sys.stdout = open(os.path.join(run_dir,"stdout"), "a")
    sys.stderr =  open(os.path.join(run_dir,"stderr"), "a")
    try:
        from .benchmark import Benchmark;
        b = Benchmark(os.path.join(run_dir,'config.py'), run_dir, queue = q);
        b.run();
        b.close();
    except Exception(e):
        print(e);







def main():
    parser = argparse.ArgumentParser(prog="sweep")
    parser.add_argument('experiment_dir', metavar="SWEEP_DIR")
    parser.add_argument('--config', dest="config", metavar="CONFIG_FILE");
    parser.add_argument('-y', dest="should_confirm", action="store_false", default=True, help="Confirm all");
    parser.add_argument('--workers', dest="workers",type=int, default = 1)

    args = parser.parse_args();
    if os.path.exists(args.experiment_dir):
        if not confirm("Are you sure you want to overwrite it?", args.should_confirm):
            parser.error("The experiment dir already exists.");
    if args.config is not None:
        config = args.config;
    else:
        config = os.path.join(args.experiment_dir,'config.py');
    if not os.path.exists(config):
        parser.error("Config not found.")
    if not os.path.isfile(config):
        parser.error("Config is not a file.")

    variables = find_variables_in_file(config);

    if len(variables) == 0:
        if not confirm("No variables found do you want to continue?", args.should_confirm):
            exit();
        setup_run(config, args.experiment_dir,[],[]);
        run_single(args.experiment_dir);
        exit();

    keys, values = map(list,zip(*variables.items()))
    combinations = product(*values);
    len_combinations = np.product([len(v) for v in values]);
    print("Found variables:")
    for var,space in variables.items():
        print("{:>20}: {}".format(var,space));
    print("This will run for {} iterations.".format(len_combinations))
    if not confirm("Are you sure you want to do this?", args.should_confirm):
        print("Exiting...");
        exit();

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir);
    log_path = os.path.join(args.experiment_dir, 'log');

    with open(log_path, 'w') as log_file:
        log_file.write(','.join(['dir']+keys)+"\n");

    print(("{:12} "+"{:20} "*len(keys)).format("run",*keys));
    manager = Manager();
    map_args =[];
    for i,combination in enumerate(combinations):
        current_dir = os.path.join(args.experiment_dir, str(i));
        setup_run(config, current_dir, keys, combination);
        with open(log_path, 'a') as log_file:
            log_file.write(','.join([str(i)]+[ str(v) for v in combination])+"\n");
        print(("{:<12} "+"{:<20} "*len(keys)).format(i,*[str(c) for c in combination]));
        q = manager.Queue(1);
        map_args.append((current_dir,q));
    print("Running on {} processes...".format(args.workers));
    with Pool(args.workers, maxtasksperchild=1) as p:
        res = p.map_async(run_single, map_args);

        bar = ProgressBar(max_value=len_combinations);
        timers =[0]*len_combinations;
        last = 0;
        while not res.ready():
            for i, (_,q) in enumerate(map_args):
                if not q.empty():
                    timers[i] = q.get();
            if abs(last - np.sum(timers))>1e-2:
                bar.update(np.sum(timers));
                last = np.sum(timers);
            time.sleep(1);



if __name__ == "__main__":
    main();
