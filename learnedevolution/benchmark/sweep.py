import argparse;
import os
import re;
import numpy as np;

from itertools import product;
import fileinput;
from shutil import copyfile

VAR_REGEX = re.compile("<<VARIABLE:[a-zA-Z0-9_.,{}()\[\]\-]+>>");
SPACE_REGEX = re.compile("{{[a-zA-Z0-9_.,()\[\]\-]+}}");
SPEC_VAR_REGEX = lambda var: "<<VARIABLE:"+var+"(|{{[a-zA-Z0-9_.,{}()\[\]\-]+}})>>";

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

def run_single(run_dir):
    from .benchmark import Benchmark;
    b = Benchmark(os.path.join(run_dir,'config.py'), run_dir);
    b.run();







def main():
    parser = argparse.ArgumentParser(prog="sweep")
    parser.add_argument('experiment_dir', metavar="SWEEP_DIR")
    parser.add_argument('--config', dest="config", metavar="CONFIG_FILE", required= True);
    parser.add_argument('-y', dest="should_confirm", action="store_false", default=True, help="Confirm all");

    args = parser.parse_args();
    if os.path.exists(args.experiment_dir):
        parser.error("The experiment dir already exists.");
    if not os.path.exists(args.config):
        parser.error("Config not found.")
    if not os.path.isfile(args.config):
        parser.error("Config is not a file.")

    variables = find_variables_in_file(args.config);

    if len(variables) == 0:
        if not confirm("No variables found do you want to continue?", args.should_confirm):
            exit();
        setup_run(args.config, args.experiment_dir,[],[]);
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

    os.makedirs(args.experiment_dir);
    log_path = os.path.join(args.experiment_dir, 'log');

    with open(log_path, 'w') as log_file:
        log_file.write(', '.join(['dir']+keys)+"\n");

    print(("{:12} "+"{:20} "*len(keys)).format("run",*keys));
    for i,combination in enumerate(combinations):
        current_dir = os.path.join(args.experiment_dir, str(i));
        setup_run(args.config, current_dir, keys, combination);
        with open(log_path, 'a') as log_file:
            log_file.write(', '.join([str(i)]+[ str(v) for v in combination])+"\n");
        print(("{:<12} "+"{:<20} "*len(keys)).format(i,*combination));
        run_single(current_dir);

if __name__ == "__main__":
    main();
