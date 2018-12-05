from .utils import confirm
import re
import ast
from tempfile import TemporaryDirectory
from time import sleep

VAR_REGEX = re.compile("<<VARIABLE:[a-zA-Z0-9_.,{} ()\+\"\'\[\]\-]+>>");
SPACE_REGEX = re.compile("{{[a-zA-Z0-9_., ()\"\'\[\]\+\-]+}}");
SPEC_VAR_REGEX = lambda var: "<<VARIABLE:"+var+"(|{{[a-zA-Z0-9_., \"\'{}()\[\]\+\-]+}})>>";

parsers = [];

def register_sweep(subparsers):
    parser = subparsers.add_parser('sweep')
    parser.set_defaults(func = main)

    parser.add_argument("log_dir", help="The directory to save/ log to")
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("variable_dir", help="variables file")

    parser.add_argument("--session_name", help = "The session name to use (will be created if doesn't exist) (DEFAULT:learnedevolution)", default="learnedevolution")
    parser.add_argument("-y","--yes",dest="should_confirm", action="store_false", default=True)
    parser.add_argument("--workers", help="Number of workers", default = 4)
    parsers.append(parser)

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
    return variables;


def get_inactive_windows(windows):
    res = []
    for window in windows:
        if window.name == "bash":
            res.append(window)
    return res

def main(args):
    import libtmux
    import os
    parser = parsers[0]

    # Check arguments
    if os.path.exists(args.log_dir):
        if not confirm("Are you sure you want to overwrite it?", args.should_confirm):
            parser.error("The experiment dir already exists.");
    if args.config_file is not None:
        config = args.config_file;
    else:
        config = os.path.join(args.log_dir, "config.py")

    if not os.path.exists(config):
        parser.error("Configuration file not found")

    if not os.path.isfile(config):
        parser.error("Configuration is not a file")

    if not os.path.exists(args.variable_dir):
        parser.error("variable_dir should exist")


    # Find the variables in the config file
    variables = find_variables_in_file(config)

    # Select variable files with appropriate variables
    variable_files = [];
    for f in os.listdir(args.variable_dir):
        f_path = os.path.join(args.variable_dir, f)
        if os.path.isfile(f_path) and f[-4:] == ".var":
            with open(f_path,'r') as of:
                contents = eval(of.read())
            for v in variables:
                if v not in contents:
                    break;
            else:
                variable_files.append(f)

    workers = min(int(args.workers), len(variable_files))

    print("-------- Summary --------")
    print("Variables: ({})".format( len(variables)))
    for v in variables:
        print("   -",v)
    print("Configurations: ({})".format(len(variable_files)))
    for f in variable_files:
        print("   -",f)
    print("Workers:", workers)
    print("Logging to:", os.path.abspath(args.log_dir))
    print("-------------------------")
    if not confirm("Run the experiment?", args.should_confirm):
        exit()
    # Create configs in temporary directory
    tempdir =TemporaryDirectory()

    for f_name in variable_files:
        f_path = os.path.join(args.variable_dir, f_name)
        with open(f_path,'r') as of:
            values = eval(of.read())
        new_config_path = os.path.join(tempdir.name, f_name[:-4]+".py")
        with open(config) as original:
            with open(new_config_path,'a') as new:
                for line in original:
                    for var in variables:
                        line = re.sub(SPEC_VAR_REGEX(var), str(values[var]), line);
                    new.write(line)


    # run the sessions
    tmux = libtmux.Server()

    # Select tmux session
    if tmux.has_session(args.session_name):
        session = tmux.find_where({ "session_name": args.session_name })
        if not confirm("Session already exists. Should I continue?", args.should_confirm):
            exit()
    else:
        session = tmux.new_session(args.session_name)

    # clean idle windows
    for window in get_inactive_windows(session.windows)[:-1]:
        window.kill_window()

    # Setup windows
    windows = []
    for i in range(workers- len(session.windows)):
        window = session.new_window()
        windows.append(window)

    queue = list(variable_files)

    while True:
        for window in get_inactive_windows(session.windows):
            if len(queue) == 0:
                if len(session.windows) > 1:
                    window.kill_window()
                break;
            f_name = queue.pop()
            f_name = f_name[:-4] #Remove .var extension
            config_path = os.path.join(tempdir.name, f_name+".py")
            current_dir = os.path.join(args.log_dir, f_name)
            window.attached_pane.send_keys("python3 -m learnedevolution benchmark {} {}".format(
            config_path,
            current_dir
            ))
            print("Running configuration", f_name, "on", window.id)
        if len(queue) == 0:
            break;
        sleep(1)
    print("All experiments have started or finished on session", session.name)
    print("Waiting before clearing temporary directory")
    sleep(100)
    tempdir.cleanup()
