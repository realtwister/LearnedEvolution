import os
from .utils import confirm
import libtmux
from time import sleep
parsers = [];
def register_evaluate(subparsers):
    parser = subparsers.add_parser('evaluate')
    parser.set_defaults(func = main)

    parser.add_argument("log_dir", help="The directory to save/ log to")
    parser.add_argument("--steps", help="step to evaluate")
    parser.add_argument("--config_file", help="evaluator configuration")

    parser.add_argument("--session_name", help = "The session name to use (will be created if doesn't exist) (DEFAULT:learnedevolution)", default="learnedevolution")
    parser.add_argument("-y","--yes",dest="should_confirm", action="store_false", default=True)
    parser.add_argument("--workers", help="Number of workers", default = 4)

    parsers.append(parser)

def get_inactive_windows(windows):
    res = []
    for window in windows:
        if window.name == "bash":
            res.append(window)
    return res

def main(args):
    # check input
    log_dir = args.log_dir
    assert os.path.exists(log_dir)
    assert os.path.exists(os.path.join(log_dir, 'saves'))
    assert os.path.isfile(os.path.join(log_dir, 'config.py'))
    if args.config_file:
        assert os.path.isfile(args.config_file)

    # available steps
    available_steps = []
    for item in os.listdir(os.path.join(log_dir, 'saves')):
        item_path = os.path.join(log_dir,'saves',item)
        if os.path.isdir(item_path):
            try:
                step = int(item)
                available_steps.append(step)
            except:
                pass
    available_steps = sorted(available_steps)
    steps = []
    if args.steps is not None:
        for step in args.steps.split(","):
            try:
                step = int(step)
            except:
                raise ValueError("Step {} could not be interpreted as integer")
            assert step in available_steps
            steps.append(step)
    else:
        steps = available_steps

    workers = min(int(args.workers), len(steps))

    print("-------- Summary --------")
    print("log_dir: {}".format(os.path.abspath(log_dir)))
    print("steps: ({})".format(len(steps)))
    print("   ",steps)
    print("workers:", workers)
    print("-------------------------")
    if not confirm("Run the experiment?", args.should_confirm):
        exit()

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

    queue = list(steps)
    log_dir = os.path.abspath(log_dir)
    if args.config_file is not None:
        args.config_file = os.path.abspath(args.config_file)
    while True:
        for window in get_inactive_windows(session.windows):
            if len(queue) == 0:
                if len(session.windows) > 1:
                    window.kill_window()
                break;
            step = queue.pop()
            if args.config_file is not None:
                window.attached_pane.send_keys("python3 -m learnedevolution evaluate_step {} {} --config {}".format(
                log_dir,
                step,
                args.config_file,
                ))
            else:
                window.attached_pane.send_keys("python3 -m learnedevolution evaluate_step {} {}".format(
                log_dir,
                step,
                ))
            print("Running step", step, "on", window.id)
        if len(queue) == 0:
            break;
        sleep(1)
    print("All experiments have started or finished on session", session.name)
