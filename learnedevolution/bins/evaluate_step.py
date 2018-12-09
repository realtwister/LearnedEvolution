import os
parsers = [];
def register_evaluate_step(subparsers):
    parser = subparsers.add_parser('evaluate_step')
    parser.set_defaults(func = main)

    parser.add_argument("log_dir", help="The directory to save/ log to")
    parser.add_argument("step", help="step to evaluate")
    parser.add_argument("--config_file", help="evaluator configuration")

    parsers.append(parser)

def main(args):
    from ..evaluator import Evaluator
    replace = dict(
        logdir = args.log_dir,
        step = args.step,
        restoredir = os.path.join(args.log_dir,"saves",args.step)
    )

    evaluator = Evaluator.from_config_file(os.path.join(args.log_dir, 'config.py'),
        replace = replace,
        overwrite_config_file = args.config_file)
    evaluator.run()
