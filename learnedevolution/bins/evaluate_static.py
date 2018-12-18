import os
parsers = [];
def register_evaluate_static(subparsers):
    parser = subparsers.add_parser('evaluate_static')
    parser.set_defaults(func = main)

    parser.add_argument("log_dir", help="The directory to log to")
    parser.add_argument("config", help="base configuration file")
    parser.add_argument("--config_file", help="evaluator configuration")

    parsers.append(parser)

def main(args):
    from ..evaluator import Evaluator
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    replace = dict(
        logdir = args.log_dir,
        step = None,
    )

    evaluator = Evaluator.from_config_file(args.config,
        replace = replace,
        overwrite_config_file = args.config_file)
    evaluator.run()
    pr.disable()
    pr.dump_stats('profile.prof')
