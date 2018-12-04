def register_benchmark(subparsers):
    parser = subparsers.add_parser('benchmark')
    parser.set_defaults(func = main)
    parser.add_argument("config_file", help = "path to the config file")
    parser.add_argument("log_dir", help = "directory to log to")


def main(args):
    from ..benchmark import Benchmark

    replace = dict(
        logdir = args.log_dir
    )

    benchmark = Benchmark.from_config_file(args.config_file, replace=replace)
    benchmark.run()
