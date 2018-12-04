from argparse import ArgumentParser
from .bins.benchmark import register_benchmark
from .bins.sweep import register_sweep
parser = ArgumentParser(prog = "learnedevolution")
subparsers = parser.add_subparsers()

# register arguments
register_sweep(subparsers)
register_benchmark(subparsers)

args = parser.parse_args()
if not hasattr(args, "func"):
    parser.print_help()
else:
    args.func(args)
