from argparse import ArgumentParser
from .bins.benchmark import register_benchmark
from .bins.sweep import register_sweep
from .bins.evaluate import register_evaluate
from .bins.evaluate_step import register_evaluate_step
from .bins.evaluate_static import register_evaluate_static
from .bins.evaluate_sweep import register_evaluate_sweep
parser = ArgumentParser(prog = "learnedevolution")
subparsers = parser.add_subparsers()

# register arguments
register_sweep(subparsers)
register_benchmark(subparsers)
register_evaluate(subparsers)
register_evaluate_step(subparsers)
register_evaluate_static(subparsers)
register_evaluate_sweep(subparsers)

args = parser.parse_args()
if not hasattr(args, "func"):
    parser.print_help()
else:
    args.func(args)
