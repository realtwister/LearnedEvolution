from learnedevolution.benchmark.benchmark import Benchmark;

name = "4";
b = Benchmark("./benchmark_config.py", "/tmp/thesis/profiler/"+name, progress=True);
b.run();
