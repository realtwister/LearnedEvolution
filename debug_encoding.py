from multiprocessing import Pool, Queue
import time;
import tqdm;
import numpy as np;
from progressbar import ProgressBar;
def worker(q):
    import time;
    import numpy as np;
    print("starting")

    s=0;
    while s <1:
        s += 0.001*np.random.rand();
        s = min(s,1);
        q.put(s);
        time.sleep(0.01)
    print("done")
    return 'hello';

qs = [Queue(1) for _ in range(10)];
p = Pool(4);
res = p.map(worker, qs)
print(res)

bar= ProgressBar(max_value=10);
timers =[0]*10;
while not res.ready():
    for i, q in enumerate(qs):
        if q.full():
            timers[i] = q.get();
    bar.update(np.sum(timers));

print(res.ready())
print("Completed")
