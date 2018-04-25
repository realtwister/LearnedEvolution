import learnedevolution as le;

import numpy as np;

problem = le.problems.Sphere([0,0],0.5);
print(problem);
print(problem._params['a']);
print(problem.fitness(np.array([[1,1],[0,0]])));


problem_generator = le.problems.Sphere.generator(dimension = 3);

for problem in problem_generator.iter():
    print(problem);
    break;

i = 0;
for problem in problem_generator.iter(100):
    i+=1;

print(i);
