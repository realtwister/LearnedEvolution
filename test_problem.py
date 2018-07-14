import learnedevolution as le;

import numpy as np;

problem = le.problems.Sphere([0,0],0.5);
print(problem);
print(problem._params['a']);
print(problem.fitness(np.array([[1,1],[0,0]])));


problem_generator = le.problems.Rosenbrock.generator(dimension = 2);

for problem in problem_generator.iter():
    print(problem);
    break;


from scipy.optimize import fmin;

print(fmin(problem.fitness,np.array([-22,70])));
