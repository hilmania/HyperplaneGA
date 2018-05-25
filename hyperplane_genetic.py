# Created by : Hilman Ibnu Assiddiq

import matplotlib.pyplot as plt
import numpy as np
import math

N_MARGIN = 3
CROSS_RATE = 0.7
MUTATE_RATE = 0.3
POP_SIZE = 500
N_GENERATIONS = 1000

class GA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, ):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size

        self.data = np.random.rand(3,3)

        self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(pop_size)])

    def translateDNA(self, DNA, data):
        w1 = np.empty_like(DNA, dtype=np.float64)
        w2 = np.empty_like(DNA, dtype=np.float64)
        b = np.empty_like(DNA, dtype=np.float64)

        for i, d in enumerate(DNA):
            point = data[d]
            w1[i] = point[i, 0]
            w2[i] = point[i, 1]
            b[i] = point[i, 2] 
        return w1, w2, b

    def count_fitness(self, w1, w2, b, x, y, x1, y1):
        fitness = np.empty((x.shape[0]+x1.shape[0],), dtype=np.float64)

        for i in range(0, x.shape[0]):
            fitness[i] = (w1[i]*x[i] + w2[i]*y[i] + b[i]) / math.sqrt(w1[i]**2 + w2[i]**2)
    
        for j in range(0, x1.shape[0]):
            fitness[j+x.shape[0]] = abs((w1[j]*x1[j] + w2[j]*y1[j] + b[j]) * -1) / math.sqrt(w1[j]**2 + w2[j]**2)
    
        return np.amin(fitness)

    def select(self, fitness):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return self.pop[idx]

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)                       
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)   
            keep_point = parent[~cross_points]                                       
            swap_point = pop[i_, np.isin(pop[i_].ravel(), keep_point, invert=True)]
            parent[:] = np.concatenate((keep_point, swap_point))
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(0, self.DNA_size)
                swapA, swapB = child[point], child[swap_point]
                child[point], child[swap_point] = swapB, swapA
        return child

    def evolve(self, fitness):
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop: 
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop

if __name__ == '__main__':
    x, y = np.loadtxt('dataA.txt', delimiter=',', unpack=True)
    x1, y1 = np.loadtxt('dataB.txt', delimiter=',', unpack=True)    

    ga = GA(DNA_size=N_MARGIN, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)
    for generation in range(N_GENERATIONS):
        w1, w2, b = ga.translateDNA(ga.pop, ga.data)
        fitness = ga.count_fitness(w1, w2, b, x, y, x1, y1)
        ga.evolve(fitness)
        best_idx = np.argmax(fitness)
        print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx],)