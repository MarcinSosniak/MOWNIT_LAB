from dimred.model import LinearDimensionalityReduction
from dimred.solver import *
from dimred.util import *

import numpy

def sort_eig(w, v):
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:,idx]
    return w, v

def z_scores(a,axis=-1):
    b=numpy.array(a).swapaxes(axis,-1)
    mu = numpy.mean(b,axis=-1)[...,numpy.newaxis]
    sigma = numpy.std(b,axis=-1)[...,numpy.newaxis]
    return (b-mu)/sigma

def ex(X, L):
    result1 = LinearDimensionalityReduction(PCA_ExplicitCov(10, 0.0001), X, L)
    result2 = LinearDimensionalityReduction(PCA_IterativeSimple(10, 0.0001), X, L)
    print(result1.transformation_matrix)
    print(result2.transformation_matrix)
    print("\n")
    print(result1.transformed_data)
    print(result2.transformed_data)
    print("\n")
    print(result1.cumulative_energy)
    print(result2.cumulative_energy)
    print("\n")

def ex1():
    X = numpy.array([
        [1., 3.],
        [2., 5.],
        [3., 7.],
        [4., 9.]
        ])
    ex(X, 1)

def ex2():
    X = numpy.array([
        [1., 0., 0.],
        [2., 0., 0.],
        [3., 0., 0.],
        [0., 1., 0.],
        [0., 2., 0.],
        [0., 3., 0.]
        ])
    ex(X, 2)

def main():
    #ex1()
    ex2()

main()