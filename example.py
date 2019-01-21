from dimred.algo import *
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
    results = {
        "PCA_EVD" : pca_evd(X, L),
        "PCA_Power" : pca_power(X, L, 10, 0.0001),
        "PCA_SVD" : pca_svd(X, L),
        "PCA_Lanczos" : pca_lanczos(X, L),
    }

    for name, result in results.items():
        print('Transformation matrix for {}'.format(name))
        print(result.transformation_matrix)
        print('\n')
    print('\n')

    for name, result in results.items():
        print('Transformed data for {}'.format(name))
        print(result.transformed_data)
        print('\n')
    print('\n')

    for name, result in results.items():
        print('Cumulative energy for {}'.format(name))
        print(result.cumulative_energy)
        print('\n')
    print('\n')

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
        [0., 4., 0.],
        [0., 7., 0.]
        ])
    ex(X, 2)


def testFa():
    X = numpy.array([
        [1., 0., 0.],
        [2., 0., 0.],
        [3., 0., 0.],
        [0., 1., 0.],
        [0., 4., 0.],
        [0., 7., 0.]
        ])
    print(sklearnLibraryFA(X,70))



def testRead():
    X=dataReader("testData1.txt")
    for row in X:
        print(row)


def testGenerateCorrelatedRandomMatrix(x,y):
    X=getCorrelatedRandomDataMatrix(x,y)
    for row in X:
        print(row)

def testGetUncorrelatedRandomMatrix(x,y):
    X=getUncorrelatedRandomDataMatrix(x,y)
    for row in X:
        print(row)

def main():
    #testGenerateCorrelatedRandomMatrix(20,3)
    #print("\n")
    #testGetUncorrelatedRandomMatrix(20,3)
    #testRead()
    #ex1()
    ex2()
    #testFa()
main()