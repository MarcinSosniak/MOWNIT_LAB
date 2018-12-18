import numpy

def sorted_eig_desc(w, v):
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:,idx]
    return w, v

def z_scores(a,axis=-1):
    b=numpy.array(a).swapaxes(axis,-1)
    mu = numpy.mean(b,axis=-1)[...,numpy.newaxis]
    sigma = numpy.std(b,axis=-1)[...,numpy.newaxis]
    return (b-mu)/sigma

def almost_zero(a, tolerance=0.00000001):
    return abs(a) < tolerance

def first_non_zero(v):
    for i, e in enumerate(v):
        if not almost_zero(e):
            return i

def make_zero_mean(X):
    X -= X.mean(axis=0)

def deflate(X, eigvec):
    #http://theory.stanford.edu/~tim/s15/l/l8.pdf
    X -= numpy.array([eigvec*numpy.dot(x, eigvec) for x in X])