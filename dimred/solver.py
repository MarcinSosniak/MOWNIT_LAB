import numpy
from . import util
import copy

class PCA_ExplicitCov:
    def __init__(self, max_iter, tolerance):
        self.max_iter = max_iter
        self.tolerance = tolerance

    def apply(self, model):
        X = model.data
        L = model.dims

        w, V = self._cov_eig_sorted(model);

        W = V[:, 0:L]
        #Z = z_scores(X) #???
        T = numpy.matmul(X, W)

        model.transformation_matrix = W
        model.transformed_data = T
        cumsum = numpy.cumsum(w)
        model.cumulative_energy = cumsum[L-1] / cumsum[-1]

    def _cov_eig_sorted(self, model):
        X = model.data
        L = model.dims
        C = numpy.cov(X, rowvar=False)
        w, V = util.sorted_eig_desc(*numpy.linalg.eig(C))
        return w, V


class PCA_IterativeSimple:
    def __init__(self, max_iter, tolerance):
        self.max_iter = max_iter
        self.tolerance = tolerance

    def apply(self, model):
        L = model.dims
        X = copy.deepcopy(model.data)
        util.make_zero_mean(X)
        W = []

        for i in range(L):
            eigval, eigvec = self._largest_cov_eig(X)
            W += [eigvec]
            util.deflate(X, eigvec)

        W = numpy.transpose(numpy.array(W))
        T = numpy.matmul(model.data, W)
        model.transformation_matrix = W
        model.transformed_data = T
        model.cumulative_energy = None

    # requires mean zero matrix
    def _largest_cov_eig(self, X):
        rows = X.shape[0]
        cols = X.shape[1]

        r = numpy.array(numpy.random.rand(cols))
        r /= numpy.linalg.norm(r)
        for i in range(self.max_iter):
            s = numpy.zeros(cols)
            for row_num in range(rows):
                row = X[row_num]
                s += row * numpy.dot(row, r)

            eigenvalue = numpy.dot(r, s)
            error = numpy.linalg.norm(eigenvalue * r - s)
            r = s / numpy.linalg.norm(s)

            if error < self.tolerance: break

        return eigenvalue, r
