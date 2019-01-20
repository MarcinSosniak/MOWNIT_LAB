import numpy
from . import util
import sklearn.decomposition
import copy

class PCA_SVD:
    '''
    Given n x n data matrix X and the final number of dimensions p:
        - performs Singular Value Decomposition on X
        - extracts p eigenvectors of C = X^T X from the result of SVD
        - transforms the data
    '''

    def __init__(self, max_iter, tolerance):
        self.max_iter = max_iter
        self.tolerance = tolerance

    def apply(self, model):
        X = model.data
        L = model.dims

        w, V = self._cov_eig_sorted(model);

        W = numpy.transpose(V[0:L])
        #Z = z_scores(X) #???
        T = numpy.matmul(X, W)

        model.transformation_matrix = W
        model.transformed_data = T
        cumsum = numpy.cumsum(w)
        model.cumulative_energy = cumsum[L-1] / cumsum[-1]

    def _cov_eig_sorted(self, model):
        X = copy.deepcopy(model.data)
        L = model.dims
        # X must be zero mean
        util.make_zero_mean(X)
        u, s, vh = numpy.linalg.svd(X)
        # s contains squared eigenvalues (from the highest)
        # vh contains eigenvectors in rows
        return numpy.square(s), vh

class PCA_EVD:
    '''
    Given n x n data matrix X and the final number of dimensions p:
        - directly computes the covariance matrix C of X (C = X^T X)
        - performs Eigen Value Decomposition on C
        - extracts p eigenvectors of C
        - transforms the data
    '''

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


class PCA_Power:
    '''
    Given n x n data matrix X and the final number of dimensions p:
        - p times:
            - computes the eigenvector v associated with the highest eigenvalue
              using power iteration algorithm, without explicitly computing
              the covariance matrix of X
            - performs deflation of X using v
        - transforms the data

    Since the process doesn't compute all eigenvalues the cumulative energy
    of the reduction is only the lower bound.
    '''

    def __init__(self, max_iter, tolerance):
        self.max_iter = max_iter
        self.tolerance = tolerance

    def apply(self, model):
        L = model.dims
        X = copy.deepcopy(model.data)
        util.make_zero_mean(X)
        W = []
        w = []

        for i in range(L):
            eigval, eigvec = self._largest_cov_eig(X)
            W += [eigvec]
            w += [eigval]
            util.deflate(X, eigvec)

        W = numpy.transpose(numpy.array(W))
        T = numpy.matmul(model.data, W)
        model.transformation_matrix = W
        model.transformed_data = T

        # we can assume all other eigenvalues are less than the smallest one
        total = sum(w)
        smallest = w[-1]
        left = X.shape[1] - L
        max_possible = total + smallest * left
        model.cumulative_energy = total / max_possible

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

class PCA_Lanczos:
    '''
    Given n x n data matrix X and the final number of dimensions p:
        - perform the Lanczos Method to find min(n, p*2) entries in Q
        - compute T, eigendecomposition of T
        - compute eigenvectors of cov(X)
        - return them in order of decreasing eigenvalues
    '''

    def __init__(self, max_iter, tolerance):
        self.max_iter = max_iter
        self.tolerance = tolerance

    def apply(self, model):
        X = copy.deepcopy(model.data)
        L = model.dims

        # X must be zero mean
        util.make_zero_mean(X)
        w, V = self._cov_eig_sorted(X, L);

        W = V[:, 0:L]
        #Z = z_scores(X) #???
        T = numpy.matmul(model.data, W)

        model.transformation_matrix = W
        model.transformed_data = T

        # we can assume all other eigenvalues are less than the smallest one
        total = sum(w)
        smallest = w[-1]
        left = X.shape[1] - L
        max_possible = total + smallest * left
        model.cumulative_energy = total / max_possible

    # requires mean zero matrix
    def _cov_eig_sorted(self, X, L):
        # https://sites.math.washington.edu/~morrow/498_13/eigenvalues3.pdf
        # http://www.physics.drexel.edu/~bob/Term_Reports/Hoppe_02.pdf

        rows = X.shape[0]
        cols = X.shape[1]

        # we compute 2*L because of the error of lanczos method
        # because of that it's better if L << n
        L *= 2
        if L > cols:
            L = cols

        def Av(v):
            s = numpy.zeros(cols)
            for row_num in range(rows):
                row = X[row_num]
                s += row * numpy.dot(row, v)
            return s

        def reorthogonalize_once(z, q, j):
            for i in range(1, j):
                z -= q[i] * numpy.dot(z, q[i])

        def reorthogonalize(z, q, j):
            reorthogonalize_once(z, q, j)
            reorthogonalize_once(z, q, j)

        q_1 = numpy.array(numpy.random.rand(cols))
        alpha = numpy.zeros(L+1)
        beta = numpy.zeros(L+1)
        beta[0] = numpy.linalg.norm(q_1)
        q = []
        q.append(numpy.zeros(cols))
        q.append(q_1 / beta[0])

        for j in range(1, L+1):
            z = Av(q[j]) - beta[j-1] * q[j-1]
            alpha[j] = numpy.dot(q[j], z)
            z -= alpha[j] * q[j]
            reorthogonalize(z, q, j)
            beta[j] = numpy.linalg.norm(z)
            q.append(z / beta[j])

        T = numpy.diag(alpha[1:], 0) + numpy.diag(beta[1:-1], -1) + numpy.diag(beta[1:-1], 1)

        Q = numpy.transpose(numpy.array(q[1:-1]))
        w, V = numpy.linalg.eig(T)
        V = numpy.matmul(Q, V)

        return util.sorted_eig_desc(w, V)


def sklearnLibraryFA(X,tolerance):
    FAnaliser=sklearn.decomposition.FactorAnalysis(tol=1-tolerance,copy=True,max_iter=1000)
    out=FAnaliser.fit_transform(numpy.array(X))
    return out