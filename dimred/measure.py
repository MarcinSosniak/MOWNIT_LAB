from dimred.algo import *
from dimred.util import *
import time

def donothing():
    a=1


def runWithTimeAndWriteDown(fname,filename,fun, *args):
    startTime = time.time()
    try:
        outcome = fun(*args)
        endTime = time.time()
        with open(filename, "a") as file:
            file.write(fname + " test time:" + str(endTime - startTime) + " dim: " + str(
                outcome.dims) + " cumulative energy: " + str(outcome.cumulative_energy) + "\n")
    except Exception:
        donothing()





def testAll(filename,X,tolerance=0.9,maxiter=2000):
    row, col = X.shape
    for i in range(2,col-1):
        runWithTimeAndWriteDown("PCA SVD",filename,pca_svd,X,i)
        runWithTimeAndWriteDown("PCA EVD",filename,pca_evd,X,i)
        runWithTimeAndWriteDown("PCA POWER",filename,pca_power,X,i,maxiter,tolerance)
        runWithTimeAndWriteDown("PCA LANCHOS",filename,pca_lanczos,X,i)
        runWithTimeAndWriteDown("PCA NIPALS",filename,pca_nipals,X,i,maxiter,tolerance)