from dimred.algo import *
from dimred.util import *
import time

def donothing():
    a=1


def runWithTimeAndWriteDown(fname,filename,fun, *args):
    startTime = time.time()
    outcome = fun(*args)
    endTime = time.time()
    with open(filename, "a") as file:
        file.write(fname + " test time:" + str(endTime - startTime) + " dim: " + str(outcome.dims) + " cumulative energy: " + str(outcome.cumulative_energy) + "\n")






def testAllSpeed(filename,rows=1500,l=5,n=25):
    tolerance = 0.9
    maxiter = 2000
    splitLiner="--------------------------------------------------------------------------------"
    X=getCorrelatedRandomDataMatrix(rows,n)
    with open(filename, "a") as file:
        file.write("\n\n"+splitLiner + "\n\n\nspeedTest with rows="+ str(rows)+ " desiredCols=" + str(l) + " cols="+ str(n) +"\n\n\n" + splitLiner + "\n\n\n")
    runWithTimeAndWriteDown("PCA SVD", filename, pca_svd, X, l)
    print("PCA SVD finished")
    runWithTimeAndWriteDown("PCA EVD", filename, pca_evd, X, l)
    print("PCA EVD finished")
    runWithTimeAndWriteDown("PCA POWER", filename, pca_power, X, l, maxiter, tolerance)
    print("PCA POWER finished")
    runWithTimeAndWriteDown("PCA LANCHOS", filename, pca_lanczos, X, l)
    print("PCA LANCHOS finished")
    runWithTimeAndWriteDown("PCA NIPALS", filename, pca_nipals, X, l, maxiter, tolerance)
    print("PCA NIPALS finished")
    runWithTimeAndWriteDown("FA", filename, sklearnLibraryFA, X,l)
    print("FA finished")


def testSpeedAllUncorrelated(filename,rows=1500,l=5,n=25):
    tolerance = 0.9
    maxiter = 2000
    splitLiner="--------------------------------------------------------------------------------"
    X=getUncorrelatedRandomDataMatrix(rows,n)
    with open(filename, "a") as file:
        file.write("\n\n"+splitLiner + "\n\n\nUncorrelated speedTest with rows="+ str(rows)+ " desiredCols=" + str(l) + " cols="+ str(n) +"\n\n\n" + splitLiner + "\n\n\n")
    runWithTimeAndWriteDown("PCA SVD", filename, pca_svd, X, l)
    print("PCA SVD finished")
    runWithTimeAndWriteDown("PCA EVD", filename, pca_evd, X, l)
    print("PCA EVD finished")
    runWithTimeAndWriteDown("PCA POWER", filename, pca_power, X, l, maxiter, tolerance)
    print("PCA POWER finished")
    runWithTimeAndWriteDown("PCA LANCHOS", filename, pca_lanczos, X, l)
    print("PCA LANCHOS finished")
    runWithTimeAndWriteDown("PCA NIPALS", filename, pca_nipals, X, l, maxiter, tolerance)
    print("PCA NIPALS finished")
    runWithTimeAndWriteDown("FA", filename, sklearnLibraryFA, X,l)
    print("FA finished")




def testAll(filename,X,tolerance=0.9,maxiter=2000):
    row, col = X.shape
    for i in range(1,col-1):
        runWithTimeAndWriteDown("PCA SVD",filename,pca_svd,X,i)
        runWithTimeAndWriteDown("PCA EVD",filename,pca_evd,X,i)
        runWithTimeAndWriteDown("PCA POWER",filename,pca_power,X,i,maxiter,tolerance)
        runWithTimeAndWriteDown("PCA LANCHOS",filename,pca_lanczos,X,i)
        runWithTimeAndWriteDown("PCA NIPALS",filename,pca_nipals,X,i,maxiter,tolerance)
        runWithTimeAndWriteDown("FA",filename,sklearnLibraryFA,X,i)