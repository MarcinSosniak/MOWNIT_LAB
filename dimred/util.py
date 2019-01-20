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

def dataReader(filename):
    matrixRows=[]
    stringLines=[]
    matrix=[]
    with open("src/"+filename,"r") as file:
        stringLines=file.read().splitlines()
    for line in stringLines:
        row=[]
        rowString=line.split(",")
        for sNumber in rowString:
            if (sNumber=="?"):
                row.append(1)
            else:
                row.append(float(sNumber))
        matrix.append(row)
    #endfor
    print("\n\n\n\n\n\nfinished function\n\n\n\n\n\n\n")
    return matrix

def copyMatrix(X):
    outMatrix=[]
    for row in X:
        outMatrix.append(row.copy())
    return outMatrix


def getUncorrelatedRandomDataMatrix(rows,cols):
    return numpy.random.rand(rows,cols)

def getCorrelatedRandomDataMatrix(rows,cols):
    meanAndDev=[]
    for i in range(cols):
        mean=abs(numpy.random.normal(50.0,25.0))
        dev=abs(numpy.random.normal(10.0,5.0))
        meanAndDev.append({mean,dev})
    outMatrix=[]
    for i in range(rows):
        row=[]
        for mAD in meanAndDev:
            mean,dev=mAD
            row.append(abs(numpy.random.normal(mean,dev)))
        #endfor
        outMatrix.append(row)
    #endfor
    return  numpy.array(outMatrix)
