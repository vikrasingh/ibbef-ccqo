def ibbef(p,n,y,X,A,b,c,X0,k):
    """IBB+ with lb using recycled echelon form


    """
    import sys
    import numpy as np
    from scipy import linalg

    epsilon=sys.float_info.epsilon    
    pivtol=max(p,n)*epsilon*np.linalg.norm( A, ord=np.inf)  #  tol to check non-zero pivot for echelon form
    R, ipiv0 = linalg.qr(A, mode='r', pivoting=True)
    diagR=np.abs(np.diag(R))
    
    npiv0=sum( i >= pivtol for i in diagR)
    ipiv0=ipiv0[0:npiv0]
