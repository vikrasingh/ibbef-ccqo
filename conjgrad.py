# conjgrad.py

def cg(n ,A ,d ,Q ,D ,neig ,x ,epsx=1e-9 ,maxDx=1e18 ,epsg=1e-9 ,nMax=500 ,isSoftStop=1 ,isTF=0 ,targetF=0 ):
    """ Conjugate gradient to min a convex quadratic function.
        
        Input:
        n: dim. of the problem
        A: Hessian matrix
        d: linear term of size 1xn
        Q: orthonormal matrix of size nxneig from spectral decomposition of A=QDQ'
        D: eigenvalues of A in ascending order of size 1xneig
        neig: no. of non zero eigenvalues of A
        x: starting point for the descent search, size 1xn

        idExit= 0  (best)by sufficient cond. i.e. |g_i|<eps for i=1...n
              = 1  the value of x-step size is small enough, So x is not updated.
              = 2  the value of x-step size gets bigger the max. value allowed. So x is not updated. 
              = 3  x is not updated for other reasons. 
              = 4  (worst)hard stop by max. iter. allowed
    """
    import numpy as np
    
    epsgg=epsg*epsg  # initialize constants

    nfAdd=0    # no. of f calls
    isNew=0    # overall update of x 
    niter=0    # no. of iter
    niters=0
    idExit=3   # initialization
    n2=100     # if isTF is not 0, check possible stop every 100 iter.
    epsxScaled=epsx/(1+A[1,1])    
    toStop=0  
    uplimit=0.5*n
    g1=np.zeros((n,1)) # initialization

    def exactLineSearch(x,g,h):
        """  search direction at x. May not be g-related. May not be descent.

           Assume convexity
        """        

        xnew, isDxSmall, isHhSmall, Ah, hAh = x, 0, 0, np.zeros((n,1)), 0

        hh=-(h.T @ g)
        #print(hh)
        # if hh<0, x is already optimal along h
        if hh<=epsgg: # no update. When hh=0, h=0, so optimal and stop
            isHhSmall=1
            return isDxSmall, isHhSmall, xnew, Ah, hAh
        else:
            if neig<(n/2):
                Ah=Q @ ( D*( Q.T @ h ) )
            else:
                Ah=A @ h

            hAh=h.T @ Ah   

        tstar=(hh/hAh)*h

        # check possible exits
        aa=max( abs(tstar) )
        if aa<=epsxScaled:
            isDxSmall=1
            return isDxSmall, isHhSmall, xnew, Ah, hAh
        
        xnew=x+tstar
        #print("xnew:",xnew)
        return isDxSmall, isHhSmall, xnew, Ah, hAh

    while toStop == 0:

        isUpdate=0  # update of 
        if neig < uplimit:
            g1= Q @ ( D*( Q.T @ x ) ) 
        else:
            g1=A @ x 
        
        #print(g1, d)
        g=g1 + d
        #print(g)
        nfAdd=nfAdd+1
        if np.max(np.abs(g))<=epsg:
            idExit=0
            xnew=x
            fxnew=0.5*(x.T @ g1) + d.T @ x
            nfAdd=nfAdd+1
            niters=niter
            return isNew, xnew, fxnew, niters, idExit

        if isTF!=0:
            if niter % n2==0:
                fxnew=0.5*(x.T @ g1) + d.T @ x
                nfAdd=nfAdd+1
                if fxnew<=(targetF+10**(-15)):
                    idExit=0
                    xnew=x
                    niters=niter
                    return isNew, xnew, fxnew, niters, idExit
            
        if niter==0: # S-D step
            h=-g       
        else:       # normal C-G step
            beta=(g.T @ Ah0)/h0Ah0
            h=-g + beta*h0
 
        # debug
        #print("iter:",niter)
        #print("x:",x)
        #print("g:",g)
        #print("h:",h)
        isDxSmall,isHhSmall,x,Ah0,h0Ah0=exactLineSearch(x,g,h)
        
        niter=niter+1
        if isHhSmall==1:
            idExit, toStop=0, 1 
            break

        if isDxSmall==1:
            idExit, toStop=0, 1 
            break

        isUpdate=1
        isNew=1
        h0=h

        if isSoftStop==0 and niter>=nMax:
            idExit, toStop=5, 1
            break

        if isUpdate==0:
            idExit, toStop=3, 1
            break

    # end while toStop=0        
    
    xnew=x
    if neig < uplimit:
        Qx=Q.T @ x
        fxnew=0.5*(Qx.T @ (D*Qx)) + d.T @ x
    else:
        g1=A @ x
        fxnew=0.5*(x.T @ g1) + d.T @ x

    nfAdd=nfAdd+1
    niters=niter

# end cg()
        




                 

        