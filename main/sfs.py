def main(p,n,y,X,k,box,xrelax):
    """ Sequential Feature Swapping for BSS
    
       X is the design matrix of size n x p
       y is the response vector of size n x 1
       k is the no. of predictors to choose from total p predictors
       I0 is the initial support, i.e. the predictors already fixed to be in the model
       box  is the current box selected
       xrelax is the ols solution 
    """

    import numpy as np
    from scipy import linalg
    import getklargest as gkl

    stop_flag = 0 
    num_iter = 0

    I0 = []
    ctr = 0
    for i in range(p):
        if box[i]!=0:
            I0 = np.concatenate((I0,[i]))
            ctr += 1

    I0 = np.array(I0,dtype=int)  
    
    #absxrelax = np.reshape(absxrelax,(1,-1))[0] # make it a 1d array
    local_supp, _ = gkl.main(np.abs(xrelax[I0]),k)
    I = I0[local_supp]  # current support of k predictors
    fitted_model = linalg.lstsq(X[:,I], y) # least square fit
    xhat = fitted_model[0]
    fxI = np.linalg.norm(y - X[:,I] @ xhat) ** 2 
    Ic = setdiff(np.array(range(p)),I)

    while True:
        num_iter += 1

        # minimize the gain after dropping one predictor
        fxIminusi, istar = leastSigniI(k,y,X,I,fxI)

        # maximize the reduction after adding one predictor
        Iminusi = setdiff(I,istar)
        fxIplusj, jstar = mostSigniIc(p,k,y,X,Iminusi,Ic,fxIminusi)

        # update the current support if possible
        if fxI > fxIplusj:
            I = setunion(Iminusi,jstar)  # new support
            Ic = setunion( setdiff(Ic,jstar),istar )
            fxI = fxIplusj

        else:
            break
    
    # fit the data again for the final output
    fitted_model = linalg.lstsq(X[:,I], y) # least square fit
    xhat = fitted_model[0]
    fstar = np.linalg.norm(y - X[:,I] @ xhat) ** 2 
    xstar = np.zeros((p,1))
    xstar[I] = xhat
    return xstar, fstar

#=====================================================================================
def leastSigniI(ki,y,X,I,fxI):
    """ to find the least significant predictor w.r.t. the set I

       return x* such that S(x*)=min( f(X-xj)-f(X) ) for xj in I
       ki is the no. of predictors in I
    """
    import numpy as np
    from scipy import linalg

    if ki == 1:
        istar = I
        fstar = y.T @ y
        return fstar, istar
    
    pS = np.inf  # initialize
    for i in range(ki):
        Imini = setdiff(I,I[i])  # discard one predictor at a time from I
        fitted_model = linalg.lstsq(X[:,Imini], y) # least square fit
        xtemp = fitted_model[0]
        fxImini = np.linalg.norm(y - X[:,Imini] @ xtemp) ** 2 
        S = fxImini - fxI
        if S < pS:
            pS = S
            istar = I[i]
            fstar = fxImini
        
    return fstar, istar

#=====================================================================================
def mostSigniIc(p,ki,y,X,I,Ic,fxI):
    """ to find the most significant predictor w.r.t. the set I complement
       
       return x* such that S(x*)=max( f(I) - f(I+xj) ) for xj in Ic
       ki is the no. of predictors in I
    """
    import numpy as np
    from scipy import linalg

    pS = -np.inf
    for i in range(p-ki):
        Iplusi = setunion(I,Ic[i])
        fitted_model = linalg.lstsq(X[:,Iplusi], y) # least square fit
        xtemp = fitted_model[0]
        fxIplusi = np.linalg.norm(y - X[:,Iplusi] @ xtemp) ** 2
        S = fxI - fxIplusi
        if pS < S:
            pS = S
            istar = Ic[i]
            fstar = fxIplusi

    return fstar, istar 

#=====================================================================================
def setdiff(I,J):
    """ return the set S=I - J of all the elements in set I but not in J preserving the order
    
       of elements in the set I
    """
    import numpy as np
 
    n = len(I)
    S = np.array(n,dtype=int) # initialization
    ctr = -1 
    for i in range(n):
        if I[i] not in J:
            ctr += 1
            S[ctr] = I[i]
        
    S = S[:ctr]
    return S

#======================================================================================
def setunion(I,J):
    """ return the set S= I union J 

    """
    import numpy as np

    n = len(I)
    m = len(J)
    S = np.array(n+m,dtype=int)
    S[:n] = I 
    ctr = -1
    for i in range(m):
        if J[ctr] not in I:
            ctr += 1
            S[n+ctr] = J[ctr]

    S = S[:(n+ctr)]
    return S

#======================================================================================
