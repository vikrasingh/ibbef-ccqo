import numpy as np
from scipy import linalg
import utility.getklargest as gkl

def main(p,n,y,X,k,box,xrelax):
    """ Sequential Feature Swapping for BSS
    
       X is the design matrix of size n x p
       y is the response vector of size n x 1
       k is the no. of predictors to choose from total p predictors
       I0 is the initial support, i.e. the predictors already fixed to be in the model
       box  is the current box selected
       xrelax is the ols solution 
    """

    stop_flag = 0 
    num_iter = 0
    #print('box:',box)

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
    Ic = setdiff(np.array(range(p),dtype=int),I)
    #print('I,fxI:',I,fxI)

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
            #print('I,fxI:',I,fxI)

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
 
    n = I.shape
    m = J.shape

    if n == () and m == (): # I and J are both integers
        if I != J:
            S = I
        
        return S

    if n == ():  # I is an integer and J is a list
        if I not in J:
            S = I
        else:
            S = []

        return S

    S = np.zeros(n,dtype=int) # initialization
    ctr = -1

    if m == (): # J is just an integer and I is a list
        for i in range(n[0]):
            if I[i] != J:
                ctr += 1
                S[ctr] = I[i]
            
        S = S[:(ctr+1)]
        return S

    for i in range(n[0]):
        if I[i] not in J:
            ctr += 1
            S[ctr] = I[i]
        
    S = S[:(ctr+1)]
    return S

#======================================================================================
def setunion(I,J):
    """ return the set S= I union J 

    """

    n = I.shape
    m = J.shape

    if n == () and m == (): # if I and J are both integers
        if I != J:
            S = np.array(range(2),dtype=int)
            S[0] = I
            S[1] = J
        
        return S
    

    if n == (): # if I is an integer and J is a list of integers
        if I not in J:
            S = np.array(range(m[0]+1),dtype=int)
            S[0] = I
            S[1:(m[0]+1)] = J      
        else:
            S = J

        return S


    if m == (): # if J is an integer and I is a list of integers
        if J not in I:
            S = np.array(range(n[0]+1),dtype=int)
            S[:n[0]] = I
            S[n[0]] = J
        else:
            S = I
                
        return S

    
    S = np.array(range(n[0]+m[0]),dtype=int)
    S[:n[0]] = I 
    ctr = -1

    for i in range(m[0]): # I and J both are list of integers
        if J[i] not in I:
            ctr += 1
            S[n[0]+ctr] = J[i]

    S = S[:(n[0]+ctr+1)]
    return S
#======================================================================================
