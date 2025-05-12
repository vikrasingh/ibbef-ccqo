def main(p,n,y,X,k):

    import numpy as np
    from scipy import linalg

    XX = X.T @ X  
    eigenvalues,_=np.linalg.eig(XX)
    real_eigenvalues=eigenvalues.real  # get the real part of the eigen values as the img part is 0
    #print('real_eig:',real_eigenvalues)
    L = np.max(real_eigenvalues) # the largest eigenvalue of X'X
    max_iter=1000  # iter. limit
    num_runs=50    # no. of runs from random starting points    
    epstol=1e-4   # this tolerance is suggested in the reference p833 
    stop_flag=0   # =0 means full convergence, =5 means, maxiter has been reached for atleast 1 of the random runs

    if p<n:
        fitted_model = linalg.lstsq(X, y) # least square fit
        beta0=fitted_model[0]
    else:
        col_sum = np.sum(X ** 2, axis=0).reshape((-1,1))
        Xy = X.T @ y
        Xy = np.reshape(Xy, (1,-1)) # make it a row vector
        #print('col_sum:',col_sum)
        #print('X*y:',X.T @ y)
        beta0 = X.T @ y/col_sum
        
    beta=beta0
    #print('beta0:',beta0)
    fun_val=np.inf # initialization
    beta_out=beta0 
    #print('beta0:',beta)
    #print('p,k=',p,k)
    for i in range(num_runs):

        for j in range(max_iter):
            beta_old = beta
            # gradient descent step
            grad = - X.T @ (y - X @ beta) 
            #print('grad, L:',grad,L)
            beta = beta - grad/L
            #print('beta=',beta)
            # pick the top k entries and set rest to zero
            sorted_indices = np.argsort(np.abs(beta[:,0]))
            #print('sorted_indices:',sorted_indices)
            #print('sorted_indicespminusk:',sorted_indices[:(p-k)])
            beta[sorted_indices[:(p-k)]]=0
            #print('beta11:',beta)
            # polish the coefficients
            #print('X:',X)
            #print('sorted_indices:',sorted_indices[-k:])
            #print('X[:,supp]',X[:,sorted_indices[-k:]])
            fitted_model = linalg.lstsq(X[:,sorted_indices[-k:]], y) # least square fit
            #print('beta1:',fitted_model[0])
            beta[sorted_indices[-k:]]=fitted_model[0]
            #print('beta2:',beta)
            # stop, if the relative difference in coeff. < tol
            rel_diff=np.linalg.norm(beta-beta_old)/np.linalg.norm(beta, ord=1)
            if rel_diff<epstol:
                break

        
        current_fun_val = np.linalg.norm(y - X @ beta) ** 2

        if current_fun_val<fun_val:
            fun_val = current_fun_val
            beta_out = beta
        
        # start the next run at a random point
        uni_rnd = np.random.rand(p)
        uni_rnd = np.reshape(uni_rnd,(-1,1))
        beta = beta0 + 2*uni_rnd*np.max(np.abs(beta0))
        #print('beta3:',beta)
        
    return beta_out, fun_val


        



