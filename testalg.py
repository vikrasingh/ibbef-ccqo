def main(p,n,k,b0,snr,mu,Sigma,num_instances,num_alg,alg_flag):
    
    import numpy as np
    from scipy import linalg
    import time

    import main.ibb as ibb
    import main.ibbef as ibbef 
    import main.mio as mio 
    import main.projgrad as projgrad

    # initialize the arrays to save the output
    rss_each_inst = np.zeros((num_alg,num_instances))
    cpu_time_each_inst = np.zeros((num_alg,num_instances))
    stop_flag_each_inst = np.zeros((num_alg,num_instances))
    sol_each_alg = np.zeros((num_alg,p))
    for j in range(num_instances):
        seed=np.random.seed(j)
        X = np.random.multivariate_normal(mu, Sigma,n )
        colmeans=np.mean(X,axis=0)  # find the mean of each column
        X= X - colmeans             # make the column mean zero
        col_norm = np.linalg.norm(X, axis=0, keepdims=True) # find 2-norm of each col
        X = X / col_norm 
        Xb0 = X @ b0 
        sigma = np.sqrt( (b0.T @ Sigma @ b0)/snr )
        e = np.random.normal(size=n)*sigma # sample error from gaussian distribution
        e = np.reshape(e,(-1,1))   # make it a col array
        y = X @ b0 + e             # generate the response vector
        y = np.reshape(y, (-1,1))  # make it a col array
        A = 2*(X.T @ X)            # hessian matrix
        b = -2*(X.T @ y)           # linear term
        c = y.T @ y                # constant
        c = c[0][0]                # make c scalar
        # for debugging purpose
        #print('X,A,b,c:',X,A,b,c)

        # find an initial feasible point
        x0,fx0=projgrad.main(p,n,y,X,k)
        print('x0,fx0:',x0,fx0)
        
        if alg_flag[0]==1:
            tstart=time.process_time() 
            xibb, rss_each_inst[0,j]=ibb.main(p,n,y,X,A,b,c,k,x0)
            tend=time.process_time()
            print('start, end:',tstart,tend)
            cpu_time_each_inst[0,j]=tend-tstart
            print('xibb:',xibb)
            print('fibb:',rss_each_inst[0,j])
            print('CPU ibb:',cpu_time_each_inst[0,j])

        if alg_flag[1]==1:  # test ibbef
            tstart=time.process_time()
            xibbef, rss_each_inst[1,j]=ibbef.main(p,n,y,X,A,b,c,k,x0)
            tend=time.process_time()
            print('start, end:',tstart,tend)
            cpu_time_each_inst[1,j]=tend-tstart
            print('xibbef:',xibbef)
            print('fibbef:',rss_each_inst[1,j])
            print('CPU ibbef:',cpu_time_each_inst[1,j])

        if alg_flag[2]==1: # test mio
            M=2*np.abs(x0).max() # uniform bound for the box
            low=-M*np.ones((p,1))
            up=M*np.ones((p,1))
            tstart=time.process_time()
            xmio, rss_each_inst[2,j] = mio.main(p,n,y,X,A,b,c,k,low,up,x0)
            tend=time.process_time()
            cpu_time_each_inst[2,j]=tend-tstart
            print('xmio:',xmio)
            print('fmio:',rss_each_inst[2,j])
            print('CPU mio:',cpu_time_each_inst[2,j])

    rss_each_alg = np.mean(rss_each_inst,axis=1)   # average data for all the instances for each algorithm
    cpu_time_each_alg = np.mean(cpu_time_each_inst,axis=1)
    stop_flag_each_alg = np.mean(stop_flag_each_inst,axis=1)
    sol_each_alg[0,:] = xibb.T
    sol_each_alg[1,:] = xibbef.T
    sol_each_alg[2,:] = xmio.T 

    return stop_flag_each_alg, cpu_time_each_alg, rss_each_alg, sol_each_alg
    
