def main(p,n,k,b0,mu,sigma,num_instances,num_alg,alg_flag):
    
    import numpy as np
    from scipy import linalg
    import time

    import ibbef 
    import mio 
    import projgrad

    # initialize the arrays to save the output
    rss_each_inst=np.zeros((num_alg,num_instances))
    cpu_time_each_inst=np.zeros((num_alg,num_instances))
    stop_flag_each_inst=np.zeros((num_alg,num_instances))
    for j in range(num_instances):
        seed=np.random.seed(j)
        X = np.random.multivariate_normal(mu, sigma,n )
        colmeans=np.mean(X,axis=0)  # find the mean of each column
        X= X - colmeans             # make the column mean zero
        col_norm = np.linalg.norm(X, axis=0, keepdims=True) # find 2-norm of each col
        X = X / col_norm 
        e = np.random.normal(size=n) # sample error from gaussian distribution
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

        if alg_flag[0]==1:  # test ibbef
            tstart1=time.process_time()
            #x0=b
            xibbef, rss_each_inst[0,j]=ibbef.main(p,n,y,X,A,b,c,k,x0)
            tend1=time.process_time()
            print('start, end:',tstart1,tend1)
            cpu_time_each_inst[0,j]=tend1-tstart1
            print('xibbef:',xibbef)
            print('fibbef:',rss_each_inst[0,j])
            print('cpu ibbef:',cpu_time_each_inst[0,j])

        if alg_flag[1]==1: # test mio
            low=-4*np.ones((p,1))
            up=4*np.ones((p,1))
            tstart2=time.process_time()
            xmio, rss_each_inst[1,j] = mio.main(p,n,y,X,A,b,c,k,low,up,x0)
            tend2=time.process_time()
            cpu_time_each_inst[1,j]=tend2-tstart2
            print('xmio:',xmio)
            print('fmio:',rss_each_inst[1,j])
            print('cpu mio:',cpu_time_each_inst[1,j])

    rss_each_alg=np.mean(rss_each_inst,axis=1)   # average data for all the instances for each algorithm
    cpu_time_each_alg=np.mean(cpu_time_each_inst,axis=1)
    stop_flag_each_alg=np.mean(stop_flag_each_inst,axis=1)

    return stop_flag_each_alg, cpu_time_each_alg, rss_each_alg
    
