def main(p,n,y,X,A,Q,D,b,c,k,x0):
    """IBB+ with lb using recycled echelon form

       x0: xRelaxedOpt
    """
    import sys
    import numpy as np
    from scipy import linalg
    import heapq

    epsilon=sys.float_info.epsilon    
    pivtol=max(p,n)*epsilon*np.linalg.norm( A, ord=np.inf)  #  tol to check non-zero pivot for echelon form
    R, ipiv0 = linalg.qr(A, mode='r', pivoting=True)
    diagR=np.abs(np.diag(R))
    
    npiv0=sum( i >= pivtol for i in diagR)
    ipiv0=ipiv0[0:npiv0]
    ipiv1=[i for i in range(p) if i not in ipiv0]
    
    print('A:',A)
    print('ipiv0:',ipiv0)
    print('ipiv1:',ipiv1)
    #print('A[ipiv0,ipiv0]',A[np.ix_(ipiv0,ipiv0)])
    #print('A[ipiv0,ipiv1]',A[np.ix_(ipiv0,ipiv1)])
    #print('A[ipiv1,ipiv0]',A[np.ix_(ipiv1,ipiv0)])
    #print('A[ipiv1,ipiv1]',A[np.ix_(ipiv1,ipiv1)])

    # Switch the cols and rows of the quadratic function
    if not ipiv1: # ipiv1=[]
       A=A[np.ix_(ipiv0,ipiv0)]
       b=b[ipiv0]
       X=X[:,ipiv0]
       x0=x0[ipiv0]
    else: 
       A=np.vstack( (  np.hstack((A[np.ix_(ipiv0,ipiv0)],A[np.ix_(ipiv0,ipiv1)])),np.hstack( (A[np.ix_(ipiv1,ipiv0)],A[np.ix_(ipiv1,ipiv1)]) )) )
       b=np.concatenate( (b[ipiv0],b[ipiv1]) )
       X=np.hstack( (X[:,ipiv0],X[:,ipiv1]) )
       x0=np.concatenate( (x0[ipiv0],x0[ipiv1]) )

    print('A:',A)
    print('b:',b)
    #print('X:',X)
    #print('x0:',x0)

    # get custom col echelon form of X
    def colech():
        """ Initial column echelon for design matrix X

           Output:
           num_piv: no. of pivot columns found during EROs
           idx_piv: indices of those pivot columns
           CE0: column echelon form of X
        """
        
        CE0=X.T           # CE0 of size p x n
        num_piv=0         # initialization
        m=min([p,n])
        idx_piv=[True]*p  # to store indices of pivot column
        
        for i in range(m):
            if np.abs(CE0[i,i])<pivtol:
                idx_piv[i]=False # i is the non pivot column
                continue
            num_piv=num_piv+1
            CE0[i,i:n]=CE0[i,i:n]/CE0[i,i] # make the pivot entry 1
            CE0i=-CE0[i,i:n]  
            for k in range(i+1,m):
                CE0[k,i:n]=CE0i*CE0[k,i] + CE0[k,i:n]  # EROs
        
        if num_piv<p:
            #print('num_piv:',num_piv)
            #print('p-num_piv:',p-num_piv)
            #print('subarray:',idx_piv[num_piv:])
            idx_piv[num_piv:]=[False]*(p-num_piv)

        CE0=CE0[idx_piv,:].T    # final CE0 of size n x num_piv
        return num_piv, idx_piv, CE0 

    npiv00,_,CE0=colech()
    if npiv00!=npiv0:
        print('No. of pivot cols of X from custom echelon form is not same as qr')

    print('npiv0, npiv00:',npiv0,npiv00)
    #print('CE0:',CE0)
    Ab=np.hstack((A,-b))   # augmented matrix for the first order linear system
    #print('Ab:',Ab)
    niter=0
    num_box=1
    B0=np.ones(p,dtype=int) # initial box of integer ones

    def rowech():
        """ row reduced echelon form
        """
        num_piv=npiv0
        idx_piv=np.array(range(num_piv))
        E0=Ab[idx_piv,:]
        num_col=p+1
        for j in range(num_piv):
            E0[j,j:num_col]=E0[j,j:num_col]/E0[j,j] # make the pivot entry 1
            invjpiv=-E0[j,j:num_col]
            for k in range(j+1,num_piv):
                E0[k,j:num_col]=invjpiv*E0[k,j]+E0[k,j:num_col] # EROs
        
        return E0


    E0=rowech()
    print('E0:',E0)
    xlb=backsub(p,npiv0,E0,np.array(range(npiv0)))
    #xlb[ipiv0]=xlb
    fxlb=fx(v,A,b,c)
    #print('xlb :',xlb)
    #print('fxlb:',fxlb)
    #print('size fxlb:',np.size(fxlb))

    # Initialize the list L
    L = []
    heapq.heappush( L, (fxlb, B0, 0, p, 0, xlb) )  # add the intial box to the list


    while True:

        # check for convergence criteria
        if num_box==0:
            print('alg. converged')
            break
        
        # select a new box to process
        V=heapq.heappop(L) # V[0]=fxlb, V[1]=box, V[2]=#0, V[3]=#1, V[4]=#2, V[5]=xlb
        num_box=num_box-1
        branch()

        def branch():
            """ partition the current box and process the child boxes

            """
            par_dir=np.arange(V[1].size)[V[1]==1][::-1]  # indices of partition direction in decreasing order
            stop_par=0  # to stop the partition loop
            isDC1=0     # flag to avoid checking DC2 for every flag 2 child box
            uplimit=p-k-V[2] 
            num_child=min(uplimit,float('inf'))

            for j in range(num_child):

                jhat=par_dir[j]  # partition index
                temp=V[1]
                temp[jhat]=0    # child box with 0 flag
                V0=(temp , V[2]+1, V[3]-1, V[4]) # initialization 
                temp=V[1]
                temp[jhat]=2   # child box with 2 flag
                V2=(temp , V[2], V[3]-1, V[4]+1)

                for ichild in range(2):

                    if ichild==0: # box with 0 flag
                        if j==(uplimit-1): # reached leaf node with k flag 1 in it
                            # find the feasible point

                        # inclusion function call
                        if V0[2]<npiv0: #  if #1 < npiv0
                            if V0[3]==0: # no flag 0 in the box
                                xhat=backsub(p, V0[2], E0, range(V0[2]))
                            else:
                                id=newpivcol



                    else: # box with 2 flag








def fx(x,A,b,c):
    """ fx=0.5 xAx + bx +c
    
     """
    #print('Ain:',A)
    #print('bin:',b)
    #print('cin:',c)
    #print('xin:',x)
    #print('fx:',0.5*(x.T@A@x)+b.T@x+c)
    #print('fx:',b.T@x)
    #Qx = Q.T @ x
    #value=(Qx.T @ (D*Qx)) + b.T @ x + c
    value=0.5*(x.T @ (A @ x) ) + b.T @ x + c 

    return value

def backsub(n,num_piv,E,idx_piv):

    import numpy as np

    x = np.zeros((n,1)) # initialize
    for i in range(num_piv-1,-1,-1):
        print('i:',i)
        sum=0
        for j in range(i+1,num_piv):
            print('j:',j)
            sum = sum + E[idx_piv[i],idx_piv[j]]*x[idx_piv[j]]
            print('sum:',sum)
        
        x[idx_piv[i]]=E[idx_piv[i],n] - sum
        #print('idx[i]:',idx_piv[i])
        #print('x:',x)

    return x
