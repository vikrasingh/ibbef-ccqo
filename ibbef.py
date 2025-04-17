def main(p,n,y,X,A,b,c,k,xrelax):
    """IBB+ with lb using recycled echelon form

       xrelax: xRelaxedOpt
    """
    import sys
    import numpy as np
    from scipy import linalg
    from scipy.optimize import minimize
    import heapq

    epsilon=sys.float_info.epsilon    
    pivtol=max(p,n)*epsilon*np.linalg.norm( A, ord=np.inf)  #  tol to check non-zero pivot for echelon form
    R, ipiv0 = linalg.qr(A, mode='r', pivoting=True)
    diagR=np.abs(np.diag(R))
    
    npiv0=sum( i >= pivtol for i in diagR)
    ipiv0=ipiv0[0:npiv0]
    ipiv1=[i for i in range(p) if i not in ipiv0]
    
    absxrelax=np.abs(xrelax) # absolute value of xrelax array
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
       xrelax=xrelax[ipiv0]
    else: 
       A=np.vstack( (  np.hstack((A[np.ix_(ipiv0,ipiv0)],A[np.ix_(ipiv0,ipiv1)])),np.hstack( (A[np.ix_(ipiv1,ipiv0)],A[np.ix_(ipiv1,ipiv1)]) )) )
       b=np.concatenate( (b[ipiv0],b[ipiv1]) )
       X=np.hstack( (X[:,ipiv0],X[:,ipiv1]) )
       xrelax=np.concatenate( (xrelax[ipiv0],xrelax[ipiv1]) )

    print('A:',A)
    print('b:',b)
    #print('X:',X)
    #print('xrelax:',xrelax)

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
    xbest=np.zeros((p,1)) # intialize xbest
    supp0,xbest[supp0]=getklargest(absxrelax,k)
    fbest=fx(xbest[supp0],A[np.ix_(supp0,supp0)],b[supp0],c)
    #print('xbest,fbest:',xbest,fbest)
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
    fxlb=fx(xlb,A,b,c)
    xhat0=xlb
    fxhat0=fxlb
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
        Y=heapq.heappop(L) # V[0]=fxlb, V[1]=box, V[2]=#0, V[3]=#1, V[4]=#2, V[5]=xlb
        num_box=num_box-1

        par_dir=np.arange(Y[1].size)[Y[1]==1][::-1]  # indices of partition direction in decreasing order
        stop_par=0  # to stop the partition loop
        isDC2true=0     # flag to avoid checking DC2 for every flag 2 child box
        uplimit=p-k-Y[2] 
        num_child=min(uplimit,float('inf'))

        for j in range(num_child):

            jhat=par_dir[j]  # partition index
            temp=Y[1]
            temp[jhat]=0    # child box with 0 flag
            V0=(temp , Y[2]+1, Y[3]-1, Y[4]) # initialization 
            temp=Y[1]
            temp[jhat]=2   # child box with 2 flag
            V2=(temp , Y[2], Y[3]-1, Y[4]+1)

            for ichild in range(2):

                if ichild==0: # box with 0 flag
                    if j==(uplimit-1): # reached leaf node with k flag 1 in it
                        # find the feasible point
                        if V0[2]<npiv0: # if #flag 1 < rank X
                            if V0[3]==0: # if there is no flag 2 in the box
                                xhat=backsub(p, V0[2], E0, range(V0[2]))

                            else:
                                id=newpivcol(V0[2], V0[3], p, npiv0, n, V0[0], X, CE0)
                                xhat=efupdate(V0[2], V0[3], p, npiv0, id, Ab, E0)

                            fxhat=fx(xhat, A, b, c)

                        else:
                            xhat=xhat0
                            fxhat=fxhat0

                        # update xbest ,fbest if possible
                        if fxhat<fbest:
                            xbest=xhat
                            fbest=fxhat
                            
                        continue # with the next ichild iteration

                    # sampling call
                    xtilde,fxtilde=getfeasiblept(p,k,V0[0],A,b,c,xrelax,absxrelax)
                    if fxtilde<fbest:
                        xbest=xtilde
                        fbest=fxtilde

                    # inclusion function call
                    if V0[2] < npiv0: #  if #1 < npiv0
                        if V0[3]==0: # no flag 0 in the box
                            xhat=backsub(p, V0[2], E0, range(V0[2]))
                        else:
                            id=newpivcol(V0[2], V0[3], p, npiv0, n, V0[0], X, CE0)
                            xhat=efupdate(V0[2], V0[3], p, npiv0, id, Ab, E0)
                            
                        fxhat=fx(xhat,A,b,c)

                    else: # if #flag 1 >= rank X
                            xhat=xhat0
                            fxhat=fxhat0
                        
                    xlb=xhat
                    fxlb=fxhat0

                    # check DC1
                    if fbest<=fxlb:
                        stop_par=1 # stop the j loop
                        continue # with the next ichild iteration

                    Y=(fxlb) + V0 + (xlb)

                else: # box with 2 flag

                    # check DC2
                    if j==0:
                        if Y[4]+1==k:
                            isDC2true=1 # DC2 is satisfied for all the subsequent child boxes,check it only once

                    if isDC2true==1:
                        # call lb QM
                        xhat,fxhat=quad_min(p,V2[0],A,b,c,xrelax)
                        # update xbest, fbest if possible
                        if fxhat<fbest:
                            xbest=xhat
                            fbest=fxhat
                        
                        continue # with the next iter of j loop

                    # call feasiblity sampling
                    xtilde,fxtilde=getfeasiblept(p,k,V2[0],A,b,c,xrelax,absxrelax)
                    if fxtilde<fbest:
                        xbest=xtilde
                        fbest=fxtilde
                            
                    xlb=Y[5]
                    fxlb=Y[0]

                    # check DC1
                    if fbest<=fxlb:
                        continue # discard the box

                    heapq.heappush( L, (fxlb) + V2 + (xlb) )  # add the V2 box to the list
            

    
    # final output
    xout=np.zeros((p,1))
    xout[ipiv0]=xbest[:npiv0]
    xout[ipiv1]=xbest[npiv0:p]
    fout=fbest

    return xout, fout
                                


def newpivcol(n1,n2,p,r,n,Y,X,CE0):
    """ Determine new pivot columns among the flag 2 columns

        Assumption: n1< rank X    
        Input:
        n1 : # flag 1
        n2 : # flag 2
        X : design matrix of order nxp
        r : rank of X
        Y : flag array of the current box Y with first n1 entries as flag 1
        CE0 : column echelon form of order nxr
        Output:
        id with pivot column indices, with some negative indices
        id[i] < 0 means x[-id[i]] is a new basic variable i.e. x[-id[i]]=/0
        id[i] > 0 means x[id[i]] is a non basic variable i.e. x[id[i]]=0
    """
    import numpy as np

    d=min(n1+n2,r)
    id=np.zeros(d) # initialization
    id[:n1]=np.array(range(n1)) # the first n1 entries of Y are flag 1
    iflag2=np.where(Y==2) # indices of flag 2 entries in Y
    id[n1:d]=iflag2[:d-n1]

    CE=np.hstack( ( CE0[:,:n1],X[:,id[n1:d]] ) ) # initialize CE as col echelon form of X
    T=CE.T   # use EROs on the transpose, T=[U, *; V, W]
    # reduce the lower left block of T corresponding to the first n1 cols
    # to zero to get ET=[U, *; 0, Z] after EROs
    ET=T # initialization
    for j in range(n1):
        ETi=-ET[j,j:n]
        for i in range(n1,d):
            ET[i,j:n]=ETi*ET[i,j] + ET[i,j:n]
        
    # now, ET=[U, *; 0, Z]

    num_npr=0  # no. of new pivot cols
    for i in range(n1,n1+n2): # first n1 rows of T are not changed
        if np.abs(ET[i,n1:d]).sum() > 1e-10: # if the row is not a zero row
            num_npr=num_npr+1
            jz=np.abs(ET[i,n1:d]).argmax()  # find the entry with largest magnitude to used as pivot entry
            jz=jz+1
            ET[i,n1:d]=ET[i,n1:d]/ET[i,jz] # make the pivot entry 1
            ETi=-ET[i,n1:d]
            for k in range(i+1,d):
                ET[k,n1:d]=ETi*ET[k,jz] + ET[k,n1:d] # EROs

            id[i]=-id[i]
            if num_npr==(r-n1):
                break  # exit i loop


    return id 

def efupdate(n1,n2,p,r,id,A0,E0):
    """ update row echelon form to get lb f(Y)
        
        Assumption: n1 < rank X
        Input:
        n1 : # flag 1
        n2 : # flag 2
        id with pivot column indices, with some negative indices
        id[i] < 0 means x[-id[i]] is a new basic variable i.e. x[-id[i]]=/0
        id[i] > 0 means x[id[i]] is a non basic variable i.e. x[id[i]]=0
        A0 : [Q|d] the original augmented matrix
        E0 : master simplified echelon form of A0 of size r x p+1, E0=[U *] with the 
        diagonal entries of the upper triangular U equal to 1. Now row switches allowed to
        get E0
        Output:
        xstar : solution of the linear system after updating echelon form and back subs 
    """
    import numpy as np

    def updatelowerblock(A0):
        """ 
        Input:
        A0 = [U *; * *]
        Output:
        E0 = [U | rhs] is in rref
        """
        E0=A0  # initialization
        q=d+1  # no. of col of A0
        # reduce the lower block to zero for the first n1 cols
        for j in range(n1):
            E0i=-E0[j,j:q]
            for i in range(n1,d):
                E0[i,j:q]=E0i*E0[i,j] + E0[i,j:q]

        return E0

    d=min(n1+n2,r)
    cia=np.zeros(d) # initialization
    cia[:n1]=np.array(range(n1)) # first n1 indices are flag 1
    for j in range(n1,d):
        if j<=(r-1) and id[j]<0:
            cia[j]=-id[j]
        
    # drop the nonbasic cols correspond to flag 2 in the box and add RHS of the system [Q|d]
    tempE=np.hstack( ( E0[:,cia],E0[:,p]) ) 
    # recycle the first n1 rows of E0 and add last r-n1 rows of A0
    E=np.vstack( (tempE[cia[:n1],:],   A0[cia[n1:d], np.hstack((cia,p)) ] ) )
    E0=updatelowerblock(E)
    xr=backsub(d,d,E0,range(d)) # back sub
    xstar=np.zeros((p,1)) # initialization
    xstar[cia]=xr

    return xstar
    
def backsub(n,num_piv,E,idx_piv):

    import numpy as np

    x = np.zeros((n,1)) # initialize
    for i in range(num_piv-1,-1,-1):
        #print('i:',i)
        sum=0
        for j in range(i+1,num_piv):
            #print('j:',j)
            sum = sum + E[idx_piv[i],idx_piv[j]]*x[idx_piv[j]]
            #print('sum:',sum)
        
        x[idx_piv[i]]=E[idx_piv[i],n] - sum
        #print('idx[i]:',idx_piv[i])
        #print('x:',x)

    return x
    
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
    value=0.5 * x.T @ A @ x  + b.T @ x + c 

    return value

def grad_fun(x,A,b):
    "g = Ax+b"
    value= A @ x + b
    return value

def quad_min(p,box,A,b,c,x0):
    " quadratic minimization using conjugate gradient method"

    import numpy as np
    from scipy.optimize import minimize
    
    supp=np.where(box!=0) # find the indices of flag 1 and flag 2
    start_pt=x0[supp]
    Ahat=A[np.ix_(supp,supp)]
    bhat=b[supp]
    obj=minimize( fx, start_pt, method='CG', jac=grad_fun, hess=Ahat)
    xout=np.zeros(p,1)
    xout[supp]=obj.x
    fout=obj.fun

    return xout, fout

def getfeasiblept(p,k,box,A,b,c,xrelax,absxrelax):
    "Find a feasible point"
    
    import numpy as np
    from scipy.optimize import minimize

    supp1=np.where(box!=0) # find the indices of flag 1 and flag 2
    local_supp,_=getklargest(absxrelax[supp1],k)
    supp=supp1[local_supp]
    start_pt=xrelax[supp]
    Ahat=A[np.ix_(supp,supp)]
    bhat=b[supp]
    obj=minimize( fx, start_pt, method='CG', jac=grad_fun, hess=Ahat)
    xout=np.zeros(p,1)
    xout[supp]=obj.x
    fout=obj.fun

    return xout,fout
    
def getklargest(given_list,select_k):
        """ Extract a sub array of size kx1 with k largest entries
    
        """
        import numpy as np
        #print('given_list:',given_list)
        #print('k:',select_k)
        if np.shape(given_list)[0]!=1: # if is not a 1D array
            given_list_1D=given_list.flatten()

        else:
            given_list_1D=given_list
        
        indices_k_largest=np.argpartition(given_list_1D,-select_k)[-select_k:]
        list_k_largest=given_list[indices_k_largest]
        return indices_k_largest,list_k_largest
