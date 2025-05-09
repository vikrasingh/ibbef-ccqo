def main(p,n,y,X,A,b,c,k,xrelax):
    """IBB+ with lb using recycled echelon form

       xrelax: xRelaxedOpt
    """
    import sys
    import numpy as np
    from scipy import linalg
    from scipy.optimize import minimize
    import heapq
    import itertools

    epsilon=sys.float_info.epsilon    
    pivtol=max(p,n)*epsilon*np.linalg.norm( A, ord=np.inf)  #  tol to check non-zero pivot for echelon form
    R, ipiv0 = linalg.qr(A, mode='r', pivoting=True)
    diagR=np.abs(np.diag(R))
    
    npiv0=sum( i >= pivtol for i in diagR)
    ipiv0=ipiv0[0:npiv0]
    ipiv1=[i for i in range(p) if i not in ipiv0]
    
    #print('A:',A)
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

    absxrelax=np.abs(xrelax) # absolute value of xrelax array
    print('A:',A)
    print('b:',b)
    print('c:',c)
    print('X:',X)
    #print('xrelax:',xrelax)

    # get custom col echelon form of X
    def colech():
        """ Initial column echelon for design matrix X

           Output:
           num_piv: no. of pivot columns found during EROs
           idx_piv: indices of those pivot columns
           CE0: column echelon form of X
        """
        
        CE0=X.copy().T           # CE0 of size p x n
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
    B0=np.ones(p,dtype=int) # initial box of integer ones
    niter=0
    num_box=1
    xbest,fbest=getfeasiblept(p,k,B0,A,b,c,xrelax,absxrelax)
    #xbest=np.zeros((p,1)) # intialize xbest
    #supp0,xbest[supp0]=getklargest(absxrelax,k)
    #fbest=fx(xbest[supp0],A[np.ix_(supp0,supp0)],b[supp0],c)
    print('xbest,fbest:',xbest,fbest)


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
    #print('E0:',E0)
    xlb=backsub(p,npiv0,E0,np.array(range(npiv0)))
    print('xlb :',xlb)
    fxlb=fx(xlb,A,b,c)
    xhat0=xlb
    fxhat0=fxlb
    print('fxlb:',fxlb)
    
    # Initialize the list L
    L = []
    box_age_ctr=0 # unique for every box we are adding, the oldest box will have the smallest counter
    heapq.heappush( L, ( fxlb, box_age_ctr , B0 , 0, p, 0,  xlb) )  # add the intial box to the list


    while True:

        # check for convergence criteria
        if num_box==0:
            print('alg. converged')
            break
        
        # select a new box to process
        Y = heapq.heappop(L) # V[0]=fxlb, V[2]=box, V[3]=#0, V[4]=#1, V[5]=#2, V[6]=xlb
        print('Selected node Y:',Y)
        num_box=num_box-1

        # branch over Y
        box_age_ctr, num_box, fbest, xbest, L = branch(p,n,y,X,A,b,c,k,L,Y,E0,CE0,Ab,xbest,fbest,xhat0,fxhat0,xrelax,absxrelax,npiv0,num_box,box_age_ctr)
            
    
    # final output
    xout=np.zeros((p,1))
    xout[ipiv0]=xbest[:npiv0]
    xout[ipiv1]=xbest[npiv0:p]
    fout=fbest

    return xout, fout
                                
def branch(p,n,y,X,A,b,c,k,L,Y,E0,CE0,Ab,xbest,fbest,xhat0,fxhat0,xrelax,absxrelax,npiv0,num_box,box_age_ctr):
    """
    
    """
    import sys
    import numpy as np
    from scipy import linalg
    from scipy.optimize import minimize
    import heapq
    print('X input in branch:',X)

    par_dir=np.arange(Y[2].size)[Y[2]==1][::-1]  # indices of partition direction in decreasing order
    print('par_dir:',par_dir)
    stop_par=0  # to stop the partition loop
    isDC2true=0     # flag to avoid checking DC2 for every flag 2 child box
    uplimit=p-k-Y[3] 
    num_child=min(uplimit,float('inf'))

    for j in range(num_child):

        jhat=par_dir[j]  # partition index
        #print('Y:',Y)
        tempV0=Y[2].copy()
        tempV0[jhat]=0    # child box with 0 flag
        #print('tempV0:',tempV0)
        V0=[tempV0 , Y[3]+1, Y[4]-1, Y[5]] # initialization 
        #print('V0:',V0)
        tempV2=Y[2].copy()
        tempV2[jhat]=2   # child box with 2 flag
        V2=[tempV2 , Y[3], Y[4]-1, Y[5]+1]
        #print('j,jhat,V2:',j,jhat,V2[0])

        for ichild in range(2):
            
            if ichild==0: # box with 0 flag
                print('V0:',V0)
                if j==(uplimit-1): # reached leaf node with k flag 1 in it
                    print('Reached leaf node for V0 box')
                    # find the feasible point
                    if V0[2]<npiv0: # if #flag 1 < rank X
                        if V0[3]==0: # if there is no flag 2 in the box
                            print('No flag 2 in V0')
                            xhat=backsub(p, V0[2], E0, range(V0[2]))
                            #print('Back sub xhat:',xhat)

                        else:
                            print('Flag 2 in V0')
                            id=newpivcol(V0[2], V0[3], p, npiv0, n, V0[0], X, CE0)
                            xhat=efupdate(V0[2], V0[3], p, npiv0, id, Ab, E0)
                            #print('Updated EF xhat:',xhat)

                        fxhat=fx(xhat, A, b, c)
                
                    else:
                        xhat=xhat0
                        fxhat=fxhat0

                    print('fxhat:',fxhat)
                    ## confirm the solution
                    ##xtemp,ftemp=quad_min(p,np.where(V0[0]!=0)[0],A,b,c,xrelax)  ## 
                    ##print('Verification: xtemp,ftemp:',xtemp,ftemp) ##

                    # update xbest ,fbest if possible
                    if fxhat<fbest:
                        xbest=xhat
                        fbest=fxhat
                            
                    continue # with the next ichild iteration

                # sampling call
                xtilde,fxtilde=getfeasiblept(p,k,V0[0],A,b,c,xrelax,absxrelax)
                print('xtilde, fxtilde:',xtilde, fxtilde)

                if fxtilde<fbest:
                    xbest=xtilde
                    fbest=fxtilde

                # inclusion function call
                print('F call for V0')
                if V0[2] < npiv0: #  if #1 < npiv0
                    print('#1 flag < npiv0')
                    if V0[3]==0: # no flag 0 in the box
                        xhat=backsub(p, V0[2], E0, range(V0[2]))
                        print('Back sub xhat:',xhat)

                    else:
                        id=newpivcol(V0[2], V0[3], p, npiv0, n, V0[0], X, CE0)
                        xhat=efupdate(V0[2], V0[3], p, npiv0, id, Ab, E0)
                        print('Updated EF xhat:',xhat)
                            
                    fxhat=fx(xhat,A,b,c)

                else: # if #flag 1 >= rank X
                    xhat=xhat0
                    fxhat=fxhat0
                        
                xlbV0=xhat
                fxlbV0=fxhat
                print('xlbV0,fxlbV0:',xlbV0,fxlbV0)
                ## confirm the solution
                ##xtemp,ftemp=quad_min(p,np.where(V0[0]!=0)[0],A,b,c,xrelax)  ## 
                ##print('Verification: xtemp,ftemp:',xtemp,ftemp) ##

                # check DC1
                if fbest<=fxlbV0:
                    stop_par=1 # stop the j loop
                    print('V0 got deleted using DC1: fbest, fxlbV0:',fbest,fxlbV0)
                    continue # with the next ichild iteration
                print('V0:',V0)
                box_age_ctr +=1
                Ytemp=(fxlbV0, box_age_ctr ,V0[0], V0[1] , V0[2] , V0[3] , xlbV0) 
                #Y=(fxlb,) + V0 + (xlb,)
                print('Ytemp:',Ytemp)

            else: # box with 2 flag
                print('V2:',V2)
                # check DC2
                if j==0:
                    if Y[5]+1==k:
                        print('DC2 is true for all V2 boxes')
                        isDC2true=1 # DC2 is satisfied for all the subsequent child boxes,check it only once

                if isDC2true==1:
                    # call lb QM
                    suppV2=np.where(V2[0]==2)[0]
                    xhat,fxhat=quad_min(p,suppV2,A,b,c,xrelax)
                    print('xhat,fhat:',xhat,fxhat)
                    # update xbest, fbest if possible
                    if fxhat<fbest:
                        xbest=xhat
                        fbest=fxhat
                        print('xbest,fbest got updated')
                        #print('xbest,fbest:',xbest,fbest)
                        
                    continue # with the next iter of j loop

                # call feasiblity sampling
                xtilde,fxtilde=getfeasiblept(p,k,V2[0],A,b,c,xrelax,absxrelax)
                if fxtilde<fbest:
                    xbest=xtilde
                    fbest=fxtilde
                    print('xbest,fbest got updated')
                    #print('xbest,fbest:',xbest,fbest)
            
                xlb=Y[6]
                fxlb=Y[0]
                print('xlb,fxlb:',xlb,fxlb)
                # check DC1
                if fbest<=fxlb:
                    print('V2 deleted DC1: fbest,fxlb',fbest,fxlb)
                    continue # discard the box

                num_box = num_box+1
                print('V2 added:',V2)
                box_age_ctr += 1
                V2temp =(fxlb, box_age_ctr,  V2[0] , V2[1] , V2[2] , V2[3] , xlb)
                heapq.heappush( L, V2temp )  # add the V2 box to the list
                #heapq.heappush( L, (fxlb,) + V2 + (xlb,) )  # add the V2 box to the list
        
        if stop_par==1: # child box with flag 1 got deleted using DC1
            break   

        if j<(uplimit-1): # if the current V0 is not the leaf node 
            Y = Ytemp  # assign the box with flag 0 to be the next parent box
        

    return box_age_ctr, num_box, fbest, xbest, L

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

    #print('n1,n2,p,r,n,Y,X,CE0:',n1,n2,p,r,n,Y,X,CE0)
    d=min(n1+n2,r)
    id=np.zeros(d,dtype=int) # initialization
    id[:n1]=np.array(range(n1)) # the first n1 entries of Y are flag 1
    #print('id:',id)
    iflag2=np.where(Y==2) # indices of flag 2 entries in Y
    #print('iflag2:',iflag2)
    id[n1:d]=iflag2[0][:d-n1]
    #print('id:',id)

    CE=np.hstack( ( CE0[:,:n1],X[:,id[n1:d]] ) ) # initialize CE as col echelon form of X
    print('CE:',CE)
    T=CE.T   # use EROs on the transpose, T=[U, *; V, W]
    # reduce the lower left block of T corresponding to the first n1 cols
    # to zero to get ET=[U, *; 0, Z] after EROs
    ET=T # initialization
    for j in range(n1):
        ETi=-ET[j,j:n]
        for i in range(n1,d):
            ET[i,j:n]=ETi*ET[i,j] + ET[i,j:n]
        
    # now, ET=[U, *; 0, Z]
    print('ET:',ET)
    num_npr=0  # no. of new pivot cols
    for i in range(n1,d): # first n1 rows of T are not changed
        if np.abs(ET[i,n1:n]).sum() > 1e-10: # if the row is not a zero row
            num_npr=num_npr+1
            jz=np.abs(ET[i,n1:n]).argmax()  # find the entry with largest magnitude to used as pivot entry
            jz=jz+n1
            ET[i,n1:n]=ET[i,n1:n]/ET[i,jz] # make the pivot entry 1
            ETi=-ET[i,n1:n]
            for k in range(i+1,d):
                ET[k,n1:n]=ETi*ET[k,jz] + ET[k,n1:n] # EROs

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
    #print('n1,n2,p,r:',n1,n2,p,r)
    #print('id:',id)
    #print('A0:',A0)
    #print('E0:',E0)
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
    cia=np.zeros(d,dtype=int) # initialization
    cia[:n1]=np.array(range(n1)) # first n1 indices are flag 1
    bool_idx_cia=np.ones(d,dtype=bool) # to discard indices of dependent cols
    for j in range(n1,d):
        if id[j]<0:
            cia[j]=-id[j]
        else:
            d=d-1
            bool_idx_cia[j]=False
    
    cia = cia[bool_idx_cia]
    # drop the nonbasic cols correspond to flag 2 in the box and add RHS of the system [Q|d]
    #print('d:',d)
    #print('cia:',cia)
    #print('E0[:,cia]:',E0[:,cia])
    #print('E0[:,p]:',E0[:,p])
    tempE=np.hstack( ( E0[:,cia],E0[:,p].reshape((-1,1))) ) 
    # recycle the first n1 rows of E0 and add last r-n1 rows of A0
    #print('tempE[cia[:n1],:]:',tempE[cia[:n1],:])
    #print('cia[n1:d], np.concatenate((cia,[p])):',cia[n1:d],np.concatenate((cia,[p])))
     
    if n1<d:
        #print('A0[cia[n1:d], np.concatenate((cia,[p])) ]:',A0[np.ix_(cia[n1:d], np.concatenate((cia,[p]))) ])
        E=np.vstack( (tempE[cia[:n1],:],  A0[np.ix_( cia[n1:d], np.concatenate((cia,[p]))) ]  ) )
        E0=updatelowerblock(E)
    else:
        E0=tempE[cia[:n1],:]

    for j in range(n1,d):
        E0[j,j:(d+1)]=E0[j,j:(d+1)]/E0[j,j]
        #print('E0:',E0)
        E0i=-E0[j,j:(d+1)]
        for i in range(j+1,d):
            E0[i,j:(d+1)]=E0i*E0[i,j] + E0[i,j:(d+1)]
            #print('E0:',E0)
            
    #print('E0:',E0)
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

    return value[0][0]

def quad_min(p,supp,A,b,c,x0):
    " quadratic minimization using conjugate gradient method"

    import numpy as np
    from scipy.optimize import minimize

    def quad_fun(x,A,b,c):
        " x is 1D array"
        value= 0.5 * x @ A @ x.T + x @ b.T + c
        
        return value

    def grad_fun(x,A,b,c):
        "g = Ax+b"
        value= x @ A + b
        
        return value
    
    #print('supp:',supp)
    start_pt=np.reshape(x0[supp],(1,-1))[0]
    Ahat=A[np.ix_(supp,supp)]
    bhat=b[supp]
    bhat=np.reshape(bhat,(1,-1))[0] # convert into a 1D array
    #print('supp,Ahat,bhat,c,start_pt:',supp,Ahat,bhat,c,start_pt)
    obj=minimize( quad_fun, start_pt, (Ahat,bhat,c), method='CG', jac=grad_fun)
    xout=np.zeros((p,1))
    xout[supp]=np.reshape(obj.x,(-1,1))
    fout=obj.fun

    return xout, fout

def getfeasiblept(p,k,box,A,b,c,xrelax,absxrelax):
    "Find a feasible point"
    
    import numpy as np
    from scipy.optimize import minimize

    def quad_fun(x,A,b,c):
        " x is 1D array"
        value= 0.5 * x @ A @ x.T + x @ b.T + c
        return value

    def grad_fun(x,A,b,c):
        "g = Ax+b"
        value= x @ A + b
        return value

    #supp1=np.where(box!=0)[0] # find the indices of flag 1 and flag 2
    supp1=[]
    ctr=0
    for i in range(p):
        if box[i]==2:
            supp1 = np.concatenate((supp1,[i]))
            ctr += 1

    j = 0
    while ctr < k:
        if box[j]==1:
            supp1 = np.concatenate((supp1,[j]))
            ctr += 1

        j += 1

    absxrelax = np.reshape(absxrelax,(1,-1))[0] # make it a 1d array
    #print('absxrelax:',absxrelax)
    supp1=np.array(supp1,dtype=int)  
    #print('supp1:',supp1)
    local_supp,_=getklargest(absxrelax[supp1],k)
    #print('local_supp:',local_supp)
    supp=supp1[local_supp]
    #print('supp:',supp)
    start_pt=xrelax[supp]
    start_pt=np.reshape(start_pt,(1,-1))[0] # convert into a 1D array
    Ahat=A[np.ix_(supp,supp)]
    bhat=b[supp]
    bhat=np.reshape(bhat,(1,-1))[0]
    #print('supp:',supp)
    #print('Ahat:',Ahat)
    #print('bhat:',bhat)
    #print('c:',c)
    obj=minimize( quad_fun, start_pt, (Ahat,bhat,c) , method='CG', jac=grad_fun)
    xout=np.zeros((p,1))
    xout[supp]=np.reshape(obj.x,(-1,1))
    fout=obj.fun
    #print('xout,fout:',xout,fout)

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
