def main(p,n,y,X,A,b,c,k,xrelax,num_cuts=1):
    """IBB+ with lb using recycled echelon form

       xrelax: xRelaxedOpt
    """
    import sys
    import numpy as np
    from scipy import linalg
    from scipy.optimize import minimize
    import heapq

    epsilon=sys.float_info.epsilon    
    
    absxrelax=np.abs(xrelax) # absolute value of xrelax array
    print('A:',A)
    print('b:',b)
    #print('X:',X)
    #print('xrelax:',xrelax)

    num_iter=0
    num_box=1
    xbest=np.zeros((p,1)) # intialize xbest
    supp0,xbest[supp0]=getklargest(absxrelax,k)
    fbest=fx(xbest[supp0],A[np.ix_(supp0,supp0)],b[supp0],c)
    #print('xbest,fbest:',xbest,fbest)
    B0=np.ones(p,dtype=int) # initial box of integer ones

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

        num_iter += 1
        # check for convergence criteria
        if num_box==0:
            print('alg. converged')
            break
        
        # select a new box to process
        Y=heapq.heappop(L) # V[0]=fxlb, V[1]=box, V[2]=#0, V[3]=#1, V[4]=#2, V[5]=xlb
        num_box -= 1
        
    
    # final output
    xout=xbest
    fout=fbest

    return xout, fout

#============================================================================================================================
def branch(p,n,y,X,A,b,c,k,L,Y,xbest,fbest,xrelax,absxrelax,npiv0,num_box,box_age_ctr,num_cuts):
    """
    
    """
    # branch and process the child boxes
    par_dir=np.arange(Y[1].size)[Y[1]==1]  # indices of partition direction in increasing order
    num_child, cut_dir, child_boxes=getchildboxes(num_cuts,par_dir,Y[6],Y[2:6])

    for j in range(num_child):

#============================================================================================================================                                
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

#============================================================================================================================
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

#============================================================================================================================
def getfeasiblept(p,n,y,X,A,b,c,k,box,xrelax,absxrelax):
    "Find a feasible point"
    
    import numpy as np
    from scipy.optimize import minimize
    import projgrad as pg

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
        if box[i]!=0:
            supp1 = np.concatenate((supp1,[i]))
            ctr += 1

    supp1=np.array(supp1,dtype=int)  
    #print('box,k:',box,k)
    #print('supp1:',supp1)

    xpg,fxpg=pg.main(ctr,n,y,X[:,supp1],k,num_runs=5) # xpg is in the reduced dim.
    #print('xpg:',xpg)
    xout=np.zeros((p,1))
    xout[supp1]=xpg
    fout=fxpg

    return xout, fout

#============================================================================================================================
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

#============================================================================================================================
def getchildboxes(num_cuts,par_dir,xlb,V):
    """ generate child boxes
    
    """
    import numpy as np

    def getallvectices(ii,n,K,V,num_cuts,ids_array,S):
        """ generate 2^(num_cuts) boxes after cutting along the provided direction

           ii : num_cuts
           n : num_cuts
           K : indices of the directions to be cut
           V : the selected parent box to be partitioned with the attributes: box array, num 0, num 1, and num 2 flags
           ids_array : array of dim 1 x num_cuts to save the 
           S : list of tuples/boxes to save the box attributes
        """
        for i in range(2):
            ids_array[ii] = i
            if ii > 0:
                S=getallvectices(ii-1,n,K,V,num_cuts,ids_array,S)

            else:
                
                # V[0]= box array, V[1]= # 0 flag, V[2]= # 1 flag, V[3]= # 2 flag 
                Vtemp=V.copy() # make a copy of the original box
                for j in range(num_cuts):
                    if ids_array[j] == 0:
                        Vtemp[K[j]] = 0 
                        Vtemp[1] += 1    # increase the no. of flag 0 by 1
                        Vtemp[2] -= 1    # decrease the no. of flag 1 by 1

                    elif ids_array[j] == 1:
                        Vtemp[K[j]] = 2 
                        Vtemp[3] += 1    # increase the no. of flag 2 by 1
                        Vtemp[2] -= 1    # decrease the no. of flag 1 by 1

                    
                S.append(Vtemp) # add the new box to the list


        return S


    S = []  # list of tuples/boxes with length 2^num_cuts 
    cut_dir=getklargest(np.abs(xlb[par_dir]),num_cuts) # the indices of the coordinate direction to be cut along
    ids_array=np.zeros(num_cuts) 
    S=getallvectices(num_cuts,num_cuts,cut_dir,V,num_cuts,ids_array,S)
    num_child=2 ** num_cuts 

    return num_child, cut_dir, S
#============================================================================================================================