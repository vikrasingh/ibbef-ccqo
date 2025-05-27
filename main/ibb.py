def main(p,n,y,X,A,b,c,k,xrelax,num_cuts=1,max_cputime=600,max_df_iter=500):
    """IBB+ with lb using recycled echelon form

       xrelax: xRelaxedOpt
    """
    import sys
    import numpy as np
    from scipy import linalg
    from scipy.optimize import minimize
    import heapq
    import time

    import getklargest as gkl

    epsilon=sys.float_info.epsilon    
    cpustart=time.process_time() # save the starting cpu time
    stopflag = 0   # initialization  
    absxrelax=np.abs(xrelax) # absolute value of xrelax array
    print('A:',A)
    print('b:',b)
    #print('X:',X)
    #print('xrelax:',xrelax)

    num_iter=0
    num_box=1
    xbest=np.zeros((p,1)) # intialize xbest
    supp0,xbest[supp0]=gkl(absxrelax,k)
    fbest=fx(xbest[supp0],A[np.ix_(supp0,supp0)],b[supp0],c)
    last_fbest_update_iter=num_iter
    #print('xbest,fbest:',xbest,fbest)
    B0=np.ones(p,dtype=int) # initial box of integer ones
    
    xlb, fxlb=quad_min(p,np.array(range(p),dtype=int),A,b,c,xrelax)
    #print('xlb :',xlb)
    #print('fxlb:',fxlb)
    #print('size fxlb:',np.size(fxlb))
    print('p type:',np.dtype(p))
    # Initialize the list L
    L = []
    box_age_ctr=0 # unique for every box we are adding, the oldest box will have the smallest counter
    heapq.heappush( L, ( fxlb, box_age_ctr , B0 , 0, p, 0,  xlb) )  # add the intial box to the list

    while True:

        num_iter += 1
        cpulapsed=time.process_time()-cpustart
        # check for convergence criteria
        if num_box==0:
            print('alg. converged')
            break

        if cpulapsed>=max_cputime:
            print('max CPU time reached')
            stopflag = 6 
            break

        if (num_iter-last_fbest_update_iter)>max_df_iter:
            print('fbest did not get updated for the given no. of iter')
            stopflag = 4
            break
        
        
        # select a new box to process
        Y=heapq.heappop(L) # V[0]=fxlb, V[1]=box, V[2]=#0, V[3]=#1, V[4]=#2, V[5]=xlb
        num_box -= 1 
        #print('Y:',Y[0],Y[1],Y[2],Y[3],Y[4],Y[5])
        is_fbest_updated, box_age_ctr, num_box, fbest, xbest, L=branch(p,n,y,X,A,b,c,k,L,Y,xbest,fbest,xrelax,absxrelax,num_box,box_age_ctr,num_cuts)

        if is_fbest_updated==1:
            last_fbest_update_iter = num_iter

    
    # final output
    xout=xbest
    fout=fbest

    return stopflag, xout, fout

#============================================================================================================================
def branch(p,n,y,X,A,b,c,k,L,Y,xbest,fbest,xrelax,absxrelax,num_box,box_age_ctr,num_cuts):
    """
    
    """
    import numpy as np
    import heapq

    is_fbest_updated = 0 # flag check if the fbest gets updated for this call of branch
    # branch and process the child boxes
    par_dir=np.arange(Y[2].size)[Y[2]==1]  # indices of partition direction in increasing order
    num_child, cut_dir, child_boxes=getchildboxes(num_cuts,par_dir,Y[6],Y[2:6])

    for j in range(num_child):
        #print('child box:',child_boxes[j])
        # check DC2
        if child_boxes[j][3]>k:  # If no. of flag 2 > k
            continue # discard the box, go to the next child box

        # check DC5
        if (child_boxes[j][2]+child_boxes[j][3])==k: # if no. of flag 1 + flag 2 = k
            supp=np.where(child_boxes[j][0]!=0)[0] # indices of flag 1 and flag 2
            xlb, fxlb=quad_min(p,supp,A,b,c,xrelax)
            # update fbest if possible
            if fxlb<fbest:
                fbest = fxlb
                xbest = xlb
                is_fbest_updated = 1
            
            continue # discard the box, go to the next child box

        if num_cuts>1: 
            if (child_boxes[j][2]+child_boxes[j][3]) < k: # if no. of flag 1 + flag 2 < k
                continue # discard the box, go to the next child box

        # find a feasible point
        xtilde, fxtilde=getfeasiblept(p,n,y,X,A,b,c,k,child_boxes[j][0],xrelax,absxrelax)

        # update fbest if possible
        if fxtilde<fbest:
            fbest = fxtilde
            xbest = xtilde
            is_fbest_updated = 1

        # find lb f(V)
        supp=np.where(child_boxes[j][0]!=0)[0] # indices of flag 1 and flag 2
        xlb, fxlb=quad_min(p,supp,A,b,c,xrelax)

        # check DC1
        if fbest<=fxlb:  # <= because we only want to find one global optimal
            continue  # discard the box, go to the next child box

        # add the box to the list for further processing
        num_box += 1
        box_age_ctr += 1
        #V2temp =(fxlb, box_age_ctr, child_boxes[j][0] , child_boxes[j][1], child_boxes[j][2] ,child_boxes[j][3] , xlb)
        heapq.heappush( L, (fxlb, box_age_ctr, child_boxes[j][0] , child_boxes[j][1], child_boxes[j][2] ,child_boxes[j][3] , xlb) ) # push the box in the list

    return is_fbest_updated, box_age_ctr, num_box, fbest, xbest, L   


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
    import main.projgrad as pg

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
def getchildboxes(num_cuts,par_dir,xlb,V):
    """ generate child boxes
    
    """
    import numpy as np
    import getklargest as gkl

    def getallvectices(ii,K,V,num_cuts,ids_array,S):
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
                S=getallvectices(ii-1,K,V,num_cuts,ids_array,S)

            else:
                
                # V[0]= box array, V[1]= # 0 flag, V[2]= # 1 flag, V[3]= # 2 flag 
                V0 = np.array(V[0])   # make a list of the tuple entries
                num0flag = V[1]
                num1flag = V[2]
                num2flag = V[3]
                for j in range(num_cuts):
                    if ids_array[j] == 0:
                        V0[K[j]] = 0 
                        num0flag += 1    # increase the no. of flag 0 by 1
                        num1flag -= 1    # decrease the no. of flag 1 by 1

                    elif ids_array[j] == 1:
                        V0[K[j]] = 2 
                        num2flag += 1    # increase the no. of flag 2 by 1
                        num1flag -= 1    # decrease the no. of flag 1 by 1

                Vtemp = (V0, num0flag, num1flag, num2flag)     
                S.append(Vtemp) # add the new box to the list


        return S


    S = []  # list of tuples/boxes with length 2^num_cuts 
    cut_dir_local,_=gkl(np.abs(xlb[par_dir]),num_cuts) # the indices of the coordinate direction to be cut along
    cut_dir = par_dir[cut_dir_local]
    #print('cut_dir:',cut_dir)
    #cut_dir=np.array(cut_dir,dtype=int)
    ids_array=np.zeros(num_cuts) 
    S=getallvectices(num_cuts-1,cut_dir,V,num_cuts,ids_array,S)
    num_child=2 ** num_cuts 

    return num_child, cut_dir, S
#============================================================================================================================