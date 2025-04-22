# solving bss using mio
def main(p,n,y,X,Hess,lin,const,k,low,up,xrelax):
    """ Solving BSS using Gurobi
        MATLAB implementation of the bss in R package, following the code from Ryan Tibshirani's github page
       https://github.com/ryantibs/best-subset/blob/master/bestsubset/R/bs.R
       which further uses the setup from Bertsimas et.al.(2015) BSS via modern optimization lens. reference
       definition of each stopflag status 
       https://www.gurobi.com/documentation/10.0/refman/optimization_status_codes.html#sec:StatusCodes

    """
    import gurobipy as gp
    from gurobipy import GRB
    import numpy as np
    import scipy.sparse as sp

    def dense_optimize(rows, cols, c, Q, A, sense, rhs, lb, ub, vtype, solution):
        model = gp.Model()

        # Add variables to model
        vars = []
        for j in range(cols):
            vars.append(model.addVar(lb=lb[j], ub=ub[j], vtype=vtype[j]))

        # Populate A matrix
        for i in range(rows):
            expr = gp.LinExpr()
            for j in range(cols):
                if A[i][j] != 0:
                    expr += A[i][j] * vars[j]
            model.addLConstr(expr, sense[i], rhs[i])

        # Populate objective
        obj = gp.QuadExpr()
        for i in range(cols):
            for j in range(cols):
                #print('Qij:',Q[i][j])
                if Q[i][j] != 0:
                    obj += Q[i][j] * vars[i] * vars[j]
        for j in range(cols):
            if c[j] != 0:
                obj += c[j] * vars[j]
        print('obj:',obj)        
        model.setObjective(obj)

        # Solve
        model.optimize()

        # Write model to a file
        #model.write("bss.lp")
        return model
    
        print('model:',model)
        if model.status == GRB.OPTIMAL:
            x = model.getAttr("X", vars)
            for i in range(cols):
                solution[i] = x[i]
            return True
        else:
            return False
        
    # Put model data into dense matrices
    XX = X.T @ X
    I = np.eye(p)
    Atemp = np.vstack(( np.hstack((I,-up * I)) , np.hstack((-I,low * I)) ))
    rvec = np.hstack((np.zeros((1,p)),np.ones((1,p))))
    A = np.vstack((Atemp,rvec))  # A x <= rhs , A is the linear constraint matrix
    sense = list((2*p+1)*GRB.LESS_EQUAL)  
    c = np.hstack((-2*(y.T @ X)[0],np.zeros(p))) # the linear term in the objective function
    rhs = np.hstack((np.zeros(2*p),[k]))
    block_diag_matrix=sp.block_diag((XX,np.zeros((p,p))))
    Q = block_diag_matrix.toarray() # convert sparse matrix to a dense matrix
    lb = np.hstack((low.T[0],np.zeros(p)))
    ub = np.hstack((up.T[0],np.ones(p)))
    vtype = np.hstack( (list(p*GRB.CONTINUOUS),list(p*GRB.BINARY)) )
    fespt=np.hstack((np.ones(k),np.zeros(p-k))) # feasible point with k non zero entries
    sol = np.hstack((fespt,fespt!=0))
    num_rows=2*p+1
    num_cols=2*p
    # Optimize
    optimized_model = dense_optimize(num_rows, num_cols, c, Q, A, sense, rhs, lb, ub, vtype, sol)

    if optimized_model.status==GRB.OPTIMAL:
        x = optimized_model.getAttr("X", vars) 
        print('xmio:',x[0:p])

    xout = x
    print('bx:', xout@lin)
    print('xAx:',xout@Hess@xout.T)
    fxout = 0.5 * (xout @ Hess @ xout.T) + xout @ lin + const

    return xout, fxout


