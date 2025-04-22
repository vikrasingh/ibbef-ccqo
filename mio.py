# solving bss using mio
def main(p,n,y,X,A,b,c,k,low,up,xrelax):
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
                print('Qij:',Q[i][j])
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
        model.write("bss.lp")

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
    Atemp = np.vstack(( np.hstack((I,-up * I)) , np.hstack((I,-low * I)) ))
    print('Atemp:',Atemp)
    rvec = np.hstack((np.zeros((1,p)),np.ones((1,p))))
    print('rvec:',rvec)
    A = np.vstack((Atemp,rvec))  # A x <= rhs , A is the linear constraint matrix
    sense = list((2*p+1)*GRB.LESS_EQUAL)   
    c = np.vstack((-2*(X.T @ y),np.zeros((p,1)))) # the linear term in the objective function
    rhs = np.vstack((np.zeros((2*p,1)),k))
    #Q = sp.csr_matrix( sp.block_diag((XX,np.zeros((p,p)))) )
    block_diag_matrix=sp.block_diag((XX,np.zeros((p,p))))
    Q = block_diag_matrix.toarray()
    print('Q:',Q)
    print('Qij:',Q[0][0])
    lb = np.vstack((low,np.zeros((p,1))))
    ub = np.vstack((up,np.zeros((p,1))))
    vtype = np.hstack( (list(p*GRB.CONTINUOUS),list(p*GRB.BINARY)) )
    sol = np.vstack((xrelax,xrelax!=0))
    num_rows=2*p+1
    num_cols=2*p
    # Optimize
    success = dense_optimize(num_rows, num_cols, c, Q, A, sense, rhs, lb, ub, vtype, sol)

    if success:
        print('xmio:',sol[0])


