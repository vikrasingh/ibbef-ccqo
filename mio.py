# solving bss using mio
def main():
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
    
    	