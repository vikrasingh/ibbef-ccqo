import numpy as np

def main(data):
    """ find relative gap
        100(f - f*)/f*
        where f* is the best solution found by any solver
         and  f is the solution found by a particular solver 
    
    """
    (m,n) = np.shape(data)
    out = np.zeros((m,n))
    for i in range(m):
        opt = np.min(data[i,:]) 
        out[i,:] = data[i,:] - opt
        rel_gap = 100*(out[i,:]/opt) 
        out[i,:] = rel_gap

    return out

