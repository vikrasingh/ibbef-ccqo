def main(given_list,select_k):
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