import numpy as np 

def compute_dcg(order, cat_size): 
    rel = list(order)
    
    for i in range(len(order)): 
        #compute relevance values
        rel[i] = ((cat_size+1) - order[i])/2 + 0.5
        
    dcg = 0 
    for i in range(len(rel)): 
        #compute dcg based on relevance 
        dcg += (2**(rel[i])-1)/(np.log2((i+1)+1))
    
    return dcg 