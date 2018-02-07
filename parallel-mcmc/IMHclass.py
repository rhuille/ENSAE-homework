import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from time import time, sleep
import random
import pandas as pd
import itertools
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize

class IMH(object):
    """
    Class IMH
    #########
    
    Implementation of IMH (Independent Metropolis Hasting) algorithm explained in :
    << Using parallel computation to improve Independent Metropolis Hasting based estimation >>
    Jacob and al. 2011
    
    Arguments
    ---------
    - omega (function) : density function of the distribution to simulate
    - gen_candidate (function(N) ) : return an array of N values generated from the candidate distribution

    - x0 (float) : initialisation
    - N (int) : length of the markov chain
    - njobs (int) : the number of jobs to run in parallel 
    - method (str) : 'simple' or 'parallel'

    Methods
    -------
    - fit_simple : implementation of the fundamental version of the IMH algorithm 
    - fit_block : implementation of the block version of the IMH algorithm 
    - fit : main method interface
    """
    
    def __init__(self, omega, gen_candidate) :
        self.omega = np.vectorize(omega)
        self.gen_candidate = gen_candidate

    def fit_simple(self, b=0):

        n = self.y.shape[0] # either equal to : 
                            # - self.N (when method ='simple')
                            # - self.nblocks (when method ='parallel')
        
        
        i = np.random.permutation(self.nblocks)
        i = [0]+list(i)
        # go
        current = 0
        weight = np.full(fill_value = 0, shape = n)
        for candidate in range(1, len(i)):
            ratio = min(1, self.omega_y[i[candidate],b]/self.omega_y[i[current],b])
            u = np.random.binomial(1,ratio)
            current += u*(candidate-current) # current is updated to candidate if u = 1 
                                             # and stay current if u = 0

            weight[i[current]] += 1 # add current value to the chain
        
        return weight

    
    def fit_block(self):
        
        weight = np.full(fill_value = 0, shape = self.y.shape)
        with Pool(self.njobs) as p:
            for b in range(self.B): 

                weight_block = np.array(p.map(self.fit_simple, [b]*self.njobs))
                weight[:,b] = weight_block.sum(axis = 0)

                if b < self.B-1 : # init the next block picking randomly in the current block
                    p_ = weight[:,b]/weight[:,b].sum()
                    self.y[0,b+1] = np.random.choice(self.y[:,b], size=1, p = p_)
                    self.omega_y[0,b+1] = self.omega(self.y[0,b+1])
        return weight


    def fit(self, x0, N, method = 'simple', B = 1, njobs = 1):
        self.B = B
        self.nblocks = int(N/B)
        self.njobs = njobs
        self.N = N     

        if method == "simple":            
            # (1) creation of candidate sample
            self.y = np.reshape(self.gen_candidate(self.N-1), newshape=(self.N-1,1))
            # (2) add init value
            self.y = np.append([[x0]], self.y, axis = 0)
            # (3) computation of omega values
            self.omega_y = self.omega(self.y)
            # (4) computation of weight with IMH algo
            self.weights = self.fit_simple()
            
        elif method == 'parallel':
            with Pool(njobs) as pool:
                # (1) creation of candidate sample
                self.y = np.array(pool.map(self.gen_candidate,[int(self.N/njobs)]*njobs))
                # (3) computation of omega values
                self.omega_y = np.array(pool.map(self.omega, list(self.y)))
            
            # reshape : this is usefull when nblocks!=njobs
            self.y = np.reshape(self.y, newshape=(self.nblocks, self.B))
            self.omega_y = np.reshape(self.omega_y,newshape=(self.nblocks, self.B))
            # (2) add init value
            self.y = np.append(np.full((1,self.B), x0), self.y, axis = 0)
            self.omega_y = np.append(np.full((1,self.B), self.omega(x0)), self.omega_y, axis = 0)
            
            # (4) computation of weight with IMH algo
            self.weights = self.fit_block() # computation of weight
        
        self.weights = np.reshape(self.weights,newshape=self.y.shape)
        self.expectation = np.average(self.y[1:,:], weights= self.weights[1:,:])
        
        return self
