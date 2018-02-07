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

# Parameters
def omega(x):  # densité de la loi normale
    return((1+x**2)*np.exp(-(x)**2/2))

def gen_candidate(N):
    np.random.seed() # for seeding the generator
    return np.random.standard_cauchy(size = N)

N_range = [4*1e3,8*1e3,4*1e4,8*1e4,4*1e5,8*1e5,4*1e6]
log_range = list(map(np.log10, N_range))
loops = 61