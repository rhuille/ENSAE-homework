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
from itertools import compress
import pickle
from IMHclass import IMH
from parameters import *

if __name__ == "__main__":
	IMH_computer  = IMH(omega, gen_candidate)

	variances_list = pd.DataFrame()
	times_list = pd.DataFrame()
	for loop in range(loops):
		times_simple = []
		variances_simple = []
		for numb in tqdm(N_range):
			t0_simple = time()
			IMH_computer.fit(x0 = 0, N = int(numb), method='simple')
			t1_simple = time()
			times_simple += [t1_simple-t0_simple]
			y_simple = np.repeat(IMH_computer.y.reshape(-1), IMH_computer.weights.reshape(-1))
			np.random.shuffle(y_simple)
			MC_simple = np.cumsum(y_simple)/np.arange(1, len(y_simple) + 1)
			variances_simple += [np.var(MC_simple)]

		times_parallel = []
		variances_parallel = []
		for numb in tqdm(N_range):
			t0_parallel = time()
			IMH_computer.fit(x0 = 0, N = int(numb), method = 'parallel', B = 10, njobs = 4)
			t1_parallel = time()
			times_parallel += [t1_parallel-t0_parallel]
			y_parallel = np.repeat(IMH_computer.y[1:,:].reshape(-1), IMH_computer.weights[1:,:].reshape(-1))
			np.random.shuffle(y_parallel)
			MC_parallel = np.cumsum(y_parallel)/np.arange(1, len(y_parallel) + 1)
			y_simple = y_parallel
			variances_parallel += [np.var(MC_parallel)]

		times = []
		variances = []
		for i in range(len(times_simple)):
			times += [times_parallel[i]/times_simple[i]]
			variances += [variances_parallel[i]/variances_simple[i]]

		with open("Parallel/Loops/" + "times_simple" + str(loop) + ".dat", "wb") as f:
			pickle.dump(times_simple, f)
		with open("Parallel/Loops/" + "times_parallel" + str(loop) + ".dat", "wb") as f:
			pickle.dump(times_parallel, f)
		with open("Parallel/Loops/" + "variances_simple" + str(loop) + ".dat", "wb") as f:
			pickle.dump(variances_simple, f)
		with open("Parallel/Loops/" + "variances_parallel" + str(loop) + ".dat", "wb") as f:
			pickle.dump(variances_parallel, f)

		times_list[str(loop)] = times
		variances_list[str(loop)] = variances

	y_time = []
	y_var = []
	for i in range(len(times)):
		y_time += [np.mean([times_list[str(loop)][i] for loop in range(loops)])]
		y_var += [np.mean([variances_list[str(loop)][i] for loop in range(loops)])]

	with open("Parallel/Plots/" + "y_time.dat", "wb") as f:
		pickle.dump(y_time, f)
	with open("Parallel/Plots/" + "y_var.dat", "wb") as f:
		pickle.dump(y_var, f)