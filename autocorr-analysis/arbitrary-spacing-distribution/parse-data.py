import matplotlib.pyplot as plt
import numpy as np
import sys

sigma = np.arange(-1.5, 0.6, 0.1)
length = len(sigma)
num_run = 20
num_trial = 10

fp = open("../signal.dat", "r")
signal = np.array(fp.readline().split()).astype(np.double)
fp.close()

L = len(signal)
rescale = np.sqrt(L/np.dot(signal, signal))
signal *= rescale

inv_s2 = 1./np.dot(signal, signal)
log_rmse = np.zeros(length)
n_max = L // 2

fout = open("Data/output-autocorr_arbitrary-spacing-distribution.dat", "w")
for i in range(length):
    if (np.abs(sigma[i]) < 1.e-10):
        infile = "Data/repos/sigma_0.0/output.dat"
    else:
        infile = "Data/repos/sigma_{0:.1f}/output.dat".format(sigma[i])

    fp = open(infile, "r")
    lines = fp.readlines()
    fp.close()
    
    rmse = np.zeros(num_run)
    rho0 = np.zeros(num_run)
    for r in range(num_run):
        idx_offset = r*num_trial*(2*n_max+1)
        cost_min = np.double(lines[idx_offset + 2*n_max].strip())
        idx = 0
        for s in range(1, num_trial):
            cur_cost = np.double(lines[idx_offset + (s+1)*(2*n_max+1) - 1].strip())
            if (cur_cost < cost_min):
                cost_min = cur_cost
                idx = s

        sol = np.array(lines[idx_offset + (idx+1)*(2*n_max+1) - 3].split()).astype(np.double)
        diff = sol - signal
        rmse[r] = np.sqrt(np.dot(diff, diff)*inv_s2)

        rho = np.array(lines[idx_offset + (idx+1)*(2*n_max+1) - 2].split()).astype(np.double)
        rho[0] *= L
        
        if_skip = 0
        for t in range(L):
            if (rho[t] > 1.):
                if_skip = 1
        if (if_skip == 1):
            continue

        out_str = "{0:d}\n{1:1.6e} ".format(i, sol[0]) 
        for t in range(1, L):
            out_str = "{0:s}{1:1.6e} ".format(out_str, sol[t])
        out_str = "{0:s}\n{1:1.6e} ".format(out_str, rho[0])
        for t in range(1, L):
            out_str = "{0:s}{1:1.6e} ".format(out_str, rho[t])
        out_str = "{0:s}\n".format(out_str)
        fout.write(out_str)

fout.close()
