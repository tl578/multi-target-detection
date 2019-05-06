import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sigma = np.arange(-1.5, 0.6, 0.1)
length = len(sigma)
num_run = 20

fp = open("../signal.dat", "r")
signal = np.array(fp.readline().split()).astype(np.double)
fp.close()

L = len(signal)
rescale = np.sqrt(L/np.dot(signal, signal))
signal *= rescale

inv_s2 = 1./np.dot(signal, signal)
log_rmse = np.zeros(length)

fout = open("Data/output-EM_arbitrary-spacing-distribution.dat", "w")
for i in range(length):
    rmse = np.zeros(num_run)
    rho0 = np.zeros(num_run)
    if (np.abs(sigma[i]) < 1.e-10):
        data_dir = "Data/repos/sigma_0.0"
    else:
        data_dir = "Data/repos/sigma_{0:.1f}".format(sigma[i])    
    filelist = os.listdir(data_dir)
    for r in range(num_run):
        file_count = 0
        identifier = "_{0:03d}-".format(r)
        for filename in filelist:
            if (identifier not in filename):
                continue

            infile = "{0:s}/{1:s}".format(data_dir, filename)
            fp = open(infile, "r")
            lines = fp.readlines()
            fp.close()

            num_sol = len(lines) // 3
            if (file_count == 0):
                max_LL = np.double(lines[-1].split()[0])
                sol = np.array(lines[3*num_sol-3].split()).astype(np.double)
                prob = np.array(lines[3*num_sol-2].split()).astype(np.double)
                psf = np.array(lines[3*num_sol-1].split()).astype(np.double)
            else:
                LL = np.double(lines[-1].split()[0])
                if (LL > max_LL):
                    sol = np.array(lines[3*num_sol-3].split()).astype(np.double)
                    prob = np.array(lines[3*num_sol-2].split()).astype(np.double)
                    psf = np.array(lines[3*num_sol-1].split()).astype(np.double)
                    max_LL = LL

            file_count += 1

        diff = sol - signal
        rmse[r] = np.sqrt(np.dot(diff, diff)*inv_s2)
        rho0[r] = prob[L]*L

        out_str = "{0:d}\n{1:1.6e} ".format(i, sol[0])
        for t in range(1, L):
            out_str = "{0:s}{1:1.6e} ".format(out_str, sol[t])
        out_str = "{0:s}\n{1:1.6e}".format(out_str, rho0[r])
        for t in range(L+1):
            out_str = "{0:s} {1:1.6e}".format(out_str, psf[t])
        out_str = "{0:s}\n".format(out_str)
        fout.write(out_str)

fout.close()
