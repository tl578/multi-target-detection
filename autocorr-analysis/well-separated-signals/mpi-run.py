#!/usr/bin/env python
from mpi4py import MPI
from subprocess import *
import numpy as np

# initialize MPI
comm = MPI.COMM_WORLD
nproc = comm.Get_size()
rank = comm.Get_rank()

num_data = np.int(10**6)
sigma = np.arange(-1.5, 0.6, 0.1)
snr = (1./10**sigma)**2
length = len(snr)
num_trial = 10
int_max = 2**31-1

for d in range(length):
    if (np.abs(sigma[d]) < 1e-10):
        data_dir = "Data/repos/sigma_0.0"
    else:
        data_dir = "Data/repos/sigma_{0:.1f}".format(sigma[d])

    if (rank == 0):
        cmd = "mkdir -p {0:s}".format(data_dir)
        p = Popen(cmd, shell=True)
        p.wait()
    
    comm.barrier()
    
    psf_file = "{0:s}/psf-{1:d}.dat".format(data_dir, rank)
    autocorr_file = "{0:s}/autocorr-{1:d}.dat".format(data_dir, rank)
    log_file = "{0:s}/log-{1:d}.log".format(data_dir, rank)
    cmd = "python make-micrograph.py {0:f} {1:d} {2:s} {3:s} > {4:s}".format(snr[d], num_data, psf_file, autocorr_file, log_file)
    p = Popen(cmd, shell=True)
    p.wait()

    for s in range(num_trial):
        outfile = "{0:s}/pamameters-{1:03d}-{2:d}.mat".format(data_dir, rank, s)
        rand_seed = np.random.randint(int_max)
        cmd = 'matlab -nodisplay -nosplash -nodesktop -r "main(\'{0:d}\', \'{1:s}\', \'{2:s}\')" >> {3:s}'.format(\
            rand_seed, autocorr_file, outfile, log_file)
        p = Popen(cmd, shell=True)
        p.wait()
