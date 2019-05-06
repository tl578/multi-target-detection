#!/usr/bin/env python
from mpi4py import MPI
from subprocess import *
import numpy as np
import sys

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
rank = comm.Get_rank()

sigma = np.double(sys.argv[1])
num_data = np.int(sys.argv[2])
snr = (1./10**sigma)**2

if (np.abs(sigma) < 1e-10):
    data_dir = "Data/repos/sigma_0.0"
else:
    data_dir = "Data/repos/sigma_{0:.1f}".format(sigma)

if (rank == 0):
    cmd = "mkdir -p {0:s}".format(data_dir)
    p = Popen(cmd, shell=True)
    p.wait()

comm.barrier()

psf_file = "{0:s}/psf-{1:d}.dat".format(data_dir, rank)
micrograph_file = "{0:s}/micrograph-{1:d}.bin".format(data_dir, rank)
log_file = "{0:s}/log-{1:d}.log".format(data_dir, rank)
cmd = "python make-micrograph.py {0:f} {1:d} {2:s} {3:s} > {4:s}".format(snr, num_data, psf_file, micrograph_file, log_file)

p = Popen(cmd, shell=True)
p.wait()
