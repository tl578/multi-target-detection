from subprocess import *
import numpy as np

num_data = np.int(10**6)
sigma = np.arange(-1.5, 0.6, 0.1)
snr = (1./10**sigma)**2
length = len(snr)

nproc = 50
num_instance = 20
num_trial = 10

for i in range(length):
    cmd = "mpirun -n {0:d} python mpi-make-data.py {1:f} {2:d}".format(num_instance, sigma[i], num_data)
    p = Popen(cmd, shell=True)
    p.wait()

    if (np.abs(sigma[i]) < 1e-10):
        data_dir = "Data/repos/sigma_0.0"
    else:
        data_dir = "Data/repos/sigma_{0:.1f}".format(sigma[i])

    for j in range(num_instance):
        micrograph_file = "{0:s}/micrograph-{1:d}.bin".format(data_dir, j)
        log_file = "{0:s}/log-{1:d}.log".format(data_dir, j)
        cmd = "mv {0:s} micrograph.bin".format(micrograph_file)
        p = Popen(cmd, shell=True)
        p.wait()

        for k in range(num_trial):
            cmd = "mpirun -np {0:d} ./recon >> {1:s}".format(nproc, log_file)
            p = Popen(cmd, shell=True)
            p.wait()
            
            cmd = "mv Data/output.dat {0:s}/output-log10sigma_{1:.1f}_{2:03d}-{3:d}.dat".format(data_dir, sigma[i], j, k)
            p = Popen(cmd, shell=True)
            p.wait()
