## multi-target detection
This repository contains programs to solve the multi-target detection (MTD) problem with autocorrelation analysis and an approximate EM algorithm. The codes for autocorrelation analysis are written in Matlab, whereas those for the EM approach are written in C. 
The [FFTW3 library](http://www.fftw.org) and the [Manopt toolbox](https://www.manopt.org) are needed to run the programs. 

For details of the MTD problem and our algorithms, see
> "Multi-target detection with an arbitrary spacing distribution", T.-Y. Lan, T. Bendory, N. Boumal & A. Singer

To run autocorrelation analysis:
```
mpirun -n [nproc] python mpi-run.py > log.log &
```

To run the EM method:
```
python run.py > log.log &
```

Send comments, bug reports, to: tiyenlan@princeton.edu
