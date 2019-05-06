"""
generate a 1D micrograph by assuming a minimum particle seperation
"""

import numpy as np
import sys

fp = open("../signal.dat", "r")
signal = np.array(fp.readline().split()).astype(np.double)
L = len(signal)
rescale = 1/np.sqrt(np.dot(signal, signal)/L)
signal *= rescale
fp.close()

## mean square of signal to noise variance
snr = np.double(sys.argv[1])
## micrograph length in unit of signal length
N = np.int64(sys.argv[2])
## number density of signal occurence per signal length
density = 0.3
## locations of the starting pixel of each signal occurence
locations = []
## minimum separation: no multiple signals in a window when W > L-1
W = L-1

Npix = N*L
mask = np.zeros(Npix, dtype=np.int)
num_particle = np.int(np.round(N*density))
trial_max = 5*num_particle
particle_count = 0
candidates = np.random.randint(0, Npix - L + 1, trial_max)
for k in range(trial_max):
    s = candidates[k]
    idx_min = np.max([0, s-W])
    idx_max = s+W+L
    if (np.sum(mask[idx_min:idx_max]) > 0):
        continue

    mask[s:s+L] = 1
    locations.append(s)
    particle_count += 1
    if (particle_count == num_particle):
        break

locations = np.sort(np.array(locations).astype(np.int))
print("trial count = {0:d}, maximum number of trial = {1:d}".format(k+1, trial_max))
print("signal count = {0:d}, micrograph length = {1:d}\n".format(particle_count, N)) 
if (particle_count < num_particle):
    print("fail to place signals!!")
    exit()

## pair seperation function
max_dist = 0
for k in range(1, num_particle):
    dist = locations[k] - locations[k-1] - L
    if (dist > max_dist):
        max_dist = dist

psf = np.zeros(max_dist+1)
for k in range(1, num_particle):
    dist = locations[k] - locations[k-1] - L
    psf[dist] += 1

psf /= np.sum(psf)
psf_file = sys.argv[3]
fp = open(psf_file, "w")
for i in range(max_dist+1):
    fp.write("{0:1.5e}\n".format(psf[i]))
fp.close()

## 1D noisy micrograph
micrograph = np.zeros(Npix)
for k in range(num_particle):
    idx_min = locations[k]
    idx_max = idx_min + L
    micrograph[idx_min:idx_max] += signal

sigma = np.sqrt(1/snr)
micrograph += np.random.randn(Npix)*sigma

## calculate autocorrelations
y = np.zeros(Npix + L)
y[0:Npix] = np.copy(micrograph)

## first order
ay1 = np.sum(y)

## second order
ay2 = np.zeros(L)
for l in range(L):
    ay2[l] = np.sum(y[0:Npix]*y[l:Npix+l])

## third order
ay3 = np.zeros((L, L))
for l2 in range(L):
    for l1 in range(l2+1):
        ay3[l1][l2] = np.sum(y[0:Npix]*y[l1:Npix+l1]*y[l2:Npix+l2])

autocorr_file = sys.argv[4]
fp = open(autocorr_file, "w")
fp.write("{0:d} {1:f}\n".format(L, sigma))
fp.write("{0:1.7e}\n".format(ay1/Npix))
for l1 in range(L):
    fp.write("{0:1.7e}\n".format(ay2[l1]/Npix))
for l2 in range(L):
    for l1 in range(l2+1):
        fp.write("{0:1.7e}\n".format(ay3[l1][l2]/Npix))
fp.close()

## statistics for sanity check
trans_sz = 2*L
win_count = np.zeros(N, dtype=np.int)
win2particle = np.zeros((N, 2), dtype=np.int)
for k in range(num_particle):
    idx_min = locations[k]
    idx_max = idx_min + L
    win_idx = idx_min // L
    win2particle[win_idx][win_count[win_idx]] = k
    win_count[win_idx] += 1
    if (idx_min % L != 0):
        win2particle[win_idx + 1][win_count[win_idx + 1]] = k
        win_count[win_idx + 1] += 1

print("number of empty windows = {0:d}".format(np.sum((win_count == 0))))
print("number of one-particle windows = {0:d}".format(np.sum((win_count == 1))))
print("number of two-particle windows = {0:d}\n".format(np.sum((win_count == 2))))

ct = trans_sz
state_map = np.ones((L, L), dtype=np.int)*(-1)
for i in range(L):
    for j in range(i+1, L):
        state_map[i][j] = ct
        ct += 1

num_state = trans_sz + L*(L-1)//2
state_count = np.zeros(num_state, dtype=np.int)
for n in range(N):
    idx_min = n*L
    idx_max = idx_min + L
    if (win_count[n] == 0):
        state_count[0] += 1
    elif (win_count[n] == 1):
        k = win2particle[n][0]
        signal_idx_start = locations[k]
        if (signal_idx_start >= idx_min):
            trans_mode = L - (signal_idx_start - idx_min)
        else:
            trans_mode = L + (idx_min - signal_idx_start)
        if (trans_mode <= 0 or trans_mode >= trans_sz):
            print("error in finding trans_mode!!")
        state_count[trans_mode] += 1
    else:
        k0 = win2particle[n][0]
        k1 = win2particle[n][1]
        signal_idx_start0 = locations[k0]
        signal_idx_start1 = locations[k1]        
        i0 = signal_idx_start0 - (idx_min - L) - 1
        j0 = signal_idx_start1 - idx_min
        if (i0 < 0 or j0 < 0 or i0 >= L or j0 >= L or state_map[i0][j0] < 0):
            print("error in locating two particles!!")
        state_count[state_map[i0][j0]] += 1

print("SNR = {0:f}\n".format(snr))
print("probabilities of empty & one-particle windows")
out_str = ""
for i in range(L):
    out_str = "{0:s}\t{1:f}".format(out_str, state_count[i]/np.double(N))
print(out_str)
out_str = ""
for i in range(L, trans_sz):
    out_str = "{0:s}\t{1:f}".format(out_str, state_count[i]/np.double(N))
print("{0:s}\n".format(out_str))

print("probabilities of two-particle windows")
for i in range(L):
    out_str = ""
    for j in range(L):
        if (state_map[i][j] < 0):
            out_str = "{0:s}\t{1:s}".format(out_str, "".ljust(10))
        else:
            out_str = "{0:s}\t{1:f}".format(out_str, state_count[state_map[i][j]]/np.double(N))
    print(out_str)
