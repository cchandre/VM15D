###################################################################################################
##              Dictionary of parameters: https://github.com/cchandre/VM15D                      ##
###################################################################################################

import numpy as xp

Tf = 80
alpha = 1
lam = 2

integrator_kinetic = 'position-Verlet'
nsteps = 50
integrator_fluid = 'DOP853'
precision = 1e-11

n_casimirs = 3

Lz = 2 * xp.pi
Lvx = 6
Lvz = 6
Nz = 2**8
Nvx = 2**8
Nvz = 2**8

A = 1e-4
k = 0.5
Tx = 1
Tz = 0.1
f_init = lambda z, vx, vz: (1 - A * xp.cos(k * z)) * xp.exp(-vx**2 / (2 * Tx)) / xp.sqrt(2 * xp.pi * Tx) * xp.exp(-vz**2 / (2 * Tz)) / xp.sqrt(2 * xp.pi * Tz)

## 'Compute', 'Plot' and/or 'Save'
Kinetic = ['Compute', 'Plot']
Fluid = ['Compute', 'Plot']

darkmode = True
tqdm_display = False

###################################################################################################
##                               DO NOT EDIT BELOW                                               ##
###################################################################################################
dict = {'Tf': Tf,
		'alpha': alpha,
		'lam': lam,
        'integrator_kinetic': integrator_kinetic,
        'nsteps': nsteps,
		'integrator_fluid': integrator_kinetic,
        'precision': precision,
        'n_casimirs': n_casimirs,
        'Lz': Lz,
		'Lvx': Lvx,
		'Lvz': Lvz,
		'Nz': Nz,
		'Nvx': Nvx,
		'Nvz': Nvz,
		'f_init': f_init,
		'Kinetic': Kinetic,
		'Fluid': Fluid,
        'darkmode': darkmode,
		'tqdm_display': tqdm_display}
###################################################################################################
