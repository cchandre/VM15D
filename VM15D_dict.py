###################################################################################################
##              Dictionary of parameters: https://github.com/cchandre/VM15D                      ##
###################################################################################################

import numpy as xp

Tf = 80

integrator_kinetic = 'position-Verlet'
nsteps = 16
precision = 1e-11

n_casimirs = 3

Lz = 2 * xp.pi
Lvx = 6
Lvz = 6
Nz = 2**7
Nvx = 2**7
Nvz = 2**7

A = 1e-6
k = 0.5
f_init = lambda z, vx, vz: (1 - A * xp.cos(k * z)) * (vx**2 + vz**2) * xp.exp(-(vx**2 + vz**2) / 2) / xp.sqrt(2 * xp.pi)

## 'Compute', 'Plot' and/or 'Save'
Kinetic = ['Compute', 'Plot']

darkmode = True

###################################################################################################
##                               DO NOT EDIT BELOW                                               ##
###################################################################################################
dict = {'Tf': Tf}
dict.update({
        'integrator_kinetic': integrator_kinetic,
        'nsteps': nsteps,
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
        'darkmode': darkmode})
###################################################################################################
