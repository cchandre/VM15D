#
# BSD 2-Clause License
#
# Copyright (c) 2021, Cristel Chandre
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as xp
from scipy.integrate import simpson
from tqdm import trange
from scipy.io import savemat
import time
from datetime import date
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def integrate(case):
	timestr = time.strftime("%Y%m%d_%H%M")
	f = case.f.copy()
	Ex = xp.zeros(case.Nz, dtype=xp.float64)
	f_ = xp.pad(f, ((0, 1),), mode='wrap')
	Ez = case.Ez(simpson(simpson(f_, case.vz_, axis=2), case.vx_, axis=1)[:-1])
	By = xp.zeros(case.Nz, dtype=xp.float64)
	H0 = case.energy_kinetic(f, Ex, Ez, By)
	C0 = case.casimirs_kinetic(f, case.n_casimirs)
	plt.ion()
	if case.darkmode:
		cs = ['k', 'w']
	else:
		cs = ['w', 'k']
	plt.rc('figure', facecolor=cs[0], titlesize=30)
	plt.rc('text', usetex=True, color=cs[1])
	plt.rc('font', family='sans-serif', size=20)
	plt.rc('axes', facecolor=cs[0], edgecolor=cs[1], labelsize=26, labelcolor=cs[1], titlecolor=cs[1])
	plt.rc('xtick', color=cs[1], labelcolor=cs[1])
	plt.rc('ytick', color=cs[1], labelcolor=cs[1])
	plt.rc('lines', linewidth=3)
	plt.rc('image', cmap='bwr')
	if 'Plot' in case.Kinetic:
		fig_d = plt.figure(figsize=(7, 6.5))
		fig_d.canvas.manager.set_window_title(r'Distribution function f(z,vx,vz,t)')
		ax_d = plt.gca()
		ax_d.set_title('$\omega_p t = 0 $', loc='right', pad=-10)
		im = plt.imshow(f[:, case.Nvx//2, :].transpose(), interpolation='gaussian', origin='lower', aspect='auto', extent=(-case.Lz, case.Lz, -case.Lvz, case.Lvz), vmin=xp.min(f), vmax=xp.max(f))
		plt.gca().set_ylabel('$v_z$')
		plt.gca().set_xlabel('$z$')
		plt.colorbar()
		fig_f = plt.figure(figsize=(8, 10))
		fig_f.canvas.manager.set_window_title('1.5D Vlasov-Maxwell simulation')
		axs_f = fig_f.add_gridspec(3, hspace=0.2).subplots(sharex=True)
		axs_f[0].set_title('$\omega_p t = 0 $', loc='right', pad=20)
		axs_f[0].plot(case.z, Ez, 'r--', linewidth=1, label=r'$E_z(0)$')
		line_Ez, = axs_f[0].plot(case.z, Ez, 'r', label=r'$E_z(t)$')
		axs_f[1].plot(case.z, Ex, 'c--', linewidth=1, label=r'$E_x(0)$')
		line_Ex, = axs_f[1].plot(case.z, Ex, 'c', label=r'$E_x(t)$')
		axs_f[2].plot(case.z, By, cs[1], linestyle='--', linewidth=1, label=r'$B_y(0)$')
		line_By, = axs_f[2].plot(case.z, By, cs[1], label=r'$B_y(t)$')
		for ax in axs_f:
			ax.set_xlim((-case.Lz, case.Lz))
			ax.legend(loc='upper right', labelcolor='linecolor')
		axs_f[-1].set_xlabel('$z$')
		plt.draw()
		plt.pause(1e-4)
	TimeStep = 1 / case.nsteps
	t_eval = xp.linspace(1/case.nsteps, 1, case.nsteps)
	start = time.time()
	for _ in trange(xp.int32(case.Tf)):
		if 'Compute' in case.Kinetic:
			for t in range(case.nsteps):
				for coeff, type in zip(case.integr5_coeff, case.integr5_type):
					if type == 1:
						f, Ex, Ez, By = case.Hpx(f, Ex, Ez, By, coeff * TimeStep)
					elif type == 2:
						f, Ex, Ez, By = case.Hpz(f, Ex, Ez, By, coeff * TimeStep)
					elif type == 3:
						f, Ex, Ez, By = case.Hcx(f, Ex, Ez, By, coeff * TimeStep)
					elif type==4:
						f, Ex, Ez, By = case.Hcz(f, Ex, Ez, By, coeff * TimeStep)
					elif type==5:
						f, Ex, Ez, By = case.Hcy(f, Ex, Ez, By, coeff * TimeStep)
				f[f<=case.precision] = 0
				f_ = xp.pad(f, ((0, 1),), mode='wrap')
				f_ *= case.f0 / simpson(simpson(simpson(f_, case.vz_, axis=2), case.vx_, axis=1), case.z_)
				f = f_[:-1, :-1, :-1]
			if 'Plot' in case.Kinetic:
				ax_d.set_title('$\omega_p t = {{{}}}$'.format(_ + 1), loc='right', pad=-10)
				axs_f[0].set_title('$\omega_p t = {{{}}}$'.format(_ + 1), loc='right', pad=20)
				im.set_data(f[:, case.Nvx//2, :].transpose())
				line_Ex.set_ydata(Ex)
				line_Ez.set_ydata(Ez)
				line_By.set_ydata(By)
				for ax in axs_f:
					ax.relim()
					ax.autoscale()
					ax.set_xlim((-case.Lz, case.Lz))
				plt.draw()
				plt.pause(1e-4)
	print('\033[90m        Computation finished in {} seconds \033[00m'.format(int(time.time() - start)))
	if 'Compute' in case.Kinetic:
		H = case.energy_kinetic(f, Ex, Ez, By)
		print('\033[90m        Error in energy = {:.2e}'.format(xp.abs(H - H0)))
		for indx, C in enumerate(case.casimirs_kinetic(f, case.n_casimirs)):
			print('\033[90m        Error in Casimir C{:d} = {:.2e}'.format(indx + 1, xp.abs(C - C0[indx])))
	plt.ioff()
	plt.show()

def save_data(state, data, timestr, case, model=[]):
	mdic = case.DictParams.copy()
	mdic.update({'final': state, 'data': data})
	date_today = date.today().strftime(" %B %d, %Y")
	mdic.update({'date': date_today, 'author': 'cristel.chandre@cnrs.fr'})
	name_file = type(case).__name__ + '_' + model + '_' + timestr + '.mat'
	savemat(name_file, mdic)
	print('\033[90m        {} results saved in {} \033[00m'.format(model, name_file))
