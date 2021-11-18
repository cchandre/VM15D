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
from scipy.integrate import solve_ivp, simpson
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
	By = xp.zeros(case.Nz, dtype=xp.float64)
	f_ = xp.pad(f, ((0, 1),), mode='wrap')
	state_f = xp.hstack((case.compute_moments(f), Ex, By))
	rho = state_f[:case.Nz]
	Ez = case.Ez(rho)
	H0_k = case.energy_kinetic(f, Ex, Ez, By)
	C0_k = case.casimirs_kinetic(f, case.n_casimirs)
	H0_f = case.energy_fluid(state_f)
	C0_f = case.casimirs_fluid(state_f, case.n_casimirs)
	if 'Plot' in case.Kinetic:
		dict_kinetic = {'\\rho': rho,
						'E_z': Ez,
						'E_x': Ex,
						'B_y': By}
		axs_kinetic, line_kinetic = display_axes(case, dict_kinetic, simul='kinetic')
		fig_d = plt.figure(figsize=(7, 6.5))
		fig_d.canvas.manager.set_window_title(r'Distribution function f(z,vx,vz,t)')
		ax_d = plt.gca()
		ax_d.set_title('$\omega_p t = 0 $', loc='right', pad=-10)
		im = plt.imshow(simpson(f_, case.vx_, axis=1)[:-1, :-1].transpose(), interpolation='gaussian', origin='lower', aspect='auto', extent=(-case.Lz, case.Lz, -case.Lvz, case.Lvz), vmin=xp.min(f), vmax=xp.max(f))
		plt.gca().set_xlabel('$z$')
		plt.gca().set_ylabel('$v_z$')
		plt.colorbar()
	if 'Plot' in case.Fluid:
		dict_fluid = {'\\rho': rho,
						'E_z': Ez,
						'E_x': Ex,
						'B_y': By}
		axs_fluid, line_fluid = display_axes(case, dict_fluid, simul='fluid')
	TimeStep = 1 / case.nsteps
	t_eval = xp.linspace(1/case.nsteps, 1, case.nsteps)
	start = time.time()
	stop_kinetic = False
	stop_fluid = False
	for _ in trange(xp.int32(case.Tf), disable=not case.tqdm_display):
		if 'Compute' in case.Kinetic and not stop_kinetic:
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
			H = case.energy_kinetic(f, Ex, Ez, By)
			if xp.abs(H - H0_k) >= 1e-2:
				print('\033[33m        Warning: kinetic simulation stopped before the end \033[00m')
				print('\033[33m        Hf = {:.6e}    H0 = {:.6e}'.format(H, H0_k))
				stop_kinetic = True
			if 'Plot' in case.Kinetic:
				ax_d.set_title('$\omega_p t = {{{}}}$'.format(_ + 1), loc='right', pad=-10)
				im.set_data(simpson(f_, case.vx_, axis=1)[:-1, :-1].transpose())
				line_kinetic[0].set_ydata(simpson(simpson(f_, case.vz_, axis=2), case.vx_, axis=1)[:-1])
				line_kinetic[1].set_ydata(Ez)
				line_kinetic[2].set_ydata(Ex)
				line_kinetic[3].set_ydata(By)
				update_axes(case, axs_kinetic, _ + 1)
		if 'Compute' in case.Fluid and not stop_fluid:
			sol = solve_ivp(case.eqn_3f, (0, 1), state_f, t_eval=t_eval, method=case.integrator_fluid, atol=case.precision, rtol=case.precision)
			if sol.status!=0:
				print('\033[33m        Warning: fluid simulation stopped before the end \033[00m')
				stop_fluid = True
			else:
				state_f = sol.y[:, -1]
				rho, Px, Pz, S20, S11, S02, Ex, By = xp.split(state_f, 8)
				if xp.min(S20) <= case.precision or xp.min(S02) <= case.precision:
					print('\033[31m        Error: fluid simulation with S2<0 \033[00m')
					stop_fluid = True
				H = case.energy_fluid(state_f)
				if xp.abs(H - H0_f) >= 1e-2:
					print('\033[33m        Warning: fluid simulation stopped before the end \033[00m')
					print('\033[33m        Hf = {:.6e}    H0 = {:.6e}'.format(H, H0_f))
					stop_fluid = True
			if 'Plot' in case.Fluid:
				line_fluid[0].set_ydata(rho)
				line_fluid[1].set_ydata(case.Ez(rho))
				line_fluid[2].set_ydata(Ex)
				line_fluid[3].set_ydata(By)
				update_axes(case, axs_fluid, _ + 1)
	print('\033[90m        Computation finished in {} seconds \033[00m'.format(int(time.time() - start)))
	if 'Compute' in case.Kinetic:
		H = case.energy_kinetic(f, Ex, Ez, By)
		print('\033[90m        Error in energy (kinetic) = {:.2e}'.format(xp.abs(H - H0_k)))
		for indx, C in enumerate(case.casimirs_kinetic(f, case.n_casimirs)):
			print('\033[90m        Error in Casimir C{:d} (kinetic) = {:.2e}'.format(indx + 1, xp.abs(C - C0_k[indx])))
	if 'Compute' in case.Fluid:
		H = case.energy_fluid(state_f)
		print('\033[90m        Error in energy (fluid) = {:.2e}'.format(xp.abs(H - H0_f)))
		for indx, C in enumerate(case.casimirs_fluid(state_f, case.n_casimirs)):
			print('\033[90m        Error in Casimir C{:d} (fluid) = {:.2e}'.format(indx + 1, xp.abs(C - C0_f[indx])))
	plt.ioff()
	plt.show()

def display_axes(case, dict, simul=None):
	plt.ion()
	if case.darkmode:
		cs = ['k', 'w', 'c', 'm', 'r']
	else:
		cs = ['w', 'k', 'c', 'm', 'r']
	plt.rc('figure', facecolor=cs[0], titlesize=30)
	plt.rc('text', usetex=True, color=cs[1])
	plt.rc('font', family='sans-serif', size=20)
	plt.rc('axes', facecolor=cs[0], edgecolor=cs[1], labelsize=26, labelcolor=cs[1], titlecolor=cs[1])
	plt.rc('xtick', color=cs[1], labelcolor=cs[1])
	plt.rc('ytick', color=cs[1], labelcolor=cs[1])
	plt.rc('lines', linewidth=3)
	plt.rc('image', cmap='bwr')
	fig = plt.figure(figsize=(8, 8))
	fig.canvas.manager.set_window_title((simul + ' simulation').capitalize())
	axs = fig.add_gridspec(len(dict), hspace=0.2).subplots(sharex=True)
	line = []
	for m, (key, value) in enumerate(dict.items()):
		axs[m].plot(case.z, value, cs[m+1], linestyle='--', linewidth=1, label=r'$' + str(key) + '(z,0)$')
		line_temp, = axs[m].plot(case.z, value, cs[m+1], label=r'$' + str(key) + '(z,t)$')
		line.append(line_temp)
	axs[0].set_title('$\omega_p t = 0 $', loc='right', pad=20)
	for ax in axs:
		ax.set_xlim((-case.Lz, case.Lz))
		ax.legend(loc='upper right', labelcolor='linecolor')
	axs[-1].set_xlabel('$z$')
	plt.draw()
	plt.pause(1e-4)
	return axs, line

def update_axes(case, axs, t):
	axs[0].set_title('$\omega_p t = {{{}}}$'.format(t), loc='right', pad=20)
	for ax in axs:
		ax.relim()
		ax.autoscale()
		ax.set_xlim((-case.Lz, case.Lz))
	plt.draw()
	plt.pause(1e-4)

def save_data(state, data, timestr, case, model=[]):
	mdic = case.DictParams.copy()
	mdic.update({'final': state, 'data': data})
	date_today = date.today().strftime(" %B %d, %Y")
	mdic.update({'date': date_today, 'author': 'cristel.chandre@cnrs.fr'})
	name_file = type(case).__name__ + '_' + model + '_' + timestr + '.mat'
	savemat(name_file, mdic)
	print('\033[90m        {} results saved in {} \033[00m'.format(model, name_file))
