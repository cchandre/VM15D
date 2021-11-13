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
from scipy.fft import rfft, irfft, rfftfreq
from scipy.integrate import simpson
from VM15D_modules import integrate
from VM15D_dict import dict

def main():
	integrate(VM15D(dict))

class VM15D:
	def __repr__(self):
		return '{self.__class__.__name__}({self.DictParams})'.format(self=self)

	def __str__(self):
		return '1.5D Vlasov-Maxwell equation ({self.__class__.__name__})'.format(self=self)

	def __init__(self, dict):
		for key in dict:
			setattr(self, key, dict[key])
		self.DictParams = dict
		self.z = xp.linspace(-self.Lz, self.Lz, self.Nz, endpoint=False, dtype=xp.float64)
		self.vx = xp.linspace(-self.Lvx, self.Lvx, self.Nvx, endpoint=False, dtype=xp.float64)
		self.vz = xp.linspace(-self.Lvz, self.Lvz, self.Nvz, endpoint=False, dtype=xp.float64)
		self.z_ = xp.linspace(-self.Lz, self.Lz, self.Nz+1, dtype=xp.float64)
		self.vx_ = xp.linspace(-self.Lvx, self.Lvx, self.Nvx+1, dtype=xp.float64)
		self.vz_ = xp.linspace(-self.Lvz, self.Lvz, self.Nvz+1, dtype=xp.float64)
		self.kz = xp.pi / self.Lz * rfftfreq(self.Nz, d=1/self.Nz)
		div = xp.divide(1, 1j * self.kz, where=self.kz!=0)
		div[0] = 0
		self.kvx = xp.pi / self.Lvx * rfftfreq(self.Nvx, d=1/self.Nvx)
		self.kvz = xp.pi / self.Lvz * rfftfreq(self.Nvz, d=1/self.Nvz)
		self.tail_indx = [(xp.s_[3*self.Nz//8:], xp.s_[:], xp.s_[:],), (xp.s_[:], xp.s_[3*self.Nvx//8:], xp.s_[:],), (xp.s_[:], xp.s_[:], xp.s_[3*self.Nvz//8:])]
		f_ = self.f_init(self.z_[:, None, None], self.vx_[None, :, None], self.vz_[None, None, :])
		self.f = f_[:-1, :-1, :-1]
		self.f0 = simpson(simpson(simpson(f_, self.vz_, axis=2), self.vx_, axis=1), self.z_)
		self.Ez = lambda rho: irfft(div * self.rfft_(rho))
		if self.integrator_kinetic == 'position-Verlet':
			self.integr2_coeff = [0.5, 1, 0.5]
			self.integr2_type = [1, 2, 1]
			self.integr5_coeff = [0.25, 0.5, 0.25, 0.25, 0.5, 0.25, 1, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25]
			self.integr5_type = [1, 2, 1, 3, 4, 3, 5, 3, 4, 3, 1, 2, 1]

	def Hpx(self, f, Ex, Ez, By, dt):
		ft = irfft(xp.exp(-1j * self.kvz[None, None, :] * self.vx[None, :, None] * By[:, None, None] * dt) * self.rfft_(f, axis=2), axis=2)
		f_ = xp.pad(f, ((0, 1),), mode='wrap')
		Etx = Ex - simpson(simpson(self.vx_[None, :, None] * f_, self.vz_, axis=2), self.vx_, axis=1)[:-1] * dt
		return ft, Etx, Ez, By

	def Hpz(self, f, Ex, Ez, By, dt):
		ft = f.copy()
		for coeff, type in zip(self.integr2_coeff, self.integr2_type):
			if type == 1:
				ft = irfft(xp.exp(-1j * self.vz[None, None, :] * self.kz[:, None, None] * coeff * dt) * self.rfft_(ft, axis=0), axis=0)
			elif type == 2:
				ft = irfft(xp.exp(1j * self.vz[None, None, :] * By[:, None, None] * self.kvx[None, :, None] * coeff * dt) * self.rfft_(ft, axis=1), axis=1)
		Etz = self.Ez(simpson(simpson(xp.pad(ft, ((0, 1),), mode='wrap'), self.vz_, axis=2), self.vx_, axis=1)[:-1])
		return ft, Ex, Etz, By

	def Hcz(self, f, Ex, Ez, By, dt):
		ft = irfft(xp.exp(-1j * Ez[:, None, None] * self.kvz[None, None, :] * dt) * self.rfft_(f, axis=2), axis=2)
		return ft, Ex, Ez, By

	def Hcx(self, f, Ex, Ez, By, dt):
		ft = irfft(xp.exp(-1j * Ex[:, None, None] * self.kvx[None, :, None] * dt) * self.rfft_(f, axis=1), axis=1)
		Bty = By - irfft(1j * self.kz * self.rfft_(Ex)) * dt
		return ft, Ex, Ez, Bty

	def Hcy(self, f, Ex, Ez, By, dt):
		Etx = Ex - irfft(1j * self.kz * self.rfft_(By)) * dt
		return f, Etx, Ez, By

	def rfft_(self, f, axis=0):
		fft_f = rfft(f, axis=axis)
		fft_f[xp.abs(fft_f) <= self.precision] = 0
		fft_f[self.tail_indx[axis][:f.ndim]] = 0
		return fft_f

	def energy_kinetic(self, f, Ex, Ez, By):
		f_ = xp.pad(f, ((0, 1),), mode='wrap')
		Ex_ = xp.pad(Ex, (0, 1), mode='wrap')
		Ez_ = xp.pad(Ez, (0, 1), mode='wrap')
		By_ = xp.pad(By, (0, 1), mode='wrap')
		return (simpson(simpson(simpson((self.vx_[None, :, None]**2 + self.vz_[None, None, :]**2) * f_, self.vz_, axis=2), self.vx_, axis=1), self.z_) + simpson(Ex_**2 + Ez_**2 + By_**2, self.z_)) / 2

	def casimirs_kinetic(self, f, n):
		f_ = xp.pad(f, ((0, 1),), mode='wrap')
		return [simpson(simpson(simpson(f_**m, self.vz_, axis=2), self.vx_, axis=1), self.z_) for m in range(1, n+1)]

if __name__ == "__main__":
	main()
