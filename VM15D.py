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
		self.div = xp.divide(1, 1j * self.kz, where=self.kz!=0)
		self.div[0] = 0
		self.kvx = xp.pi / self.Lvx * rfftfreq(self.Nvx, d=1/self.Nvx)
		self.kvz = xp.pi / self.Lvz * rfftfreq(self.Nvz, d=1/self.Nvz)
		self.tail_indx = [(xp.s_[3*self.Nz//8:], xp.s_[:], xp.s_[:],), (xp.s_[:], xp.s_[3*self.Nvx//8:], xp.s_[:],), (xp.s_[:], xp.s_[:], xp.s_[3*self.Nvz//8:])]
		f_ = self.f_init(self.z_[:, None, None], self.vx_[None, :, None], self.vz_[None, None, :])
		self.f = f_[:-1, :-1, :-1]
		self.f0 = simpson(simpson(simpson(f_, self.vz_, axis=2), self.vx_, axis=1), self.z_)
		self.Ez = lambda rho: irfft(self.div * self.rfft_(rho))
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

	def closure(self, f, By):
		rho, Px, Pz, S20, S11, S02 = xp.split(f, 6)
		Pix = Px + irfft(self.div * self.rfft_(By))
		S21 = S11 * (self.alpha - Pix) - S20 * S11 / (self.alpha - Pix)
		S12 = S02 * (self.alpha - Pix) - S11**2 / (self.alpha - Pix)
		S03 = S11 / S20 * (3 * S02 - 2 * S11**2 / S20) * (self.alpha - Pix) - S11**3 / S20 / (self.alpha - Pix) + self.lam * (S02 - S11**2 / S20)**(4/3)
		return S21, S12, S03

	def eqn_3f(self, t, f, Ex, By):
		rho, Px, Pz, S20, S11, S02 = xp.split(f, 6)
		S21, S12, S03 = self.closure(f, By)
		Ez = self.Ez(rho)
		rho_dot = - irfft(1j * self.kz * self.rfft_(rho * Pz))
		Px_dot = - Pz * irfft(1j * self.kz * self.rfft_(Px)) + Ex - Pz * By - irfft(1j * self.kz * self.rfft_(rho**2 * S11)) / rho
		Pz_dot = - Pz * irfft(1j * self.kz * self.rfft_(Pz)) + Ez + Px * By - irfft(1j * self.kz * self.rfft_(rho**3 * S02)) / rho
		S20_dot = - Pz * irfft(1j * self.kz * self.rfft_(S20)) - 2 * rho * S11 * (By + irfft(1j * self.kz * self.rfft_(Px))) - irfft(1j * self.kz * self.rfft_(rho**2 * S21)) / rho
		S11_dot = - Pz * irfft(1j * self.kz * self.rfft_(S11)) + By * S20 / rho - rho * S02 * (By + irfft(1j * self.kz * self.rfft_(Px))) - irfft(1j * self.kz * self.rfft_(rho**3 * S12)) / rho**2
		S02_dot = - Pz * irfft(1j * self.kz * self.rfft_(S02)) + 2 * By * S11 / rho - irfft(1j * self.kz * self.rfft_(rho**4 * S03)) / rho**3
		Ex_dot = -irfft(1j * self.kz * self.rfft_(By))) - rho * Px
		By_dot = -irfft(1j * self.kz * self.rfft_(Ex)))

	def compute_moments(self, f, n):
		f_ = xp.pad(f, ((0, 1),), mode='wrap')
		rho = simpson(simpson(f_, self.vz_, axis=2), self.vx_, axis=1)
		Px = simpson(simpson(self.vx_[None, :, None] * f_, self.vz_, axis=2), self.vx_, axis=1) / rho
		Pz = simpson(simpson(self.vz_[None, None, :] * f_, self.vz_, axis=2), self.vx_, axis=1) / rho
		S20 = simpson(simpson((self.vx_ - Px)[None, :, None]**2 * f_, self.vz_, axis=2), self.vx_, axis=1) / rho
		S11 = simpson(simpson((self.vx_ - Px)[None, :, None] * (self.vz_ - Pz)[None, None, :] * f_, self.vz_, axis=2), self.vx_, axis=1) / rho**2
		S02 = simpson(simpson((self.vz_ - Pz)[None, None, :]**2 * f_, self.vz_, axis=2), self.vx_, axis=1) / rho**3
		return rho[:-1], Px[:-1], Pz[:-1], S20[:-1], S11[:-1], S02[:-1]

	def rfft_(self, f, axis=0):
		fft_f = rfft(f, axis=axis)
		fft_f[xp.abs(fft_f) <= self.precision] = 0
		fft_f[self.tail_indx[axis][:f.ndim]] = 0
		return fft_f

	def energy_fluid(self, f, Ex, Ez, By):
		rho, Px, Pz, S20, S11, S02 = [xp.pad(_, (0, 1), mode='wrap') for _ in xp.split(f, 6)]
		Ex_ = xp.pad(Ex, (0, 1), mode='wrap')
		Ez_ = xp.pad(Ez, (0, 1), mode='wrap')
		By_ = xp.pad(By, (0, 1), mode='wrap')
		return simpson(rho * (Px**2 + Pz**2) + rho * S20 + rho**3 * S02 + Ex_**2 + Ez_**2 + By_**2, self.z_) / 2

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
