# VM15D : Hamiltonian fluid reduction of the 1.5D Vlasov-Maxwell equations

- **VM15D.nb**: [Mathematica notebook] checks the Jacobi identity and the Casimir invariants for the bracket Bracket[F,G] given by Eq. (6) of the article  *Hamiltonian fluid reduction of the 1.5D Vlasov-Maxwell equations* by C. Chandre and B.A. Shadwick


- **GrowthRates.mlx**: [Matlab livescript] computes the growth rates for the linearized equations of motion near a homogeneous equilibrium. It reproduces the figures of the article *Hamiltonian fluid reduction of the 1.5D Vlasov-Maxwell equations* by C. Chandre and B.A. Shadwick

```bibtex
@article{chandre2021,
         title = {Hamiltonian fluid reduction of the 1.5D Vlasov–Maxwell equations},
         author = {Chandre, C.  and Shadwick, B.A. },
         journal = {Physics of Plasmas},
         volume = {28},
         number = {9},
         pages = {092114},
         year = {2021},
         doi = {10.1063/5.0056155},
         URL = {https://doi.org/10.1063/5.0056155}
}
```

- `VM15D python code`
  - [`VM15D_dict.py`](https://github.com/cchandre/VM15D/blob/main/VM15D_dict.py): to be edited to change the parameters of the VM15D computation (see below for a dictionary of parameters)

  - [`VM15D.py`](https://github.com/cchandre/VM15D/blob/main/VM15D.py): contains the VM15D class and main functions (not to be edited)

  - [`VM15D_modules.py`](https://github.com/cchandre/VM15D/blob/main/VM15D_modules.py): contains the methods to run VM15D (not to be edited)

  - Once [`VM15D_dict.py`](https://github.com/cchandre/VM15D/blob/main/VM15D_dict.py) has been edited with the relevant parameters, run the file as 
  ```sh
  python3 VM15D.py
  ```
  - NB: in case of error, check your version of the python modules used in the code (see [`modules_version.txt`](https://github.com/cchandre/VM15DD/blob/main/modules_version.txt))

___
##  Parameter dictionary for VM15D

- *Tf*: double; duration of the integration (in units of *&omega;<sub>p</sub><sup>-1</sup>*)
- *integrator_kinetic*: string ('position-Verlet'); choice of solver for the integration of the Vlasov equation
- *nsteps*: integer; number of steps in one period of plasma oscillations (1/*&omega;<sub>p</sub><sup>*) for the integration of the Vlasov equation
- *precision*: double; threshold for the Fourier transforms
- *n_casimirs*: integer; number of Casimir invariants to be monitored 

- *Lz*: double; the *z*-axis is (-*Lz*, *Lz*)
- *Lvx*: double; the *vx*-axis is (-*Lvx*, *Lvx*)
- *Lvz*: double; the *vz*-axis is (-*Lvz*, *Lvz*)
- *Nz*: integer; number of points in *z* to represent the field variables
- *Nvx*: integer; number of points in *vx* to represent the field variables
- *Nvz*: integer; number of points in *vz* to represent the field variables
- *f_init*: lambda function; initial distribution *f*(*z*,*vx*,*vz*,*t*=0)

- *Kinetic*: list of strings in ['Compute', 'Plot']; list of instructions for the 1.5D Vlasov-Maxwell simulation

- *darkmode*: boolean; if True, plots are done in dark mode
         
         
For more information: <cristel.chandre@cnrs.fr>
