"""
Dedalus script for 3D Rayleigh-Benard convection with MHD.

This script uses a Fourier basis in the x direction with periodic boundary
conditions.  The equations are scaled in units of the buoyancy time (Fr = 1).
MHD is implemented via a potential (vector A and scalar phi) approach,
while the hydro equations utilize a vorticity formulation for the diffusive terms.

Equations here are solved are identical to the 2.5D version but retain terms
with dy() in them.

This version of the script is intended for scaling and performance tests.

Usage:
    rayleigh_benard_mhd_A_2.5d_scaling.py [options]

Options:
    --nz=<nz>                 Number of Chebyshev modes [default: 128]
    --nx=<nx>                 Number of Fourier modes; default is aspect*nz
    --ny=<ny>                 Number of Fourier modes; default is aspect*nz
    --niter=<niter>           Iterations to run scaling test for (+1 automatically added to account for startup) [default: 10]

    --aspect=<aspect>         Aspect ratio [default: 2]
    --Rayleigh=<Rayleigh>     Rayleigh number of the convection [default: 1e6]

    --mesh=<mesh>             Processor mesh if distributing in 2-D

"""

import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

initial_time = time.time()

from docopt import docopt
args = docopt(__doc__)
nz = int(args['--nz'])
nx = args['--nx']
aspect = int(args['--aspect'])
if nx is None:
    nx = nz*aspect
else:
    nx = int(nx)
ny = args['--ny']
if ny is None:
    ny = nz*aspect
else:
    ny = int(ny)

Rayleigh_string = args['--Rayleigh']
Rayleigh = float(Rayleigh_string)

mesh = args['--mesh']
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]

# Parameters
Lx, Ly, Lz = (aspect, aspect, 1.)
Prandtl = 1.
MagneticPrandtl = 1.

# Create bases and domain
x_basis = de.Fourier(  'x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier(  'y', ny, interval=(0, Ly), dealias=3/2)
z_basis = de.Chebyshev('z', nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64, mesh=mesh)

# 3D Boussinesq magnetohydrodynamics with vector potential formulism
problem = de.IVP(domain, variables=['T','T_z','Ox','Oy','p','u','v','w','phi','Ax','Ay','Az','Bx','By'])
#problem.meta['p','T','u','v','w','Ay','Ax','Az','phi']['z']['dirichlet'] = True
problem.meta[:]['z']['dirichlet'] = True

problem.substitutions['UdotGrad(A,A_z)'] = '(u*dx(A) + v*dy(A) + w*(A_z))'
problem.substitutions['BdotGrad(A,A_z)'] = '(Bx*dx(A) + By*dy(A) + Bz*(A_z))'
problem.substitutions['Lap(A,A_z)'] = '(dx(dx(A)) + dy(dy(A)) + dz(A_z))'
problem.substitutions['Bz'] = '(dx(Ay) - dy(Ax))'
problem.substitutions['Jx'] = '(dy(Bz) - dz(By))'
problem.substitutions['Jy'] = '(dz(Bx) - dx(Bz))'
problem.substitutions['Jz'] = '(dx(By) - dy(Bx))'
problem.substitutions['Oz'] = '(dx(v)  - dy(u))'
problem.substitutions['Kx'] = '(dy(Oz) - dz(Oy))'
problem.substitutions['Ky'] = '(dz(Ox) - dx(Oz))'
problem.substitutions['Kz'] = '(dx(Oy) - dy(Ox))'

problem.parameters['P'] = (Rayleigh * Prandtl)**(-1/2)
problem.parameters['R'] = (Rayleigh / Prandtl)**(-1/2)
problem.parameters['Rm'] = (Rayleigh / MagneticPrandtl)**(-1/2)
problem.parameters['F'] = F = 1
problem.parameters['pi'] = np.pi
problem.add_equation("dt(T) - P*Lap(T, T_z)         - F*w = -UdotGrad(T, T_z)")
# O == omega = curl(u);  K = curl(O)
problem.add_equation("dt(u)  + R*Kx  + dx(p)              =  v*Oz - w*Oy + Jy*Bz - Jz*By")
problem.add_equation("dt(v)  + R*Ky  + dy(p)              =  w*Ox - u*Oz + Jz*Bx - Jx*Bz")
problem.add_equation("dt(w)  + R*Kz  + dz(p)    -T        =  u*Oy - v*Ox + Jx*By - Jy*Bx")
problem.add_equation("dt(Ax) + Rm*Jx + dx(phi)            =  v*Bz - w*By")
problem.add_equation("dt(Ay) + Rm*Jy + dy(phi)            =  w*Bx - u*Bz")
problem.add_equation("dt(Az) + Rm*Jz + dz(phi)            =  u*By - v*Bx")
problem.add_equation("dx(u) + dy(v) + dz(w) = 0")
problem.add_equation("dx(Ax) + dy(Ay) + dz(Az) = 0")
problem.add_equation("Bx + dz(Ay) - dy(Az) = 0")
problem.add_equation("By - dz(Ax) + dx(Az) = 0")
problem.add_equation("Ox + dz(v) - dy(w) = 0")
problem.add_equation("Oy - dz(u) + dx(w) = 0")
problem.add_equation("T_z - dz(T) = 0")
problem.add_bc("left(T) = 0")
problem.add_bc("left(u) = 0")
problem.add_bc("left(v) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(T) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("right(v) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0) or (ny != 0)")
problem.add_bc("right(p) = 0", condition="(nx == 0) and (ny == 0)")
problem.add_bc("left(Jx) = 0")
problem.add_bc("left(Jy) = 0")
problem.add_bc("left(Az) = 0")
problem.add_bc("right(Jx) = 0")
problem.add_bc("right(Jy) = 0")
problem.add_bc("right(Az) = 0", condition="(nx != 0) or (ny != 0)")
problem.add_bc("right(phi) = 0", condition="(nx == 0) and (ny == 0)")


# Build solver
#solver = problem.build_solver(de.timesteppers.RK443)
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

# Initial conditions
x = domain.grid(0)
y = domain.grid(1)
z = domain.grid(-1)
T = solver.state['T']
Bx = solver.state['Bx']
Ay = solver.state['Ay']

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]

# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
pert =  1e-3 * noise * (zt - z) * (z - zb)
T['g'] = F * pert
# if you set to scales(1), see obvious divU error early on; at 1/2 or 1/4, no divU error
T.set_scales(1/4, keep_data=True)
T['c']
T['g']
T.set_scales(1, keep_data=True)


B0 = 1e-6
Ay['g'] =  B0*np.sin(np.pi*z/Lz)
Bx['g'] = (-1*np.pi/Lz)*B0*np.cos(np.pi*z/Lz)

# Initial timestep
dt = 1e-3

# Integration parameters
startup_iter = 5
niter = int(float(args['--niter']))+startup_iter

solver.stop_sim_time = 50
solver.stop_wall_time = 30 * 60.
solver.stop_iteration = niter

max_dt = 0.5
# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=0.8/2,
                     max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.05)
CFL.add_velocities(('u', 'v', 'w'))
CFL.add_velocities(('Bx', 'By', 'Bz'))


# Main
try:
    logger.info('Starting loop')
    while solver.ok:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        log_string = 'Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt)
        logger.info(log_string)
        if solver.iteration == startup_iter:
            start_time = time.time()
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()

    if (domain.distributor.rank==0):
        N_TOTAL_CPU = domain.distributor.comm_cart.size

        # Print statistics
        print('-' * 40)
        total_time = end_time-initial_time
        main_loop_time = end_time - start_time
        startup_time = start_time-initial_time
        n_steps = solver.iteration-startup_iter
        print('  startup time:', startup_time)
        print('main loop time:', main_loop_time)
        print('    total time:', total_time)
        print('    iterations:', n_steps)
        print(' loop sec/iter:', main_loop_time/n_steps)
        print('    average dt:', solver.sim_time/n_steps)
        print("          N_cores, Nx, Nz, startup,    main loop,   main loop/iter, DOF-cycles/cpu-second")
        print('scaling:',
              ' {:d} {:d} {:d}'.format(N_TOTAL_CPU,nx,nz),
              ' {:8.3g} {:8.3g} {:8.3g} {:8.3g} {:8.3g}'.format(startup_time,
                                                                main_loop_time,
                                                                main_loop_time/n_steps,
                                                                nx*ny*nz*n_steps/(N_TOTAL_CPU*main_loop_time))
        print('-' * 40)
