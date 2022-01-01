"""
Dedalus script for 3D Rayleigh-Benard convection with MHD.
This version of the script is intended for scaling and performance tests.

Usage:
    RB_mhd_3d.5d_scaling.py [options]

Options:
    --aspect=<aspect>         Aspect ratio [default: 2]
    --nz=<nz>                 Number of Chebyshev modes [default: 128]
    --nx=<nx>                 Number of Fourier modes; default is aspect*nz
    --ny=<ny>                 Number of Fourier modes; default is aspect*nz
    --Rayleigh=<Rayleigh>     Rayleigh number of the convection [default: 1e6]
    --mesh=<mesh>             Processor mesh if distributing in 2-D
    --niter=<niter>           Timing iterations [default: 100]
    --nstart=<nstart>         Startup iterations [default: 10]

"""

import numpy as np
from mpi4py import MPI
import time
from docopt import docopt
from dedalus import public as de
import logging
logger = logging.getLogger(__name__)

# Process arguments
args = docopt(__doc__)
aspect = int(args['--aspect'])
nz = int(args['--nz'])
nx = args['--nx']
if nx is None:
    nx = nz*aspect
else:
    nx = int(nx)
ny = args['--ny']
if ny is None:
    ny = nz*aspect
else:
    ny = int(ny)
Rayleigh = float(args['--Rayleigh'])
mesh = args['--mesh']
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]
niter = int(args['--niter'])
nstart = int(args['--nstart'])

# Fixed parameters
Prandtl = 1.
MagneticPrandtl = 1.
ts = de.timesteppers.SBDF2
dt = 1e-3
timeout = 15 * 60

# Derived parameters
Lx, Ly, Lz = (aspect, aspect, 1.)

# Create bases and domain
initial_time = time.time()
x_basis = de.Fourier(  'x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier(  'y', ny, interval=(0, Ly), dealias=3/2)
z_basis = de.Chebyshev('z', nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64, mesh=mesh)

# 3D Boussinesq magnetohydrodynamics with vector potential formulism
problem = de.IVP(domain, variables=['T','T_z','Ox','Oy','p','u','v','w','phi','Ax','Ay','Az','Bx','By', 'Jx', 'Jy'])
problem.substitutions['UdotGrad(A,A_z)'] = '(u*dx(A) + v*dy(A) + w*(A_z))'
problem.substitutions['BdotGrad(A,A_z)'] = '(Bx*dx(A) + By*dy(A) + Bz*(A_z))'
problem.substitutions['Lap(A,A_z)'] = '(dx(dx(A)) + dy(dy(A)) + dz(A_z))'
problem.substitutions['Bz'] = '(dx(Ay) - dy(Ax))'
problem.substitutions['Jz'] = '(dx(By) - dy(Bx))'
problem.substitutions['Oz'] = '(dx(v)  - dy(u))'
problem.substitutions['Kx'] = '(dy(Oz) - dz(Oy))'
problem.substitutions['Ky'] = '(dz(Ox) - dx(Oz))'
problem.substitutions['Kz'] = '(dx(Oy) - dy(Ox))'
problem.parameters['P'] = (Rayleigh * Prandtl)**(-1/2)
problem.parameters['R'] = (Rayleigh / Prandtl)**(-1/2)
problem.parameters['Rm'] = (Rayleigh / MagneticPrandtl)**(-1/2)
problem.parameters['pi'] = np.pi
problem.add_equation("dt(T) - P*Lap(T, T_z)         - w = -UdotGrad(T, T_z)")
# O == omega = curl(u);  K = curl(O)
problem.add_equation("dt(u)  +  R*Kx + dx(p)              =  v*Oz - w*Oy + Jy*Bz - Jz*By")
problem.add_equation("dt(v)  +  R*Ky + dy(p)              =  w*Ox - u*Oz + Jz*Bx - Jx*Bz")
problem.add_equation("dt(w)  +  R*Kz + dz(p)    -T        =  u*Oy - v*Ox + Jx*By - Jy*Bx")
problem.add_equation("dt(Ax) + Rm*Jx + dx(phi)            =  v*Bz - w*By")
problem.add_equation("dt(Ay) + Rm*Jy + dy(phi)            =  w*Bx - u*Bz")
problem.add_equation("dt(Az) + Rm*Jz + dz(phi)            =  u*By - v*Bx")
problem.add_equation("dx(u) + dy(v) + dz(w) = 0")
problem.add_equation("dx(Ax) + dy(Ay) + dz(Az) = 0")
problem.add_equation("Bx + dz(Ay) - dy(Az) = 0")
problem.add_equation("By - dz(Ax) + dx(Az) = 0")
problem.add_equation("Jx - (dy(Bz) - dz(By)) = 0")
problem.add_equation("Jy - (dz(Bx) - dx(Bz)) = 0")
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
solver = problem.build_solver(ts)
logger.info('Solver built')

# Initial conditions
x, y, z = domain.grids()
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
T['g'] =  1e-3 * noise * (zt - z) * (z - zb)
# if you set to scales(1), see obvious divU error early on; at 1/2 or 1/4, no divU error
T.set_scales(1/4, keep_data=True)
T['g']
T.set_scales(1, keep_data=True)

# Small and smooth magnetic field
B0 = 1e-6
Ay['g'] =  B0*np.sin(np.pi*z/Lz)
Bx['g'] = (-1*np.pi/Lz)*B0*np.cos(np.pi*z/Lz)

# Integration parameters
solver.stop_wall_time = timeout
solver.stop_iteration = nstart + niter

# Main
try:
    logger.info('Starting loop')
    while solver.ok:
        solver.step(dt)
        log_string = 'Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt)
        logger.info(log_string)
        if solver.iteration == nstart:
            start_time = time.time()
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()

    # Print statistics
    if domain.distributor.rank == 0:
        print('-' * 40)
        n_proc = domain.distributor.comm_cart.size
        total_time = end_time - initial_time
        main_loop_time = end_time - start_time
        startup_time = start_time - initial_time
        n_steps = solver.iteration - nstart
        print('  startup time:', startup_time)
        print('main loop time:', main_loop_time)
        print('    total time:', total_time)
        print('    iterations:', n_steps)
        print(' loop sec/iter:', main_loop_time/n_steps)
        print('    average dt:', solver.sim_time/solver.iteration)
        print("          N_cores, Nx, Nz, startup,    main loop,   main loop/iter, DOF-cycles/cpu-second")
        print('scaling:',
              ' {:d} {:d} {:d}'.format(n_proc, nx, nz),
              ' {:12.7g} {:12.7g} {:12.7g} {:12.7g}'.format(startup_time,
                                                            main_loop_time,
                                                            main_loop_time/n_steps,
                                                            nx*ny*nz*n_steps/(n_proc*main_loop_time)))
        print('-' * 40)
