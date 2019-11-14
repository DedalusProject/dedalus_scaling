"""
Dedalus script for solution of 3D Navier-Stokes spin down.

This script uses a Fourier basis in the x and z directions with periodic boundary
conditions.

Usage:
    incompressible_NS_TG.py [options]

Options:
    --mesh=<mesh>             Processor mesh if distributing in 2-D
    --nz=<nz>                 Chebyshev resolution [default: 128]
    --nx=<nx>                 Fourier resolution [default: 128]
    --ny=<ny>                 Fourier resolution [default: 128]
    --niter=<niter>           Iterations to run scaling test for (+1 automatically added to account for startup) [default: 100]
    --IO                      Do analysis IO

"""
from docopt import docopt
args = docopt(__doc__)

import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

mesh = args['--mesh']
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]

initial_time = time.time()
# Parameters
Lx, Ly, Lz = (1., 1., 1.)
nx, ny, nz = (int(args['--nx']),int(args['--ny']),int(args['--nz'])) # grid resolution is 3/2 higher

Reynolds = 1
v0, k0 = (1,4) # Amplitude and wavenumber of the initial conditions

# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', ny, interval=(0, Ly), dealias=3/2)
z_basis = de.Fourier('z', nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64, mesh=mesh)

# Implementation of Navier-Stokes equation system
problem = de.IVP(domain, variables=['p','u','v','w'])
problem.parameters['nu'] = 1 / Reynolds

problem.substitutions["L(A)"] = 'nu*(d(A,x=2) + d(A,y=2) + d(A,z=2))'
problem.substitutions["N(A)"] = '-(u*dx(A) + v*dy(A) + w*dz(A))'
problem.substitutions['enstrophy'] = '(dy(w)-dz(v))**2 + (dz(u)-dx(w))**2 + (dx(v)-dy(u))**2'

problem.add_equation("u=0", condition="(nx==0) and (ny==0) and (nz==0)")
problem.add_equation("v=0", condition="(nx==0) and (ny==0) and (nz==0)")
problem.add_equation("w=0", condition="(nx==0) and (ny==0) and (nz==0)")
problem.add_equation("p=0", condition="(nx==0) and (ny==0) and (nz==0)")

problem.add_equation("dt(u) - L(u) + dx(p)  = N(u) ", condition="(nx!=0) or  (ny!=0) or  (nz!=0)")
problem.add_equation("dt(v) - L(v) + dy(p)  = N(v) ", condition="(nx!=0) or  (ny!=0) or  (nz!=0)")
problem.add_equation("dt(w) - L(w) + dz(p)  = N(w) ", condition="(nx!=0) or  (ny!=0) or  (nz!=0)")
problem.add_equation("dx(u) + dy(v) + dz(w) = 0",     condition="(nx!=0) or  (ny!=0) or  (nz!=0)")


# Build solver
#solver = problem.build_solver(de.timesteppers.RK222)
solver = problem.build_solver(de.timesteppers.SBDF2)
logger.info('Solver built')

# Initial conditions : Taylor-Green forcing

u = solver.state['u']
v = solver.state['v']
w = solver.state['w']


x = domain.grid(0)
y = domain.grid(1)
z = domain.grid(2)

u['g'] = v0*np.sin(k0*x)*np.cos(k0*y)*np.cos(k0*z)
v['g'] = -v0*np.cos(k0*x)*np.sin(k0*y)*np.cos(k0*z)
w['g'] = 0

# Initial timestep
dt = 0.125

# Integration parameters
startup_iter = 5
niter = int(float(args['--niter']))+startup_iter

solver.stop_sim_time = 25
solver.stop_wall_time = 30 * 60.
solver.stop_iteration = niter

if args['--IO']:
    # Analysis
    snapshots = solver.evaluator.add_file_handler('snapshots', max_writes=50, sim_dt=0.125)
    snapshots.add_system(solver.state)
    snapshots.add_task('enstrophy',name='enstrophy')
    snapshots.add_task('integ(enstrophy)',name='enstrophy_tot')
    snapshots.add_task('integ(u**2+v**2+w**2)',name='KE')

# CFL
#CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=1,
#                     max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
#CFL.add_velocities(('u', 'v', 'w'))

# Main loop
try:
    logger.info('Starting loop')
    while solver.ok:
        #dt = CFL.compute_dt()
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
              ' {:12.7g} {:12.7g} {:12.7g} {:12.7g}'.format(startup_time,
                                                                main_loop_time,
                                                                main_loop_time/n_steps,
                                                                nx*ny*nz*n_steps/(N_TOTAL_CPU*main_loop_time)))
        print('-' * 40)
