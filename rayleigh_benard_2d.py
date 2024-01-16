"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.

Usage:
    rayleigh_benard_2d.py [options]

Options:
    --nx=<nx>              Horizontal modes; default is aspect x Nz
    --nz=<nz>              Vertical modes [default: 64]
    --aspect=<aspect>      Aspect ratio of domain [default: 4]

    --Rayleigh=<Rayleigh>  Rayleigh number [default: 1e6]

    --niter=<iter>         How many iterations to run for [default: 100]
    --nstart=<nstart>      Startup iterations [default: 10]

    --label=<label>        Additional label for run output directory
"""
from mpi4py import MPI
import numpy as np
import time
import sys
import os

from docopt import docopt
args = docopt(__doc__)

aspect = float(args['--aspect'])
# Parameters
Lx, Lz = aspect, 1
nz = int(args['--nz'])
if args['--nx']:
    nx = int(args['--nx'])
else:
    nx = int(aspect*nz)

import dedalus.public as d3

import logging
logger = logging.getLogger(__name__)

# TODO: maybe fix plotting to directly handle vectors
# TODO: optimize and match d2 resolution
# TODO: get unit vectors from coords?

comm = MPI.COMM_WORLD
rank = comm.rank
ncpu = comm.size

Rayleigh = float(args['--Rayleigh'])
Prandtl = 1
dealias = 3/2

stop_sim_time = np.inf
stop_iter = int(float(args['--niter']))

timestepper = d3.SBDF2
max_timestep = 1e-3
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'],  size=nz, bounds=(0, Lz), dealias=dealias)
x = dist.local_grid(xbasis)
z = dist.local_grid(zbasis)

ba = (xbasis,zbasis)
ba_p = (xbasis)
# Fields
p = dist.Field(name='p', bases=ba)
b = dist.Field(name='b', bases=ba)
u = dist.VectorField(coords, name='u', bases=ba)
τp = dist.Field(name='τp')
τ1b = dist.Field(name='τ1b', bases=ba_p)
τ2b = dist.Field(name='τ2b', bases=ba_p)
τ1u = dist.VectorField(coords, name='τ1u', bases=ba_p)
τ2u = dist.VectorField(coords, name='τ2u', bases=ba_p)

grid = lambda A: d3.Grid(A)
div = lambda A: d3.Divergence(A, index=0)
from dedalus.core.operators import Skew
skew = lambda A: Skew(A)
integ = lambda A: d3.Integrate(d3.Integrate(A, 'x'), 'z')
avg = lambda A: integ(A)/(Lx*Lz)
x_avg = lambda A: d3.Integrate(A, coords['x'])/(Lx)
dot = lambda A, B: d3.DotProduct(A, B)
grad = lambda A: d3.Gradient(A, coords)

# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)

zb1 = zbasis.clone_with(a=zbasis.a+1, b=zbasis.b+1)
zb2 = zbasis.clone_with(a=zbasis.a+2, b=zbasis.b+2)

ez = dist.VectorField(coords, name='ez', bases=zbasis)
ez1 = dist.VectorField(coords, name='ez1', bases=zb1)
ez2 = dist.VectorField(coords, name='ez2', bases=zb2)
ez['g'][1] = 1
ez1['g'][1] = 1
ez2['g'][1] = 1

ezg = grid(ez).evaluate()

lift = lambda A, n: d3.Lift(A, zb2, n)
lift1 = lambda A, n: d3.Lift(A, zb1, n)

b0 = dist.Field(name='b0', bases=zbasis)
b0['g'] = Lz - z

# Problem
problem = d3.IVP([p, b, u, τp, τ1b, τ2b, τ1u, τ2u], namespace=locals())
problem.add_equation("div(u) + lift1(τ2u,-1)@ez1 + τp = 0")
problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(τ2u,-2) + lift(τ1u,-1) - b*ez2 = -skew(grid(u))*div(skew(u))")
problem.add_equation("dt(b) + u@grad(b0) - kappa*lap(b) + lift(τ2b,-2) + lift(τ1b,-1) = - (u@grad(b))")
problem.add_equation("b(z=0) = 0")
problem.add_equation("u(z=0) = 0")
problem.add_equation("b(z=Lz) = 0")
problem.add_equation("u(z=Lz) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time
solver.stop_iteration = stop_iter

# Initial conditions
b.fill_random('g', seed=42, distribution='normal', scale=1e-5) # Random noise
b['g'] *= z * (Lz - z) # Damp noise at walls
b.low_pass_filter(scales=0.25)

cadence = 10
# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=1, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=cadence)
flow.add_property(np.sqrt(d3.dot(u,u))/nu, name='Re')

startup_iter = int(float(args['--nstart']))
# Main loop
try:
    good_solution = True
    logger.info('Starting loop')
    start_time = time.time()
    while solver.proceed and good_solution:
        if solver.iteration == startup_iter:
            main_start = time.time()
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % cadence == 0:
            avg_Re = flow.grid_average('Re')
            logger.info('Iteration={:d}, Time={:.3e}, dt={:.1e}, Re={:.2g}'. format(solver.iteration, solver.sim_time, timestep, avg_Re))
            good_solution = np.isfinite(avg_Re)
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()

    startup_time = main_start - start_time
    main_loop_time = end_time - main_start
    DOF = nx*nz
    niter = solver.iteration - startup_iter
    if rank==0:
        print('performance metrics:')
        print('    startup time   : {:}'.format(startup_time))
        print('    main loop time : {:}'.format(main_loop_time))
        print('    main loop iter : {:d}'.format(niter))
        print('    wall time/iter : {:f}'.format(main_loop_time/niter))
        print('          iter/sec : {:f}'.format(niter/main_loop_time))
        print('DOF-cycles/cpu-sec : {:}'.format(DOF*niter/(ncpu*main_loop_time)))
        print('scaling:',
              ' {:d} {:d} {:d}'.format(ncpu, nx, nz),
              ' {:12.7g} {:12.7g} {:12.7g} {:12.7g}'.format(startup_time,
                                                            main_loop_time,
                                                            main_loop_time/niter,
                                                            DOF*niter/(ncpu*main_loop_time)))
    solver.log_stats()
    logger.info("mode-stages/DOF = {}".format(solver.total_modes/DOF))
