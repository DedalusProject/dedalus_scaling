"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.

Usage:
    rayleigh_benard_2d.py [options]

Options:
    --nx=<nx>              Horizontal modes; default is aspect x nz
    --nz=<nz>              Vertical modes [default: 64]

    --aspect=<aspect>      Aspect ratio of domain [default: 4]

    --Rayleigh=<Rayleigh>  Rayleigh number [default: 1e6]

    --niter=<iter>         Timing iterations [default: 100]
    --nstart=<nstart>      Startup iterations [default: 10]

    --dealias=<dealias>    Dealiasing [default: 1.5]
"""
from mpi4py import MPI
import numpy as np
import time

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

import dedalus.public as de

import logging
logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD
rank = comm.rank
ncpu = comm.size

Rayleigh = float(args['--Rayleigh'])
Prandtl = 1
dealias = float(args['--dealias'])

stop_sim_time = np.inf

timestepper = de.SBDF2
max_timestep = 1e-3
dtype = np.float64

# Bases
coords = de.CartesianCoordinates('y', 'x', 'z', right_handed=False)
dist = de.Distributor(coords, mesh=[1,ncpu], dtype=dtype)
xbasis = de.RealFourier(coords['x'], size=nx, bounds=(0, Lx), dealias=dealias)
zbasis = de.ChebyshevT(coords['z'],  size=nz, bounds=(0, Lz), dealias=dealias)
x = dist.local_grid(xbasis)
z = dist.local_grid(zbasis)

ba = (xbasis,zbasis)
ba_p = (xbasis)
# Fields
p = dist.Field(name='p', bases=ba)
b = dist.Field(name='b', bases=ba)
u = dist.VectorField(coords, name='u', bases=ba)
τp = dist.Field(name='τp')
τb1 = dist.Field(name='τb1', bases=ba_p)
τb2 = dist.Field(name='τb2', bases=ba_p)
τu1 = dist.VectorField(coords, name='τu1', bases=ba_p)
τu2 = dist.VectorField(coords, name='τu2', bases=ba_p)

curl = lambda A: de.Curl(A)
ω = curl(u)

# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)

ey, ex, ez = coords.unit_vector_fields(dist)

lift_basis = zbasis.derivative_basis(1)
lift = lambda A: de.Lift(A, lift_basis, -1)

lift_basis2 = zbasis.derivative_basis(2)
lift2 = lambda A: de.Lift(A, lift_basis2, -1)
lift2_2 = lambda A: de.Lift(A, lift_basis2, -2)

b0 = dist.Field(name='b0', bases=zbasis)
b0['g'] = Lz - z

# Problem
problem = de.IVP([p, u, b, τp, τu1, τu2, τb1, τb2], namespace=locals())
problem.add_equation("div(u) + lift(τp) = 0")
problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift2_2(τu1) + lift2(τu2) - b*ez = cross(u, ω)")
problem.add_equation("dt(b) + u@grad(b0) - kappa*lap(b) + lift2_2(τb1) + lift2(τb2) = - (u@grad(b))")
problem.add_equation("u(z=0) = 0", condition="nx!=0")
problem.add_equation("p(z=0) = 0", condition="nx==0") # Pressure gauge
problem.add_equation("ex@u(z=0) = 0", condition="nx==0")
problem.add_equation("ey@u(z=0) = 0", condition="nx==0")
problem.add_equation("ez@τu1 = 0", condition="nx==0")
problem.add_equation("b(z=0) = 0")
problem.add_equation("u(z=Lz) = 0")
problem.add_equation("b(z=Lz) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time
solver.stop_iteration = int(float(args['--niter'])) + int(float(args['--nstart']))

# Initial conditions
b.fill_random('g', seed=42, distribution='normal', scale=1e-5) # Random noise
b['g'] *= z * (Lz - z) # Damp noise at walls
b.low_pass_filter(scales=0.25)

cadence = 100
# CFL
CFL = de.CFL(solver, initial_dt=max_timestep, cadence=1, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = de.GlobalFlowProperty(solver, cadence=cadence)
flow.add_property(np.sqrt(u@u)/nu, name='Re')

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
