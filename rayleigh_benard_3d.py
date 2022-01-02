"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.

Usage:
    rayleigh_benard_3d.py [options]

Options:
    --nx=<nx>              Horizontal modes; default is 2x nz
    --ny=<ny>              Horizontal modes; default is 2x nz
    --nz=<nz>              Vertical modes   [default: 64]

    --niter=<iter>         Timing iterations [default: 100]

    --dealias=<dealias>         Dealiasing [default: 1.5]

    --mesh=<mesh>               Parallel decomposition mesh
"""
from mpi4py import MPI
import numpy as np
import time
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

from docopt import docopt
args = docopt(__doc__)

# TODO: maybe fix plotting to directly handle vectors
# TODO: optimize and match d2 resolution
# TODO: get unit vectors from coords?

comm = MPI.COMM_WORLD
rank = comm.rank
ncpu = comm.size

# Parameters
Lx, Ly, Lz = 4, 4, 1
nz = int(args['--nz'])
if args['--nx']:
    nx = int(args['--nx'])
else:
    nx = 2*nz
if args['--ny']:
    ny = int(args['--ny'])
else:
    ny = 2*nz

mesh = args['--mesh']
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]
else:
    log2 = np.log2(ncpu)
    if log2 == int(log2):
        mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
logger.info("running on processor mesh={}".format(mesh))


Rayleigh = 1e6
Prandtl = 1
dealias = float(args['--dealias'])
stop_sim_time = np.inf
timestepper = d3.SBDF2 #RK222
max_timestep = 0.1
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, mesh=mesh, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=ny, bounds=(0, Ly), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'],  size=nz, bounds=(0, Lz), dealias=dealias)
x = xbasis.local_grid(1)
y = ybasis.local_grid(1)
z = zbasis.local_grid(1)
ba = (xbasis,ybasis,zbasis)
ba_p = (xbasis,ybasis)

# Fields
p = dist.Field(name='p', bases=ba)
b = dist.Field(name='b', bases=ba)
u = dist.VectorField(coords, name='u', bases=ba)
τp = dist.Field(name='τp')
τ1b = dist.Field(name='τ1b', bases=ba_p)
τ2b = dist.Field(name='τ2b', bases=ba_p)
τ1u = dist.VectorField(coords, name='τ1u', bases=ba_p)
τ2u = dist.VectorField(coords, name='τ2u', bases=ba_p)

# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)

zb1 = zbasis.clone_with(a=zbasis.a+1, b=zbasis.b+1)
zb2 = zbasis.clone_with(a=zbasis.a+2, b=zbasis.b+2)

ez = dist.VectorField(coords, name='ez', bases=zb2)
ez1 = dist.VectorField(coords, name='ez', bases=zb1)
ez['g'][2] = 1
ez1['g'][2] = 1

lift1 = lambda A, n: d3.LiftTau(A, zb1, n)
lift = lambda A, n: d3.LiftTau(A, zb2, n)

integ = lambda A: d3.Integrate(d3.Integrate(d3.Integrate(A, 'x'),'y'),'z')

b0 = dist.Field(name='b0', bases=zbasis)
b0['g'] = Lz - z

dot = lambda A, B: d3.DotProduct(A, B)

# Problem
problem = d3.IVP([p, b, u, τp, τ1b, τ2b, τ1u, τ2u], namespace=locals())
problem.add_equation("div(u) + dot(lift1(τ2u,-1),ez1) + τp = 0")
problem.add_equation("dt(b) + dot(u, grad(b0)) - kappa*lap(b) + lift(τ2b,-2) + lift(τ1b,-1) = - dot(u,grad(b))")
# TODO: go to cross(u, curl(u)) form of momentum nonlinearity
problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(τ2u,-2) + lift(τ1u,-1) - b*ez = -dot(u,grad(u))") #cross(u, curl(u))")
problem.add_equation("b(z=0) = 0")
problem.add_equation("u(z=0) = 0")
problem.add_equation("b(z=Lz) = 0")
problem.add_equation("u(z=Lz) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time
solver.stop_iteration = int(float(args['--niter']))

# Initial conditions
zb, zt = zbasis.bounds
b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
b['g'] *= z * (Lz - z) # Damp noise at walls
b.low_pass_filter(scales=0.25)

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=1, safety=0.5, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

cadence = 10
# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=cadence)
flow.add_property(np.sqrt(d3.dot(u,u))/nu, name='Re')

startup_iter = 10
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
    DOF = nx*ny*nz
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
