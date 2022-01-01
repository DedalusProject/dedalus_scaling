"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.

Usage:
    rayleigh_benard_3d.py [options]

Options:
    --Nx=<Nx>              Horizontal modes; default is 2x Nz
    --Ny=<Ny>              Horizontal modes; default is 2x Nz
    --Nz=<Nz>              Vertical modes   [default: 64]

    --run_time_iter=<iter>      How many iterations to run for [default: 20]

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
Nz = int(args['--Nz'])
if args['--Nx']:
    Nx = int(args['--Nx'])
else:
    Nx = 2*Nz
if args['--Ny']:
    Ny = int(args['--Ny'])
else:
    Ny = 2*Nz

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
dealias = 3/2
stop_sim_time = 30
timestepper = d3.SBDF2 #RK222
max_timestep = 0.1
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, mesh=mesh, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
x = xbasis.local_grid(1)
y = ybasis.local_grid(1)
z = zbasis.local_grid(1)
ba = (xbasis,ybasis,zbasis)
ba_p = (xbasis,ybasis)

# Fields
p = dist.Field(name='p', bases=ba)
b = dist.Field(name='b', bases=ba)
u = dist.VectorField(coords, name='u', bases=ba)
tau1b = dist.Field(name='tau1b', bases=ba_p)
tau2b = dist.Field(name='tau2b', bases=ba_p)
tau1u = dist.VectorField(coords, name='tau1u', bases=ba_p)
tau2u = dist.VectorField(coords, name='tau2u', bases=ba_p)

# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)

ex = dist.VectorField(coords, name='ex')
ey = dist.VectorField(coords, name='ey')
ez = dist.VectorField(coords, name='ez')
ex['g'][0] = 1
ey['g'][1] = 1
ez['g'][2] = 1

exg = d3.Grid(ex).evaluate()
eyg = d3.Grid(ey).evaluate()
ezg = d3.Grid(ez).evaluate()

lift_basis = zbasis.clone_with(a=1/2, b=1/2) # First derivative basis
lift = lambda A, n: d3.LiftTau(A, lift_basis, n)
grad_u = d3.grad(u) + ez*lift(tau1u,-1) # First-order reduction
grad_b = d3.grad(b) + ez*lift(tau1b,-1) # First-order reduction

dx = lambda A: d3.Differentiate(A, coords[0])
dz = lambda A: d3.Differentiate(A, coords[1])
dot = lambda A, B: d3.DotProduct(A, B)
# curl_u_2d = dx(dot(u,ez)) - dz(dot(u,ex))
# curl_u_2d.store_last = True
# cross_u_curl_u = curl_u_2d*dot(u, exg)*ezg - curl_u_2d*dot(u, ezg)*exg

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, b, u, tau1b, tau2b, tau1u, tau2u], namespace=locals())
problem.add_equation("div(u) + dot(lift(tau2u,-1),ez) = 0")
problem.add_equation("dt(b) - kappa*lap(b) - dot(u,ez) + lift(tau2b,-2) + lift(tau1b,-1) = - dot(u,grad(b))")
problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau2u,-2) + lift(tau1u,-1) - b*ez = -dot(u,grad(u))") #cross(u, curl(u))")
problem.add_equation("b(z=0) = 0")
problem.add_equation("u(z=0) = 0")
problem.add_equation("b(z=Lz) = 0")
problem.add_equation("u(z=Lz) = 0", condition="nx != 0 or ny != 0")
problem.add_equation("dot(ex,u)(z=Lz) = 0", condition="nx == 0 and ny == 0")
problem.add_equation("dot(ey,u)(z=Lz) = 0", condition="nx == 0 and ny == 0")
problem.add_equation("p(z=Lz) = 0", condition="nx == 0 and ny == 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time
solver.stop_iteration = int(float(args['--run_time_iter']))

# Initial conditions
zb, zt = zbasis.bounds
b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
b['g'] *= z * (Lz - z) # Damp noise at walls
#b['g'] += Lz - z # Add linear background

# Analysis
# snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=50)
# snapshots.add_task(p)
# snapshots.add_task(b)
# snapshots.add_task(d3.dot(u,ex), name='ux')
# snapshots.add_task(d3.dot(u,ez), name='uz')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=1, safety=0.5, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(d3.dot(u,u))/nu, name='Re')

startup_iter = 10
# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.proceed:
        if solver.iteration == startup_iter:
            main_start = time.time()
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()

    startup_time = main_start - start_time
    main_loop_time = end_time - main_start
    DOF = Nx*Ny*Nz
    niter = solver.iteration - startup_iter
    if rank==0:
        print('performance metrics:')
        print('    startup time   : {:}'.format(startup_time))
        print('    main loop time : {:}'.format(main_loop_time))
        print('    main loop iter : {:d}'.format(niter))
        print('    wall time/iter : {:f}'.format(main_loop_time/niter))
        print('          iter/sec : {:f}'.format(niter/main_loop_time))
        print('DOF-cycles/cpu-sec : {:}'.format(DOF*niter/(ncpu*main_loop_time)))
    solver.log_stats()
