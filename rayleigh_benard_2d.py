"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.

Usage:
    rayleigh_benard_2d.py [options]

Options:
    --Nx=<Nx>              Horizontal modes; default is aspect x Nz
    --Nz=<Nz>              Vertical modes [default: 64]
    --aspect=<aspect>      Aspect ratio of domain [default: 4]

    --tau_drag=<tau_drag>       1/Newtonian drag timescale; default is zero drag

    --Rayleigh=<Rayleigh>       Rayleigh number [default: 1e6]

    --run_time_iter=<iter>      How many iterations to run for
    --run_time_simtime=<run>    How long (simtime) to run for

    --label=<label>             Additional label for run output directory
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
Nz = int(args['--Nz'])
if args['--Nx']:
    Nx = int(args['--Nx'])
else:
    Nx = int(aspect*Nz)

data_dir = './'+sys.argv[0].split('.py')[0]
data_dir += '_Ra{}'.format(args['--Rayleigh'])
if args['--tau_drag']:
    τ_drag = float(args['--tau_drag'])
    data_dir += '_tau{}'.format(args['--tau_drag'])
else:
    τ_drag = 0
data_dir += '_Nz{}_Nx{}'.format(Nz, Nx)
if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])

import logging
logger = logging.getLogger(__name__)
dlog = logging.getLogger('evaluator')
dlog.setLevel(logging.WARNING)

from dedalus.tools.config import config
config['logging']['filename'] = os.path.join(data_dir,'logs/dedalus_log')
config['logging']['file_level'] = 'DEBUG'

from dedalus.tools.parallel import Sync
with Sync() as sync:
    if sync.comm.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))
        logdir = os.path.join(data_dir,'logs')
        if not os.path.exists(logdir):
            os.mkdir(logdir)

import dedalus.public as d3

# TODO: maybe fix plotting to directly handle vectors
# TODO: optimize and match d2 resolution
# TODO: get unit vectors from coords?

comm = MPI.COMM_WORLD
rank = comm.rank
ncpu = comm.size

Rayleigh = float(args['--Rayleigh'])
Prandtl = 1
dealias = 3/2
if args['--run_time_simtime']:
    stop_sim_time = float(args['--run_time_simtime'])
else:
    stop_sim_time = np.inf
if args['--run_time_iter']:
    stop_iter = int(float(args['--run_time_iter']))
else:
    stop_iter = np.inf
timestepper = d3.SBDF2
max_timestep = 0.1
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
x = xbasis.local_grid(1)
z = zbasis.local_grid(1)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
b = dist.Field(name='b', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
taup = dist.Field(name='taup')
tau1b = dist.Field(name='tau1b', bases=xbasis)
tau2b = dist.Field(name='tau2b', bases=xbasis)
tau1u = dist.VectorField(coords, name='tau1u', bases=xbasis)
tau2u = dist.VectorField(coords, name='tau2u', bases=xbasis)

grid = lambda A: d3.Grid(A)
div = lambda A: d3.Divergence(A, index=0)
from dedalus.core.operators import Skew
skew = lambda A: Skew(A)
#avg = lambda A: d3.Integrate(A, coords)/(Lx*Lz)
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
ex = dist.VectorField(coords, name='ex')
ez = dist.VectorField(coords, name='ez', bases=(zbasis))

ez1 = dist.VectorField(coords, name='ez1', bases=(zb1))
ez2 = dist.VectorField(coords, name='ez2', bases=(zb2))
ex['g'][0] = 1
ez['g'][1] = 1
ez1['g'][1] = 1
ez2['g'][1] = 1

exg = grid(ex).evaluate()
ezg = grid(ez).evaluate()

lift_basis = zbasis.clone_with(a=zbasis.a+2, b=zbasis.b+2)
lift = lambda A, n: d3.LiftTau(A, lift_basis, n)

lift_basis1 = zbasis.clone_with(a=zbasis.a+1, b=zbasis.b+1)
lift1 = lambda A, n: d3.LiftTau(A, lift_basis1, n)

b0 = dist.Field(name='b0', bases=zbasis)
b0['g'] = Lz - z

# Problem
problem = d3.IVP([p, b, u, taup, tau1b, tau2b, tau1u, tau2u], namespace=locals())
problem.add_equation("div(u) + dot(lift1(tau2u,-1),ez1) + taup = 0")
problem.add_equation("dt(u) + τ_drag*u - nu*lap(u) + grad(p) + lift(tau2u,-2) + lift(tau1u,-1) - b*ez2 = -skew(grid(u))*div(skew(u))")
problem.add_equation("dt(b) + dot(u, grad(b0)) - kappa*lap(b) + lift(tau2b,-2) + lift(tau1b,-1) = - dot(u,grad(b))")
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
zb, zt = zbasis.bounds
b.fill_random('g', seed=42, distribution='normal', scale=1e-5) # Random noise
b['g'] *= z * (Lz - z) # Damp noise at walls
b.low_pass_filter(scales=0.25)



KE = 0.5*dot(u,u)
PE = b+b0
ω = -div(skew(u))
flux_c = dot(u, ez)*(b0+b)
flux_c.store_last=True
flux_κ = -kappa*dot(grad(b+b0),ez)
flux_κ.store_last=True

# Analysis
snapshots = solver.evaluator.add_file_handler(data_dir+'/snapshots', sim_dt=0.1, max_writes=10)
snapshots.add_task(b+b0, name='b')
snapshots.add_task(ω, name='vorticity')
snapshots.add_task(ω**2, name='enstrophy')

traces = solver.evaluator.add_file_handler(data_dir+'/traces', sim_dt=0.1, max_writes=np.inf)
traces.add_task(avg(KE), name='KE')
traces.add_task(avg(PE), name='PE')
traces.add_task(np.sqrt(2*avg(KE))/nu, name='Re')
traces.add_task(avg(ω**2), name='enstrophy')
traces.add_task(1 + avg(flux_c)/avg(flux_κ), name='Nu')
traces.add_task(x_avg(np.sqrt(dot(tau1u,tau1u))), name='τu1')
traces.add_task(x_avg(np.sqrt(dot(tau2u,tau2u))), name='τu2')
traces.add_task(x_avg(np.sqrt(tau1b**2)), name='τb1')
traces.add_task(x_avg(np.sqrt(tau2b**2)), name='τb2')
traces.add_task(np.sqrt(taup**2), name='τp')

cadence = 10
# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=1, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=cadence)
flow.add_property(np.sqrt(d3.dot(u,u))/nu, name='Re')
flow.add_property(KE, name='KE')
flow.add_property(PE, name='PE')
flow.add_property(flux_c, name='f_c')
flow.add_property(flux_κ, name='f_κ')
flow.add_property(np.sqrt(dot(tau1u,tau1u)), name='τu1')
flow.add_property(np.sqrt(dot(tau2u,tau2u)), name='τu2')
flow.add_property(np.sqrt(tau1b**2), name='τb1')
flow.add_property(np.sqrt(tau2b**2), name='τb2')
flow.add_property(np.sqrt(tau2b**2), name='τb2')
flow.add_property(np.sqrt(taup**2), name='τp')

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
            max_Re = flow.max('Re')
            avg_Re = flow.grid_average('Re')
            avg_PE = flow.grid_average('PE')
            avg_KE = flow.grid_average('KE')
            avg_Nu = 1+flow.grid_average('f_c')/flow.grid_average('f_κ')
            max_τ = np.max([flow.max('τu1'),flow.max('τu2'),flow.max('τb1'),flow.max('τb2'),flow.max('τp')])
            logger.info('Iteration={:d}, Time={:.3e}, dt={:.1e}, PE={:.3e}, KE={:.3e}, Re={:.2g}, Nu={:.2g}, τ={:.2e}'. format(solver.iteration, solver.sim_time, timestep, avg_PE, avg_KE, avg_Re, avg_Nu, max_τ))
            good_solution = np.isfinite(max_Re)
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()

    startup_time = main_start - start_time
    main_loop_time = end_time - main_start
    DOF = Nx*Nz
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
