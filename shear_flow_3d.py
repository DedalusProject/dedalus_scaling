"""
Dedalus script simulating 3D triply-periodic shear flow.

Usage:
    shear_flow_3d.py [options]

Options:
    --nx=<nx>              Horizontal modes; default is nz
    --ny=<ny>              Horizontal modes; default is nz
    --nz=<nz>              Vertical modes   [default: 1024]

    --niter=<iter>         Timing iterations [default: 100]
    --nstart=<nstart>      Startup iterations [default: 10]

    --dealias=<dealias>    Dealiasing [default: 1.5]

    --mesh=<mesh>          Parallel decomposition mesh
"""
from mpi4py import MPI
import numpy as np
import dedalus.public as de
import logging
logger = logging.getLogger(__name__)

from docopt import docopt
args = docopt(__doc__)

comm = MPI.COMM_WORLD
rank = comm.rank
ncpu = comm.size

# Parameters
LL = 1
nz = int(args['--nz'])
if args['--nx']:
    nx = int(args['--nx'])
else:
    nx = nz
if args['--ny']:
    ny = int(args['--ny'])
else:
    ny = nz

mesh = args['--mesh']
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]
else:
    log2 = np.log2(ncpu)
    if log2 == int(log2):
        mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
logger.info("running on processor mesh={}".format(mesh))


Reynolds = 5e4
Schmidt = 1
dealias = float(args['--dealias'])
stop_iteration = int(float(args['--niter'])) + int(float(args['--nstart']))
timestepper = de.RK222
timestep = 0.001
dtype = np.float64

# Bases
coords = de.CartesianCoordinates('x', 'y', 'z')
dist = de.Distributor(coords, dtype=dtype, mesh=mesh)
xbasis = de.RealFourier(coords['x'], size=nx, bounds=(0, LL), dealias=dealias)
ybasis = de.RealFourier(coords['y'], size=ny, bounds=(0, LL), dealias=dealias)
zbasis = de.RealFourier(coords['z'], size=nz, bounds=(0, LL), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,ybasis,zbasis))
s = dist.Field(name='s', bases=(xbasis,ybasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis,zbasis))
tau_p = dist.Field(name='tau_p')

curl = lambda A: de.Curl(A)
ω = curl(u)

# Substitutions
nu = 1 / Reynolds
D = nu / Schmidt
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
ex, ey, ez = coords.unit_vector_fields(dist)

# Problem
problem = de.IVP([u, s, p, tau_p], namespace=locals())
problem.add_equation("dt(u) + grad(p) - nu*lap(u) = cross(u, ω)")
problem.add_equation("dt(s) - D*lap(s) = -(u@grad(s))")
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = np.inf
solver.stop_iteration = stop_iteration

# Initial conditions
# Background shear
u['g'][0] = 1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1))
# Match tracer to shear
s['g'] = u['g'][0]
# Add small vertical velocity perturbations localized to the shear layers
u['g'][1] += 0.1 * np.sin(2*np.pi*x/LL) * np.exp(-(z-0.5)**2/0.01)
u['g'][1] += 0.1 * np.sin(2*np.pi*x/LL) * np.exp(-(z+0.5)**2/0.01)

cadence = 100
# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        solver.step(timestep)
        if (solver.iteration-1) % cadence == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
