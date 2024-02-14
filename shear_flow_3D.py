import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Parameters
LL = 1
NN = 1024
Reynolds = 5e4
Schmidt = 1
dealias = 3/2
stop_iteration = 200
timestepper = d3.RK222
timestep = 0.001
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=dtype, mesh=(128,128))
xbasis = d3.RealFourier(coords['x'], size=NN, bounds=(0, LL), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=NN, bounds=(0, LL), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=NN, bounds=(0, LL), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,ybasis,zbasis))
s = dist.Field(name='s', bases=(xbasis,ybasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis,zbasis))
tau_p = dist.Field(name='tau_p')

# Substitutions
nu = 1 / Reynolds
D = nu / Schmidt
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
ex, ey, ez = coords.unit_vector_fields(dist)

# Problem
problem = d3.IVP([u, s, p, tau_p], namespace=locals())
problem.add_equation("dt(u) + grad(p) - nu*lap(u) = - u@grad(u)")
problem.add_equation("dt(s) - D*lap(s) = - u@grad(s)")
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper, profile=True, profile_mode="full")
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

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

