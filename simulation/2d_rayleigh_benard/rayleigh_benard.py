"""
Dedalus script for 2D Rayleigh-Benard convection.

This script uses a Fourier basis in the x direction with periodic boundary
conditions.  The equations are scaled in units of the buoyancy time (Fr = 1).

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge_procs` command can
be used to merge distributed analysis sets from parallel runs, and the
`plot_slices.py` script can be used to plot the snapshots.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs snapshots
    $ mpiexec -n 4 python3 plot_slices.py snapshots/*.h5

This script can restart the simulation from the last save of the original
output to extend the integration.  This requires that the output files from
the original simulation are merged, and the last is symlinked or copied to
`restart.h5`.

To run the original example and the restart, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs snapshots
    $ ln -s snapshots/snapshots_s2.h5 restart.h5
    $ mpiexec -n 4 python3 rayleigh_benard.py

The simulations should take a few process-minutes to run.

"""

import numpy as np
from mpi4py import MPI
import time
import pathlib
import argparse
import os

from dedalus import public as d3
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


def get_args():
    
    parser = argparse.ArgumentParser(
        description='Simulation script for Rayleigh-Benard 2D using Dedalus')

    ## high resolution actual
    parser.add_argument('--dir', default='outputs/rb2d_DEFAULT', type=str,
                        help='Output directory location')
    parser.add_argument('--lx', default=4.0, type=float,
                        help='Physical length in x dimension. (default: 4.0)')
    parser.add_argument('--lz', default=1.0, type=float,
                        help='Physical length in z dimension. (default: 1.0)')
    parser.add_argument('--res_x', default=512, type=int,
                        help='Simulation resolution in x dimension. (default: 512)')
    parser.add_argument('--res_z', default=128, type=int,
                        help='Simulation resolution in z dimension. (default: 128)')
    ## replaced by time stepper??
    # parser.add_argument('--dt', default=0.125, type=float,
                        # help='Simulation step size in time. (default: 0.125)')
    parser.add_argument('--stop_sim_time', default=50., type=float,
                        help='Simulation stop time. (default: 50)')
    parser.add_argument('--rayleigh', default=1e6, type=float,
                        help='Simulation Rayleigh number. (default: 1e6)')
    parser.add_argument('--prandtl', default=1., type=float,
                        help='Simulation Prandtl number. (default: 1.0)')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed for initial perturbations. (default: 42)')
                        
    ## low resolution for testing!
    # parser.add_argument('--dir', default='outputs/rb2d_DEFAULT', type=str,
                        # help='Output directory location')
    # parser.add_argument('--lx', default=4.0, type=float,
                        # help='Physical length in x dimension. (default: 4.0)')
    # parser.add_argument('--lz', default=1.0, type=float,
                        # help='Physical length in z dimension. (default: 1.0)')
    # parser.add_argument('--res_x', default=64, type=int,
                        # help='Simulation resolution in x dimension. (default: 512)')
    # parser.add_argument('--res_z', default=32, type=int,
                        # help='Simulation resolution in z dimension. (default: 128)')
    # ## replaced by time stepper??
    # # parser.add_argument('--dt', default=0.125, type=float,
                        # # help='Simulation step size in time. (default: 0.125)')
    # parser.add_argument('--stop_sim_time', default=10., type=float,
                        # help='Simulation stop time. (default: 50)')
    # parser.add_argument('--rayleigh', default=1e6, type=float,
                        # help='Simulation Rayleigh number. (default: 1e6)')
    # parser.add_argument('--prandtl', default=1., type=float,
                        # help='Simulation Prandtl number. (default: 1.0)')
    # parser.add_argument('--seed', default=42, type=int,
                        # help='Random seed for initial perturbations. (default: 42)')
    
    args = parser.parse_args()
    return args
    

def main():

    ## 2/16/23
    ## for masters thesis, "glorified grad report", 40 pages, short and sweet and concise, brief introduction (5 pages), describe problem/investigation and tools to use
    ## then graphs and demonstration
    ## discussion and conclusion
    ## layout
    
    ## todo: start writing beginning of thesis with background (based on the meshfree flow net)
    
    ## how to compare outputs of low resolution with superresolution fto high resolution
    ##  - can compare the average temperature as function of height
    ##  - variance in temperature at certain height
    ##  - averaging vorticity out of plane component
    ## done after steady state...
    ##  - values over time
    ##  
    ## 2/23/23
    ## continuing off of before, can also include other features that are used to assess performance in mffn
    
    args = get_args() 
    
    ## can't do this in the MPI...it will make multiple
    # os.mkdir(f"{args.dir}")
    
    # Parameters
    # modifying resolution to see how superresolution works
    
    
    Lx, Lz = args.lx, args.lz # width, height of 2-D box
    Nx, Nz = args.res_x, args.res_z # resolution x, resolution z
    Rayleigh = args.rayleigh # rayleigh number for the fluid. since the number is high, convection naturally occurs
        # thermal transport via diffusion/(thermal transport via convection)
    Prandtl = args.prandtl # momentum diffusivity / thermal diffusivity
    dealias = 3/2
    stop_sim_time = args.stop_sim_time
    timestepper = d3.RK222
    # maybe related to variable timestep?
    max_timestep = 0.125
    dtype = np.float64

    # Bases
    coords = d3.CartesianCoordinates('x', 'z') # regular x/z 2d
    dist = d3.Distributor(coords, dtype=dtype) # directs parallelized distribution and transformation of fields
    # distributing d dimensional data over r dimensional mesh
    ## https://dedalus-project.readthedocs.io/en/latest/autoapi/dedalus/core/distribuqtor/index.html

    # specific transformations of fields
    ## Fourier
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
    ## chebyshev (https://en.wikipedia.org/wiki/Discrete_Chebyshev_transform)
    zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

    # Fields
    # in newer version, vectorized velocities
    p = dist.Field(name='p', bases=(xbasis,zbasis))
    b = dist.Field(name='b', bases=(xbasis,zbasis))
    
    ## duplicating for the code? not sure how to do this after the fact...
    # ux = dist.Field(name='ux', bases=(xbasis,zbasis))
    # uz = dist.Field(name='uz', bases=(xbasis,zbasis))
    
    u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
    tau_p = dist.Field(name='tau_p')
    tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
    tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

    # Substitutions
    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)
    x, z = dist.local_grids(xbasis, zbasis)
    ## unit vector fields!!
    ex, ez = coords.unit_vector_fields(dist)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
    grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction

    #ux = u["g"][0]
    #uz = u["g"][1]
    
    # Problem
    # First-order form: "div(f)" becomes "trace(grad_f)"
    # First-order form: "lap(f)" becomes "div(grad_f)"

    ## all the values that will be solved for, not sure what the tau's are?
    ## taus are used to enforce the boundary conditions. in this case, we use 4, 2 for b and 2 for u
    problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
    problem.add_equation("trace(grad_u) + tau_p = 0")
    problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
    problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")
    problem.add_equation("b(z=0) = Lz")
    problem.add_equation("u(z=0) = 0")
    problem.add_equation("b(z=Lz) = 0")
    problem.add_equation("u(z=Lz) = 0")
    problem.add_equation("integ(p) = 0") # Pressure gauge

    # Solver 
    ## see if you can do fixed timesteps? usually flag that can be implemented
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time
    logger.info('Solver built')

    # # Initial conditions or restart
    # if not pathlib.Path('restart.h5').exists():

        # # Initial conditions
        # # x = domain.grid(0)
        # # z = domain.grid(1)
        # b = solver.state['b']
        # bz = solver.state['bz']

        # # Random perturbations, initialized globally for same results in parallel
        # # gshape = domain.dist.grid_layout.global_shape(scales=1)
        # # slices = domain.dist.grid_layout.slices(scales=1)
        # # rand = np.random.RandomState(seed=seed)
        # # noise = rand.standard_normal(gshape)[slices]

        # # Linear background + perturbations damped at walls
        # zb, zt = z_basis.interval
        # pert =  1e-3 * noise * (zt - z) * (z - zb)
        # b['g'] += F * pert
        # b.differentiate('z', out=bz)

        # # Timestepping and output
        # dt = args.dt
        # stop_sim_time = args.stop_sim_time
        # fh_mode = 'overwrite'
    # else:
        # # Restart
        # write, last_dt = solver.load_state('restart.h5', -1)

        # # Timestepping and output
        # dt = last_dt
        # stop_sim_time = args.stop_sim_time
        # fh_mode = 'append'
    
    # logger.info(f"mode: {fh_mode}")
    
    ## restart, or overwrite the file
    if pathlib.Path('restart.h5').exists():
        solver.load_state('restart.h5', -1)
        fh_mode = 'append'
    else:
        fh_mode = 'overwrite'
        
    print(fh_mode) 
    # Initial conditions
    b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
    b['g'] *= z * (Lz - z) # Damp noise at walls
    b['g'] += Lz - z # Add linear background

    # Analysis
    ## modify to take different snapshots of the fluid, stored in "tasks" in the hd5 file!
    ## in snap shots, lets save these: 'p', 'b', 'u', 'w', 'bz', 'uz', 'wz'
    
    
    snapshot_directory = f"{args.dir}/snapshots"
    
    # snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50, mode=fh_mode)
    snapshots = solver.evaluator.add_file_handler(snapshot_directory, sim_dt=0.25, max_writes=50, mode=fh_mode)
    # 90 degree positive rotation of 2d vector field (why?)
    
    ## properties to add 
    """
    Below is a description of the variables in this file:
    - p: pressure, shape (200, 512, 128)
    - b: temperature, shape (200, 512, 128)
    - u: velocity in the x direction, shape (200, 512, 128)
    - w: velocity in the z direction, shape (200, 512, 128)
    - bz: the z derivative of b, shape (200, 512, 128)
    - uz: the z derivative of u, shape (200, 512, 128)
    - wz: the z derivative of w, shape (200, 512, 128)
    - write_number: the sequence index of the simulation frames
    - sim_time: simulation time.
    """

    snapshots.add_task(b, name='buoyancy')
    snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
    snapshots.add_task(u, name='flow velocity')
    snapshots.add_task(np.sqrt(u@u)/nu, name='Re')
    #snapshots.add_task(u, name='u')
    #snapshots.add_system(solver.state) ## doesn't work!
    
    snapshots.add_task(p, name='p')
    snapshots.add_task(b, name='b')
    snapshots.add_task(u@ex, name='u')
    snapshots.add_task(u@ez, name='w')
    snapshots.add_task(d3.Differentiate(b, coords['z']), name='bz')
    snapshots.add_task(d3.Differentiate(u@ex, coords['z']), name='uz')
    snapshots.add_task(d3.Differentiate(u@ez, coords['z']), name='wz')
    
    # CFL
    CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
                 max_change=1.5, min_change=0.5, max_dt=max_timestep)
    CFL.add_velocity(u)

    # Flow properties
    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property(np.sqrt(u@u)/nu, name='Re')
    # @ matrix component multiplication
    ## Reynolds number changes with the velocity of system: v*<characteristic length scale>/<viscosity>
    


    # Main loop
    startup_iter = 10
    try:
        logger.info('Starting main loop')
        start_time = time.time()
        while solver.proceed:
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
        logger.info('Iterations: %i' %solver.iteration)
        logger.info('Sim end time: %f' %solver.sim_time)
        logger.info('Run time: %.2f sec' %(end_time-start_time))
        #logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))

if __name__ == '__main__':
    main()
    
## TO DO: 
## try to implement previous version of the rb simulations with the previous version of code
## with modified code, see outputs and compare
## try with fixed timestep/dynamic

## updates 3/11/23
## in the simulation:
## changed scripts so that it stores the relevant information needed for using in the MLP in mffn
## compare to "old", and new "run_simulation_steps.sh" which uses the new code
## in the experiment:
## a couple changes, a lot of annoyances with the cuda drivers...requires wsl version 2 and also cuda drivers/toolkit installed!