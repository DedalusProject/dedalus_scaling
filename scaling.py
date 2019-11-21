#!/usr/bin/env python3
"""
Perform scaling runs on special scaling scripts.  This driver script
should be run serially, and it will then spawn off a series of MPI
processes to test the scaling performance of a given machine.

The target script <scaling_script> is assumed to take the following
command line inputs:

2-D scaling script (1-D processor decomposition):

   scaling_script.py --nz=nz --nx=nx

3-D scaling script (1- or 2-D processor decomposition):

   scaling_script.py --nz=nz --nx=nx --ny=ny --mesh=p1,p2

where nz is the Chebyshev modal resolution, and nx and ny are the
Fourier modal resolutions.  The mesh keyword should accept the 2-D
processor mesh, with p1 and p2 the processor mesh values.

If the 3-D scaling script is not passed the mesh keyword, it should default
to a 1-D domain decomposition.

These scaling scripts should output well formated scaling outputs,
following the example scripts.  In a future revision, that output will
be rolled into this scaling.py package.


Usage:
    scaling.py run <scaling_script> [<nz> options]
    scaling.py plot <files>... [options]

Options:
    <nz>                        resolution in z (fourier or chebyshev) direction [default: 256]
    --nx=<nx>                   resolution in x (fourier); default is nz
    --ny=<ny>                   resolution in y (fourier); default is nz
    --label=<label>             Label for output file
    --niter=<niter>             Number of iterations to run for [default: 100]
    --verbose                   Print verbose output at end of each run (stdout and stderr)
    --3D                        Run 3D script with 2D mesh domain decomposition
    --one-pencil                Push to one pencil per core in coeff space
    --test-type=<test-type>     Mesh-selection strategy [default: exhaustive]
    --max-cores=<max-cores>     Max number of available cores
    --min-cores=<min-cores>     Min number of cores to use
    --output=<dir>              Output directory [default: ./scaling]
    --clean_plot                Remove run-specific labels during plotting (e.g., for proposals or papers)
    --OpenMPI                   Assume we're in an OpenMPI env; default if nothing else is selected
    --MPISGI                    Assume we're in a SGI-MPT env (e.g., NASA Pleiades)
    --IntelMPI                  Assume we're in an IntelMPI env (e.g., PSC Bridges)

"""

import os
import numpy as np
import itertools
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import time
import h5py
import pathlib


def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


######################
## Scaling Routines ##
######################

def build_mesh_list(n_z, mesh_dim=2, test_type='exhaustive', one_pencil=None, max_cores=None, min_cores=None):

        if test_type == "simple":
            nz = n_z
            # Try all powers of 2 between min and max
            ln2_min = np.floor(np.log2(min_cores))
            ln2_max = np.floor(np.log2(max_cores))
            corelist = 2**np.arange(ln2_min, ln2_max+1, dtype=int)
            # Build mesh list
            if mesh_dim == 1:
                mesh_list = []
                # Assume nx = nz
                nx = nz
                px = nx // 2
                # Scale until empty cores
                for cores in corelist:
                    if cores <= px:
                        mesh_list.append(cores)
            elif mesh_dim == 2:
                mesh_list = []
                # Assume nx = ny = nz
                nx = ny = nz
                px = nx // 2
                py = ny
                # Scale fully over x, then over y until empty cores
                for cores in corelist:
                    if cores <= px:
                        mesh_list.append((cores, 1))
                    elif cores <= (px * py):
                        mesh_list.append((px, cores//px))
            return mesh_list

        if one_pencil:
            print("Pushing to one pencil per core in coeff space; this may be inefficient depending on dealias padding choice.")
            n_z_2 = np.log(n_z)/np.log(2)
        else:
            n_z_2 = np.log(n_z)/np.log(2)-1 # 2 pencils per core min

        if max_cores is not None:
            log2_max = np.log(max_cores)/np.log(2)
            if mesh_dim == 2:
                log2_max = log2_max/2
            log2_max = np.floor(log2_max)
            if n_z_2 > log2_max:
                n_z_2 = log2_max

        log_2_span = 3
        n_z_2_min = n_z_2-log_2_span

        if min_cores is not None:
            min_cores = np.int(args['--min-cores'])
            log2_min = np.log(min_cores)/np.log(2)
            if mesh_dim == 2:
                log2_min = log2_min/2
            log2_min = np.ceil(log2_min)

            n_z_2_min = log2_min

        n_z_2 = np.floor(n_z_2)
        n_z_2_min = np.ceil(n_z_2_min)

        CPU_set = (2**np.arange(n_z_2_min, n_z_2+1)).astype(int)[::-1] # flip order so large numbers of cores are done first (and arange goes to -1 of top)

        if mesh_dim == 2:
            import itertools
            CPU_set_1 = CPU_set
            CPU_set_2 = CPU_set

            if max_cores is not None:
                if (np.max(CPU_set_1)**2) < max_cores:
                    # append new element to front of set_2
                    CPU_set_2 = np.append(2*np.max(CPU_set_2), CPU_set_2)
            if min_cores is not None:
                if (np.min(CPU_set_1)*np.min(CPU_set_2)) > min_cores:
                    # append new element to end of set_1
                    CPU_set_1 = np.append(CPU_set_1, np.int(np.min(CPU_set_1)/2))
            print('testing from {:d} to {:d} cores'.format(np.min(CPU_set_1)*np.min(CPU_set_2),np.max(CPU_set_1)*np.max(CPU_set_2)))
            if test_type=='exhaustive':
                print('doing exhaustive scaling test')
                CPU_set = itertools.product(CPU_set_1, CPU_set_2)
            elif test_type=='patient':
                print('doing patient scaling test')
                CPU_set = itertools.combinations_with_replacement(CPU_set, 2)
            else:
                # symmetric_cobminations
                print('doing minimal scaling test')
                CPU_set = zip(CPU_set_1, CPU_set_2)
        else:
            print('testing {}, from {:d} to {:d} cores'.format(scaling_script, np.min(CPU_set),np.max(CPU_set)))

        mesh_list = list(CPU_set)
        return mesh_list


def do_scaling_run(scaling_script, resolution, CPU_set,
                   niter=None, mesh_dim=2,
                   verbose=None, label=None,
                   OpenMPI=None, MPISGI=None, IntelMPI=None):
    if OpenMPI is None and IntelMPI is None and MPISGI is None:
        OpenMPI = True

    dim = len(resolution)
    sim_nx = resolution[0]
    sim_nz = resolution[-1]
    if dim==3:
        sim_ny = resolution[1]
        res_string = '{:d}x{:d}x{:d}'.format(sim_nx, sim_ny, sim_nz)
    else:
        res_string = '{:d}x{:d}'.format(sim_nx, sim_nz)

    # create scaling data file
    file_label = scaling_script.split('.py')[0]
    file_name = 'scaling_data_'+file_label
    if not label is None:
        file_name = file_name+'_'+label
    file_name = file_name+'.h5'
    print("writing file {}".format(file_name))
    scaling_file = h5py.File(file_name, 'a')
    if 'details' not in scaling_file:
        scaling_file['details/script'] = scaling_script
    res_group = 'data/'+res_string
    scaling_data = scaling_file.require_group(res_group)
    start_time = time.time()

    for CPUs in CPU_set:
        if mesh_dim == 2:
            ENV_N_TOTAL_CPU = np.prod(CPUs)
            cpu_group = '{:d}x{:d}'.format(CPUs[0],CPUs[1])
        else:
            ENV_N_TOTAL_CPU = CPUs
            cpu_group = '{:d}'.format(CPUs[0])

        print("scaling test of {}".format(scaling_script),
              " at {:s}".format(res_string),
              " on {:d} cores".format(ENV_N_TOTAL_CPU))

        if not cpu_group in scaling_data:
            print('testing cpu set: {:}'.format(cpu_group))
            scaling_file.close()

            test_env = dict(os.environ,
                            N_X='{:d}'.format(sim_nx),
                            N_Z='{:d}'.format(sim_nz),
                            N_TOTAL_CPU='{:d}'.format(ENV_N_TOTAL_CPU))
            if OpenMPI:
                commands = ["mpirun", "-n","{:d}".format(ENV_N_TOTAL_CPU),
                            "--bind-to", "core", "--map-by", "core"]
            elif MPISGI:
                commands = ['mpiexec_mpt', "-n","{:d}".format(ENV_N_TOTAL_CPU)]
            elif IntelMPI:
                commands = ['mpirun', "-n","{:d}".format(ENV_N_TOTAL_CPU)]
            else:
                commands = ['mpirun', "-n","{:d}".format(ENV_N_TOTAL_CPU)]

            commands += ["python3", scaling_script, "--nz={:d}".format(sim_nz), "--nx={:d}".format(sim_nx)]
            if mesh_dim == 2:
                commands.append("--mesh={:d},{:d}".format(CPUs[0], CPUs[1]))
                commands.append("--ny={:d}".format(sim_ny))
                print(" pencils/core (0): {:g}x{:g}={:g}".format(1/2*sim_nx/CPUs[0], sim_ny/CPUs[1], 1/2*sim_nx*sim_ny/(CPUs[0]*CPUs[1])))
                print(" pencils/core (2): {:g}x{:g}={:g}".format(1/2*sim_nx/CPUs[0], 3/2*sim_nz/CPUs[1], 1/2*sim_nx*3/2*sim_nz/(CPUs[0]*CPUs[1])))
                print(" pencils/core (4): {:g}x{:g}={:g}".format(3/2*sim_ny/CPUs[0], 3/2*sim_nz/CPUs[1], 3/2*sim_ny*3/2*sim_nz/(CPUs[0]*CPUs[1])))
                min_pencils_per_core = 1/2*sim_nx*sim_ny/(CPUs[0]*CPUs[1])
            else:
                print(" pencils/core: {:g} ({:g}) and {:g} ({:g})".format(1/2*sim_nx/ENV_N_TOTAL_CPU, 3/2*sim_nx/ENV_N_TOTAL_CPU,
                                                                              sim_nz/ENV_N_TOTAL_CPU, 3/2*sim_nz/ENV_N_TOTAL_CPU))
                min_pencils_per_core = min(1/2*sim_nx/ENV_N_TOTAL_CPU, sim_nz/ENV_N_TOTAL_CPU)
            if niter is not None:
                commands += ["--niter={:d}".format(niter)]

            print("command: "+" ".join(commands))
            proc = subprocess.run(commands,
                                  env=test_env,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            stdout, stderr = proc.stdout, proc.stderr

            if verbose:
                for line in stdout.splitlines():
                    print("out: {}".format(line))

                for line in stderr.splitlines():
                    print("err: {}".format(line))

            for line in stdout.splitlines():
                if line.startswith('scaling:'):
                    split_line = line.split()
                    print(split_line)
                    N_total_cpu=num(split_line[1])

                    N_x = num(split_line[2])
                    N_z = num(split_line[3])

                    startup_time = num(split_line[4])
                    wall_time = num(split_line[5])
                    wall_time_per_iter = num(split_line[6])

                    DOF_cyles_per_cpusec = num(split_line[7])

            scaling_file = h5py.File(file_name, 'a')
            scaling_data = scaling_file.require_group(res_group)

            data_set = scaling_data.create_group(cpu_group)

            data_set['N_total_cpu'] = N_total_cpu
            data_set['min_pencils_per_core'] = min_pencils_per_core

            data_set['N_z'] = N_z
            data_set['N_x'] = N_x
            data_set['sim_nx'] = sim_nx
            data_set['sim_nz'] = sim_nz
            if dim == 3:
                data_set['N_y'] = N_x
                data_set['sim_ny'] = sim_ny

            data_set['startup_time'] = startup_time
            data_set['wall_time'] = wall_time
            data_set['wall_time_per_iter'] = wall_time_per_iter
            data_set['DOF_cyles_per_cpusec'] = DOF_cyles_per_cpusec

            data_set['dim'] = dim
            if dim == 3:
                data_set['plot_label'] = r'${:d}\times{:d}\times{:d}$'.format(sim_nx, sim_ny, sim_nz)
                data_set['plot_label_short'] = r'${:d}^3$'.format(sim_nz)
            else:
                data_set['plot_label'] = r'${:d}\times{:d}$'.format(sim_nx, sim_nz)
                data_set['plot_label_short'] = r'${:d}^2$'.format(sim_nz)
            if mesh_dim == 2:
                data_set['mesh'] = [CPUs[0], CPUs[1]]
                data_set['N_x_cpu'] = CPUs[0]
                data_set['N_y_cpu'] = CPUs[1]
            else:
                data_set['mesh'] = None

        else:
            print('cpu set {:} has already been tested; skipping.'.format(cpu_group))

    scaling_file.close()

    end_time = time.time()
    print(40*'*')
    print('time to test {:s}: {:8.3g}'.format(res_string, end_time-start_time))
    print(40*'*')


def read_scaling_run(file):
    print("opening file {}".format(file))
    scaling_file = h5py.File(file, 'r')
    script_set = {}
    for res in scaling_file['data']:
        res_set = {}
        data = {}
        for cpus in scaling_file['data'][res]:
            data[cpus] = {}
            for item in scaling_file['data'][res][cpus]:
                data[cpus][item] =  scaling_file['data'][res][cpus][item][()]

            for item in next(iter(data.values())):
                res_set[item] = []
            for cpus in data:
                for item in data[cpus]:
                    res_set[item].append(data[cpus][item])

        for item in res_set:
            res_set[item] = np.array(res_set[item])
        script_set[res] = res_set

    scaling_file.close()
    return script_set


#######################
## Plotting Routines ##
#######################

def plot_scaling_run(data_set, ax_set,
                     ideal_curves = True,
                     linestyle='solid', marker='o', color='None',
                     explicit_label = True, clean_plot=False,
                     dim=None):

    sim_nx = data_set['sim_nx']
    sim_nz = data_set['sim_nz']
    N_total_cpu = data_set['N_total_cpu']
    min_pencils_per_core = data_set['min_pencils_per_core']
    N_x = data_set['N_x']
    N_z = data_set['N_z']
    if dim is None:
        if 'dim' in data_set:
            dim = int(data_set['dim'][0])
        else:
            dim = 2

    if dim==3:
        sim_ny = data_set['sim_ny']
        N_y = data_set['N_y']
        N_x_cpu = data_set['N_x_cpu']
        N_y_cpu = data_set['N_y_cpu']

    startup_time = data_set['startup_time']
    wall_time = data_set['wall_time']
    wall_time_per_iter = data_set['wall_time_per_iter']
    DOF_cyles_per_cpusec = data_set['DOF_cyles_per_cpusec']

    if dim == 2:
        resolution = [sim_nx, sim_nz]
    elif dim == 3 :
        resolution = [sim_nx, sim_ny, sim_nz]

    if color is 'None':
        color=next(ax_set[0]._get_lines.prop_cycler)['color']

    if clean_plot:
        plot_label = data_set['plot_label'][0].split('-')[0]
    else:
        plot_label = data_set['plot_label'][0]

    if explicit_label:
        label_string = plot_label
    else:
        label_string = data_set['plot_label_short'][0]

    ax_set[0].plot(N_total_cpu, wall_time, label=label_string,
                   marker=marker, linestyle=linestyle, color=color)

    ax_set[1].plot(N_total_cpu, wall_time_per_iter, label=label_string,
                   marker=marker, linestyle='none', color=color, alpha=0.5)

    ax_set[2].plot(min_pencils_per_core, DOF_cyles_per_cpusec, label=label_string,
                   marker=marker, linestyle='none', color=color, alpha=0.5)

    ax_set[3].plot(N_total_cpu, startup_time, label=label_string,
                   marker=marker,  linestyle='none', color=color)

    if ideal_curves:
        ideal_cores = np.sort(N_total_cpu)
        i_min = np.argmin(N_total_cpu)
        ideal_time = wall_time[i_min]*(N_total_cpu[i_min]/ideal_cores)
        ideal_time_per_iter = wall_time_per_iter[i_min]*(N_total_cpu[i_min]/ideal_cores)

        ax_set[0].plot(ideal_cores, ideal_time, linestyle='--', color='black', zorder=0)
        ylim_0 = min(ax_set[1].get_ylim()[0], np.min(wall_time_per_iter))
        ax_set[1].plot(ideal_cores, ideal_time_per_iter, linestyle='--', color='black', zorder=0)
        ax_set[1].set_ylim(bottom=ylim_0)
        #ax_set[1].set_ylim(emit=True)

    for i in range(4):
        ax_set[i].set_xscale('log', basex=2)
        ax_set[i].set_yscale('log')
        ax_set[i].margins(x=0.05, y=0.05)

    #i_max = N_total_cpu.argmax()
    i_max = min_pencils_per_core.argmin()
    ax_set[4].plot(N_total_cpu[i_max], DOF_cyles_per_cpusec[i_max], label=label_string +' ({:d}/core)'.format(int(min_pencils_per_core[i_max])),
                     marker=marker,  linestyle=linestyle, color=color)


def initialize_plots(num_figs, fontsize=12):
    import scipy.constants as scpconst
    fig_set = []
    ax_set = []

    x_size = 7 # width of single column in inches
    y_size = x_size/scpconst.golden

    for i in range(num_figs):
        fig = plt.figure(figsize=(x_size, y_size))
        ax = fig.add_subplot(1,1,1)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)

        fig_set.append(fig)
        ax_set.append(ax)
    return fig_set, ax_set


def legend_with_ideal(ax, loc='lower left', fontsize=8):
    handles, labels = ax.get_legend_handles_labels()
    idealArtist = plt.Line2D((0,1),(0,0), color='black', linestyle='--')
    ax.legend([handle for i,handle in enumerate(handles)]+[idealArtist],
              [label for i,label in enumerate(labels)]+['ideal'],
              loc=loc, prop={'size':fontsize})


def add_base10_axis(ax):
    #######################################################
    # from http://stackoverflow.com/questions/31803817/how-to-add-second-x-axis-at-the-bottom-of-the-first-one-in-matplotlib
    ax10 = ax.twiny()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Add some extra space for the second axis at the bottom
    #fig.subplots_adjust(bottom=0.2)

    # Move twinned axis ticks and label from top to bottom
    ax10.xaxis.set_ticks_position("bottom")
    ax10.xaxis.set_label_position("bottom")

    # Offset the twin axis below the host
    ax10.spines["bottom"].set_position(("axes", -0.15))

    # Turn on the frame for the twin axis, but then hide all
    # but the bottom spine
    ax10.set_frame_on(True)
    ax10.patch.set_visible(False)
    for sp in ax10.spines.values():
        sp.set_visible(False)
    ax10.spines["bottom"].set_visible(True)

    tick_locs = ax.xaxis.get_ticklocs()
    ax10.set_xscale('log', basex=2)
    ax10.grid()
    #ax10.grid(b=False) # suppress gridlines
    ax10.set_xticks(tick_locs)
    ax10.set_xticklabels(["{:d}".format(int(V)) for V in tick_locs])
    ax10.set_xlim(xlim)
    ax10.set_ylim(ylim)
    return ax10
    #######################################################


def finalize_plots(fig_set, ax_set):

    ax_set[0].set_xlabel('N-core')
    ax_set[0].set_ylabel('total time [s]')
    legend_with_ideal(ax_set[0], loc='lower left')
    #fig_set[0].savefig('scaling_time.pdf')

    ax10 = add_base10_axis(ax_set[1])
    ax_set[1].set_xlabel('N-core')
    ax_set[1].set_ylabel('time/iter [s]')
    legend_with_ideal(ax_set[1], loc='lower left')
    xlim = ax_set[1].get_xlim()
    ax_set[1].set_xlim(0.9*xlim[0],1.1*xlim[1])
    ylim = ax_set[1].get_ylim()
    ax_set[1].set_ylim(1/2*ylim[0],2*ylim[1])
    fig_set[1].subplots_adjust(bottom=0.2)
    fig_set[1].savefig('scaling_time_per_iter.pdf')

    #ax_set[2].set_xlabel('N-core')
    xlim = ax_set[2].get_xlim()
    ax_set[2].set_xlim(xlim[1],xlim[0])
    ax_set[2].set_xlabel('Pencils/core')
    ax_set[2].set_ylabel('DOF-cycles/cpu-sec')
    ax_set[2].legend(loc='upper right')
    ax_set[2].set_yscale('linear')
    fig_set[2].savefig('scaling_DOF.pdf')

    ax_set[3].set_xlabel('N-core')
    ax_set[3].set_ylabel('startup time [s]')
    ax_set[3].legend(loc='lower left')
    fig_set[3].savefig('scaling_startup.pdf')

    ax_set[4].set_xlabel('N-core')
    ax_set[4].set_ylabel('DOF-cycles/cpu-sec')
    ax_set[4].legend(loc='upper left')
    fig_set[4].savefig('scaling_DOF_weak.pdf')


if __name__ == "__main__":

    from docopt import docopt
    import logging
    logger = logging.getLogger(__name__)

    # Parse arguments
    args = docopt(__doc__)

    # Run
    if args['run']:
        # Resolutions
        n_z = num(args['<nz>'])
        n_x = num(args['--nx']) if args['--nx'] else n_z
        n_y = num(args['--ny']) if args['--ny'] else n_z
        if args['--3D']:
            resolution = [n_x, n_y, n_z]
            mesh_dim = 2
        else:
            resolution = [n_x, n_z]
            mesh_dim = 1
        # Core limits
        max_cores = np.int(args['--max-cores']) if args['--max-cores'] else None
        min_cores = np.int(args['--min-cores']) if args['--min-cores'] else None
        # Get CPU set
        print(40*'=')
        print("beginning scaling run with resolution: {}".format(resolution))
        mesh_list = build_mesh_list(n_z, one_pencil=args['--one-pencil'], test_type=args['--test-type'], mesh_dim=mesh_dim, max_cores=max_cores, min_cores=min_cores)
        print("final mesh list:", mesh_list)
        print(40*'=')
        # Do scaling run
        start_time = time.time()
        do_scaling_run(args['<scaling_script>'], resolution, mesh_list, niter=int(float(args['--niter'])), mesh_dim=mesh_dim, verbose=args['--verbose'], label=args['--label'], OpenMPI=args['--OpenMPI'], MPISGI=args['--MPISGI'], IntelMPI=args['--IntelMPI'])
        end_time = time.time()
        print(40*'=')
        print('scaling run finished')
        print('time to do all tests: {:f}'.format(end_time-start_time))
        print(40*'=')

    # Plot
    elif args['plot']:
        # Helpers
        import re
        def natural_sort(l, reverse=False):
            convert = lambda text: int(text) if text.isdigit() else text.lower()
            alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
            return sorted(l, key = alphanum_key, reverse=reverse)
        # Setup output
        output_path = pathlib.Path(args['--output']).absolute()
        if not output_path.exists():
            output_path.mkdir()
        # Plot
        fig_set, ax_set = initialize_plots(5)
        for file in args['<files>']:
            data_set = read_scaling_run(file)
            for res in natural_sort(data_set.keys(), reverse=True):
                print('plotting run: {:}'.format(res))
                plot_scaling_run(data_set[res], ax_set, clean_plot=args['--clean_plot'])
        finalize_plots(fig_set, ax_set)
