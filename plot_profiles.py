"""
Tool for ploting profiling results from Dedalus script runs.

Usage:
    plot_profiles.py [options]

Options:
    --profile=<profile>    Profile data to plot (e.g., runtime, setup, warmup) [default: runtime]

    --thresh=<thresh>      Theshold for trimming output, as a fraction of total time [default: 0.02]

    --max_profiles=<max>   Maximum number of profiles to output [default: 50]

    --directory=<dir>      Location of profile data [default: profiles]

    --verbose              Display text verbose output to screen

"""
import os
import pstats
import numpy as np
import pathlib
import pickle

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



def make_graph(profile, output_png_file, node_thresh=0.5):

    import subprocess

    proc_graph = subprocess.Popen(["gprof2dot", "--skew", "0.5", "-n", "{:f}".format(node_thresh),
                                   "-f", "pstats", profile],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


    # the directed graph is produced by proc_graph.stdout
    proc_dot = subprocess.Popen(["dot", "-Tpng", "-o", output_png_file],
                                stdin = proc_graph.stdout,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    stdout, stderr = proc_dot.communicate()


def sort_dict(dict_to_sort):
    sorted_list = sorted(dict_to_sort.items(), key=lambda data_i: test_criteria(data_i[1]), reverse=True)
    return sorted_list

def sort_by_total(joined_stat):
    return sorted(joined_stat.items(), key=lambda kv: np.sum(kv[1]), reverse=True)

def test_criteria(data):
    return np.max(data)

def natural_sort(l):
    import re

    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(l, key=alphanum_key)

def clean_display(ax):
    # from http://nbviewer.ipython.org/gist/anonymous/5357268
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

def plot_per_core_performance(stats_pdf_dict,
                              label='', N_profiles=50,
                              thresh=0.02, verbose=False,
                              dir=pathlib.Path('.')):

    cmap = mpl.colormaps['tab20']
    cmap_group = mpl.colormaps['tab10']
    #set_plot_defaults(rcParams)

    sorted_list = sort_by_total(stats_pdf_dict)
    composite_data_set = []
    composite_label = []
    composite_key_label = []

    total_time = []
    for item in tottime.items():
        total_time.append(np.mean(item[1]))
    total_time = np.sum(total_time)
    print(total_time)

    fig_stacked = plt.figure(figsize=[8,8/2])
    ax_stacked = fig_stacked.add_subplot(1,1,1)

    fig_group = plt.figure(figsize=[8,8/2])
    ax_group = fig_group.add_subplot(1,1,1)

    group = {'linear algebra':["gssv", "apply_sparse", "superlu"],
                  'MPI':["mpi4py.MPI", "fftw.fftw_wrappers.Transpose"],
                  'FFT':["ifft", "_dct", "rfft", "unpack_rescale", 'repack_rescale', "forward", "backward"],
                  'arithmetic':["(operate)", "einsum", "arithmetic"],
                  'copy':["copyto", "gather_inputs", "gather_outputs", 'scatter_inputs', "scatter_outputs"],
                  'exclude': ["load_dynamic", "__init__", "<frozen", 'importlib']}

    group_data = {key:{} for key in group}
    group_data['other'] = {}

    N_cores = 0

    for i_sort, (func, data) in enumerate(sorted_list):
        if i_sort+1 == N_profiles:
            break
        #print(i_sort, func, data)
        found_category = False
        data = np.array(data)
        for key in group:
            tests = [item.lower() for item in group[key]]
            if (any(item.lower() in func[0].lower() for item in group[key]) or any(item.lower() in func[2].lower() for item in group[key])) and test_criteria(data)/total_time > thresh:
                if verbose:
                    print(f"found {key:s} call: {func[2]} in {tests} at {i_sort:d}")
                group_data[key][func] = data
                found_category = True
        if not found_category and test_criteria(data)/total_time > thresh:
             group_data['other'][func] = data
        N_cores = max(N_cores, data.size)

    for key in group_data:
        print(key, ':', group_data[key].keys())

    if verbose:
        for func in group_data['exclude']:
            print(f"found excluded call: {func[2]}, popping...")
    excluded = group_data.pop('exclude', None)
    if verbose: print(excluded)

    routine_text = "top {:d} routines for {:s}".format(N_profiles, label)
    if verbose:
        print()
        print("{:80s}".format(routine_text),"     min      mean       max   (mean%total)   (m%t cum.)")
        print(120*"-")

    def percent_time(sub_time):
        sub_string = "{:4.2f}%".format(100*sub_time/total_time)
        return sub_string

    if N_cores > 200:
        N_bins = 100
        logscale = True
    else:
        N_bins = int(N_cores/4)
        if N_bins == 0 : N_bins = 1
        logscale = False
    i_fig = 0
    running = 0
    previous_group_data = np.zeros(N_cores)
    previous_data = np.zeros(N_cores)

    for i_group, key in enumerate(group_data):
        group_time = np.zeros(N_cores)
        for func, data_list in group_data[key].items():
            data = np.array(data_list)
            N_missing = previous_data.size - data.size

            if N_missing != 0:
                if verbose:
                    print("missing {:d} values; setting to zero".format(N_missing))
                for i in range(N_missing):
                    data_list.insert(N_missing*(i+1)-1, 0)
                data = np.array(data_list)

            group_time += data

            if func[0] == '~':
                title_string = func[2]
            else:
                title_string = "{:s}:{:d}:{:s}".format(*func)

            running += np.mean(data)
            timing_data_string = "{:8.2g} |{:8.2g} |{:8.2g}  ({:s}) ({:s})".format(np.min(data), np.mean(data), np.max(data), percent_time(np.mean(data)), percent_time(running))

            if verbose:
                print("{:80s} = {:s}".format(title_string, timing_data_string))

            timing_data_string = "min {:s} | {:s} | {:s} max".format(percent_time(np.min(data)), percent_time(np.mean(data)), percent_time(np.max(data)))

            title_string += "\n{:s}".format(timing_data_string)

            key_label = "{:s} {:s}".format(percent_time(np.mean(data)),func[2])
            short_label = "{:s}".format(percent_time(np.mean(data)))

            composite_data_set.append([data])
            composite_label.append(short_label)
            composite_key_label.append(key_label)

            q_color = cmap(i_fig) #next(ax_stacked._get_lines.prop_cycler)['color']

            fig = plt.figure(figsize=[8,8/2])

            # pdf plot over many cores
            ax1 = fig.add_subplot(1,2,1)

            #hist_values, bin_edges = np.histogram(data, bins=N_bins)
            #ax1.barh(hist_values, bin_edges[1:])
            ax1.hist(data, bins=N_bins, orientation='horizontal', log=logscale, linewidth=0, color=q_color)
            ax1.set_xlabel("N cores/bin")
            ax1.set_ylabel("time (sec)")
            ax1.grid(axis = 'x', color ='white', linestyle='-')


            # bar plot for each core
            ax2 = fig.add_subplot(1,2,2)
            ax2.bar(np.arange(N_cores)+1, data, linewidth=0, width=1, color=q_color)
            ax2.set_xlim(0.5, N_cores+0.5)
            ax2.set_xlabel("core #")
            clean_display(ax2)

            ax2.grid(axis = 'y', color ='white', linestyle='-')

            # end include

            ax1.set_ylim(0, 1.1*np.max(data))
            ax2.set_ylim(0, 1.1*np.max(data))

            fig.suptitle(title_string)
            fig.tight_layout()
            fig.savefig(dir / f'{label:s}_{i_fig+1:06d}.png', dpi=200)
            plt.close(fig)

            ax_stacked.bar(np.arange(N_cores)+1, data, bottom=previous_data, label=short_label, linewidth=0,
                           width=1, color=q_color)
            previous_data += data
            i_fig += 1
        ax_group.bar(np.arange(N_cores)+1, group_time, bottom=previous_group_data, label=key, linewidth=0,
                       width=1, color=cmap_group(i_group))
        previous_group_data += group_time

    clean_display(ax_stacked)
    ax_stacked.set_xlim(0.5, N_cores+0.5)
    ax_stacked.set_xlabel('core #')
    ax_stacked.set_ylabel('total time (sec)')
    #ax_stacked.legend(loc='upper left', bbox_to_anchor=(1.,1.), fontsize=10)
    ax_stacked.set_title("per core timings for routines above {:g}% total time".format(thresh*100))
    ax_stacked.grid(axis = 'y', color ='white', linestyle='-')
    #ax_stacked.set_aspect(N_data/total_time)
    points_per_data = 10
    fig_x_size = 8

    fig_stacked.tight_layout()
    fig_stacked.savefig(dir / f"{label:s}_per_core_timings.png", dpi=max(200, N_cores*points_per_data/fig_x_size))
    plt.close(fig_stacked)

    clean_display(ax_group)
    ax_group.legend()
    ax_group.set_xlim(0.5, N_cores+0.5)
    ax_group.set_xlabel('core #')
    ax_group.set_ylabel('total time (sec)')
    #ax_stacked.legend(loc='upper left', bbox_to_anchor=(1.,1.), fontsize=10)
    ax_group.set_title("per core timings for groups, routines above {:g}% total time".format(thresh*100))
    ax_group.grid(axis = 'y', color ='white', linestyle='-')
    fig_group.tight_layout()
    fig_group.savefig(dir / f"{label:s}_group_per_core_timings.png", dpi=max(200, N_cores*points_per_data/fig_x_size))
    plt.close(fig_group)


    # pdf plot over many cores
    fig_composite = plt.figure(figsize=[8,8/2])
    ax_composite = fig_composite.add_subplot(1,1,1)

    # print(composite_data_set)
    # n, bins, patches = ax_composite.hist(composite_data_set, bins=N_bins, orientation='vertical', log=logscale, linewidth=0, stacked=True,
    #                                      label=composite_label)
    #
    # clean_display(ax_composite)
    # ax_composite.grid(axis = 'y', color ='white', linestyle='-')
    #
    # ax_composite.set_ylabel("N cores/bin")
    # ax_composite.set_xlabel("total time (sec)")
    # ax_composite.set_ylim(0, 1.1*np.max(composite_data_set))
    # ax_composite.legend(loc='upper left', bbox_to_anchor=(1.,1.), fontsize=10)
    #
    # fig_composite.suptitle("composite PDF for routines above {:g}% total time".format(thresh*100))
    # fig_composite.savefig(label+'_composite.png', dpi=200)
    # plt.close(fig_composite)
    #
    # fig_key = plt.figure()
    # plt.figlegend(patches, composite_key_label, 'center')
    # #ax_key.legend(loc='center')
    # fig_key.savefig(label+"_composite_key.png")
    # plt.close(fig_key)

def read_database(file):
    with (open(file, "rb")) as f:
        data = pickle.load(f)

    primcalls = data['primcalls']
    totcalls = data['totcalls']
    tottime = data['tottime']
    cumtime = data['cumtime']

    #average_runtime = shelf['average_runtime']
    #n_processes = shelf['n_processes']

    return primcalls, totcalls, tottime, cumtime


if __name__ == "__main__":
    from docopt import docopt
    args = docopt(__doc__)

    dir = pathlib.Path(args['--directory'])

    joint_file = str(args['--profile'])+'.prof'
    profiles_file = str(args['--profile'])+'_parallel.pickle'

    summed_stats = pstats.Stats(str(dir / joint_file))

    primcalls, totcalls, tottime, cumtime = read_database(dir / profiles_file)

    # per-core plots
    plot_per_core_performance(tottime, label="tt", thresh=float(args['--thresh']), verbose=args['--verbose'], N_profiles=int(float(args['--max_profiles'])), dir=dir)
    # Graphs
