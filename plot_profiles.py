"""
Tool for ploting profiling results from Dedalus script runs.

Usage:
    plot_profiles.py [options]

Options:
    --profile=<profile>    Profile data to plot (e.g., runtime, setup, warmup) [default: runtime]
    --thresh=<thresh>      Theshold for trimming output, as a fraction of total time [default: 0.02]

    --verbose              Display text verbose output to screen

"""
import os
import shelve
import pstats
import numpy as np

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
                              thresh=0.02, verbose=False):

    cmap = mpl.colormaps['tab20']
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

    group = {'linear algebra':["gssv", "apply_sparse", "superlu"],
             'MPI':["mpi4py.MPI", "fftw.fftw_wrappers.Transpose"],
             'FFT':["ifft", "_dct", "rfft", "unpack_rescale", "repack_rescale", "forward", "backward"],
             'arithmetic':["(operate)", "einsum"],
             'copy':["copyto", "gather_inputs", "gather_outputs", "scatter_inputs", "scatter_outputs"],
             'exclude': ["load_dynamic", "__init__", "<frozen", "importlib"]}

    group_indices = {}
    for key in group:
        group_indices[key] = []
    group_indices['other'] = []

    for i_sort, (func, data_list) in enumerate(sorted_list):
        if i_sort+1 == N_profiles:
            break

        found_category = False
        for key in group:
            data = np.array(data_list)
            if any(item.lower() in func[2].lower() for item in group[key]) and test_criteria(data)/total_time > thresh:
                if verbose:
                    print(f"found {key:s} call: {func[2]} at {i_sort:d}")
                group_indices[key].append(i_sort)
                found_category = True
        if not found_category:
            data = np.array(data_list)
            if test_criteria(data)/total_time > thresh :
                group_indices['other'].append(i_sort)

    print(group_indices)
    # bubble sparse solve to the top
    last_insert = 0
    for key in group_indices:
        for i_resort in group_indices[key]:
            sorted_list.insert(last_insert,sorted_list.pop(i_resort))
            if verbose:
                print("moved entry {:d}->{:d}".format(i_resort, last_insert))
            last_insert += 1

    for i_sort, (func, data_list) in enumerate(sorted_list):
        if i_sort+1 == N_profiles:
            break
        if any((exclude_type in func[0] or exclude_type in func[2]) for exclude_type in group['exclude']):
            if verbose:
                print("found excluded call:",func[2], " at ", i_sort, " ... popping.")
            sorted_list.pop(i_sort)


    routine_text = "top {:d} routines for {:s}".format(N_profiles, label)
    if verbose:
        print()
        print("{:80s}".format(routine_text),"     min      mean       max   (mean%total)   (m%t cum.)")
        print(120*"-")
    running=0
    for i_fig, (func, data_list) in enumerate(sorted_list):
        data = np.array(data_list)
        N_data = data.shape[0]
        # if i_fig == N_profiles:
        #     break
        # if test_criteria(data)/total_time < thresh:
        #     break
        if i_fig+1 == N_profiles or (i_fig > last_insert and test_criteria(data)/total_time < thresh):
            break

        if i_fig == 0:
            previous_data = np.zeros_like(data)

        N_missing = previous_data.size - data.size

        if N_missing != 0:
            if verbose:
                print("missing {:d} values; setting to zero".format(N_missing))
            for i in range(N_missing):
                data_list.insert(N_missing*(i+1)-1, 0)
            data = np.array(data_list)
            N_data = data.shape[0]

        if func[0] == '~':
            title_string = func[2]
        else:
            title_string = "{:s}:{:d}:{:s}".format(*func)

        def percent_time(sub_time):
            sub_string = "{:4.2f}%".format(100*sub_time/total_time)
            return sub_string

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


        if N_data > 200:
            N_bins = 100
            logscale = True
        else:
            N_bins = int(N_data/4)
            if N_bins == 0 : N_bins = 1
            logscale = False

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
        ax2.bar(np.arange(N_data), data, linewidth=0, width=1, color=q_color)
        ax2.set_xlim(-0.5, N_data+0.5)
        ax2.set_xlabel("core #")
        clean_display(ax2)

        ax2.grid(axis = 'y', color ='white', linestyle='-')

        # end include

        ax1.set_ylim(0, 1.1*np.max(data))
        ax2.set_ylim(0, 1.1*np.max(data))


        fig.suptitle(title_string)
        fig.tight_layout()
        fig.savefig(label+'_{:06d}.png'.format(i_fig+1), dpi=200)
        plt.close(fig)

        ax_stacked.bar(np.arange(N_data), data, bottom=previous_data, label=short_label, linewidth=0,
                       width=1, color=q_color)
        previous_data += data

    clean_display(ax_stacked)
    ax_stacked.set_xlim(-0.5, N_data+0.5)
    ax_stacked.set_xlabel('core #')
    ax_stacked.set_ylabel('total time (sec)')
    #ax_stacked.legend(loc='upper left', bbox_to_anchor=(1.,1.), fontsize=10)
    ax_stacked.set_title("per core timings for routines above {:g}% total time".format(thresh*100))
    ax_stacked.grid(axis = 'y', color ='white', linestyle='-')
    #ax_stacked.set_aspect(N_data/total_time)
    points_per_data = 10
    fig_x_size = 8

    fig_stacked.tight_layout()
    fig_stacked.savefig(label+"_per_core_timings.png", dpi=max(200, N_data*points_per_data/fig_x_size))
    plt.close(fig_stacked)


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
    from contextlib import closing

    with closing(shelve.open(file, flag='r')) as shelf:
        primcalls = shelf['primcalls']
        totcalls = shelf['totcalls']
        tottime = shelf['tottime']
        cumtime = shelf['cumtime']

    #average_runtime = shelf['average_runtime']
    #n_processes = shelf['n_processes']

    return primcalls, totcalls, tottime, cumtime


if __name__ == "__main__":
    from docopt import docopt
    args = docopt(__doc__)

    joint_file = args['--profile']+'.prof'
    profiles_file = args['--profile']+'_profiles'

    summed_stats = pstats.Stats(joint_file)

    primcalls, totcalls, tottime, cumtime = read_database(profiles_file)

    # per-core plots
    plot_per_core_performance(tottime, label="tt", thresh=float(args['--thresh']), verbose=args['--verbose'])
    # Graphs
