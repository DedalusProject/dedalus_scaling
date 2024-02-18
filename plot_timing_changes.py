"""
Tool for ploting profiling results from Dedalus script runs.

Usage:
    plot_timing_changes.py <directory>... [options]

Options:
    --profile=<profile>    Profile data to plot (e.g., runtime, setup, warmup) [default: runtime]

    --thresh=<thresh>      Threshold for trimming output, as a fraction of total time [default: 1e-4]

    --max_profiles=<max>   Maximum number of profiles to output

    --label=<label>        Optional label to add to output figures

    --subtimings           Produce subtiming outputs

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
cmap = mpl.colormaps['tab10']
cmap_sub = mpl.colormaps['tab20c']
debug = False


def sort_by_total(joined_stat):
    return sorted(joined_stat.items(), key=lambda kv: np.sum(kv[1]), reverse=True)

def test_criteria(data):
    return np.max(data)

def clean_display(ax):
    # from http://nbviewer.ipython.org/gist/anonymous/5357268
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

group = {'linear algebra':["gssv", "apply_sparse", "superlu", "linalg"],
              'MPI':["mpi4py.MPI", "fftw.fftw_wrappers.Transpose", "fftw_RL_to_CL", "fftw_CL_to_RL", "localize_columns", "localize_rows", "RL_fftw", "CL_fftw"],
              'FFT':["ifft", "_dct", "rfft", "unpack_rescale", 'repack_rescale', "forward", "backward"],
              'arithmetic':["operate", "einsum", "arithmetic"],
              'copy':["copyto", "gather_inputs", "gather_outputs", 'scatter_inputs', "scatter_outputs"],
              'exclude': ["load_dynamic", "__init__", "<frozen", 'importlib']}

def identify_categories(stats_pdf_dict,
                              N_profiles=np.inf,
                              thresh=0.02, verbose=False,
                              dir=pathlib.Path('.')):

    sorted_list = sort_by_total(stats_pdf_dict)

    total_time = []
    for item in tottime.items():
        total_time.append(np.mean(item[1]))
    total_time = np.sum(total_time)

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
                if verbose and debug:
                    print(f"found {key:s} call: {func[2]} in {tests} at {i_sort:d}")
                group_data[key][func] = data
                found_category = True
        if not found_category and test_criteria(data)/total_time > thresh:
             group_data['other'][func] = data
        N_cores = max(N_cores, data.size)
    N_profiles = 0
    for key in group_data:
        N_profiles += len(group_data[key])

    if verbose and debug:
        for func in group_data['exclude']:
            print(f"found excluded call: {func[2]}, popping...")
    excluded = group_data.pop('exclude', None)
    if verbose and debug: print(excluded)

    timings = {key:{} for key in group_data}
    subtimings = {key:{} for key in group_data}
    for i_group, key in enumerate(group_data):
        group_time = np.zeros(N_cores)
        first_item = True
        for func, data_list in group_data[key].items():
            data = np.array(data_list)
            N_missing = N_cores - data.size

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

            timing_data_string = "{:8.2g} |{:8.2g} |{:8.2g}".format(np.min(data), np.mean(data), np.max(data))

            if verbose:
                if first_item:
                    print(f'{key:>60s} :')
                    first_item = False
                print("{:60s} = {:s}".format(title_string, timing_data_string))

            subtimings[key][func] = {}
            subtimings[key][func]['min']=np.min(data)
            subtimings[key][func]['max']=np.max(data)
            subtimings[key][func]['mean']=np.mean(data)

        timings[key]['min']=np.min(group_time)
        timings[key]['max']=np.max(group_time)
        timings[key]['mean']=np.mean(group_time)
    return timings, subtimings, N_cores

def read_database(file):
    with (open(file, "rb")) as f:
        data = pickle.load(f)

    primcalls = data['primcalls']
    totcalls = data['totcalls']
    tottime = data['tottime']
    cumtime = data['cumtime']

    return primcalls, totcalls, tottime, cumtime


if __name__ == "__main__":
    from docopt import docopt
    args = docopt(__doc__)

    label = args['--label']
    if label:
        label = '_'+str(label)
    else:
        label = ''
    verbose = args['--verbose']
    N_profiles = args['--max_profiles']
    if N_profiles:
        N_profiles = int(N_profiles)
    else:
        N_profiles = np.inf

    mean_time = {}
    mean_sub_time = {}
    n_cores = []
    total_times = []

    for directory in args['<directory>']:
        if verbose:
            print(f'opening {directory:s}')
        dir = pathlib.Path(directory)
        profiles_file = str(args['--profile'])+'_parallel.pickle'
        primcalls, totcalls, tottime, cumtime = read_database(dir / profiles_file)

        timings, subtimings, N_cores = identify_categories(tottime, thresh=float(args['--thresh']), N_profiles=N_profiles, verbose=verbose)

        total_time=0
        n_cores.append(N_cores)
        for key in timings:
            if key not in mean_time:
                mean_time[key]=[]
            mean_time[key].append(timings[key]['mean'])
            total_time += timings[key]['mean']
            if key not in mean_sub_time:
                mean_sub_time[key]={}
            for sub_key in subtimings[key]:
                if sub_key not in mean_sub_time[key]:
                    mean_sub_time[key][sub_key] = []
                mean_sub_time[key][sub_key].append(subtimings[key][sub_key]['mean'])
        total_times.append(total_time)

    i_sort = np.argsort(np.array(n_cores))

    n_cores = np.array(n_cores)[i_sort]
    total_times = np.array(total_times)[i_sort]
    N_cases = i_sort.size
    for key in mean_time:
        mean_time[key] = np.array(mean_time[key])[i_sort]
    for key in mean_sub_time:
        short_keys = []
        for sub_key in mean_sub_time[key]:
            if len(mean_sub_time[key][sub_key]) == N_cases:
                mean_sub_time[key][sub_key] = np.array(mean_sub_time[key][sub_key])[i_sort]
            else:
                short_keys.append(sub_key)
        for sub_key in short_keys:
            mean_sub_time[key].pop(sub_key, None)
    fig, ax = plt.subplots()
    previous_data = np.zeros(len(n_cores))
    for i, key in enumerate(mean_time):
        data = mean_time[key]
        ax.fill_between(n_cores, data+previous_data, y2=previous_data, label=key, color=cmap(i),step='mid')
        previous_data += data
    ax.set_xlabel('N cores')
    ax.set_ylabel('time [sec]')
    ax.set_xscale('log', base=2)
    ax.set_ylim(0,np.max(previous_data))
    ax.set_xlim(np.min(n_cores),np.max(n_cores))
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'total_group_time{label:s}.png', dpi=300)

    fig, ax = plt.subplots()
    previous_data = np.zeros(len(n_cores))
    for i, key in enumerate(mean_time):
        data = mean_time[key]/total_times
        ax.fill_between(n_cores, data+previous_data, y2=previous_data, label=key, color=cmap(i),step='mid')
        previous_data += data
    ax.set_xscale('log', base=2)
    ax.set_ylim(0,1)
    ax.set_xlim(np.min(n_cores),np.max(n_cores))
    ax.set_xlabel('N cores')
    ax.set_ylabel('%time')
    ax.legend(loc='center right', fontsize='small', framealpha=0.7)
    fig.tight_layout()
    fig.savefig(f'percent_group_time{label:s}.png', dpi=300)

    print(total_times)
    if args['--subtimings']:
        for key in subtimings:
            fig, ax = plt.subplots()
            previous_data = np.zeros(len(n_cores))
            print(key, mean_time[key]/total_times)
            for i, sub_key in enumerate(mean_sub_time[key]):
                data = mean_sub_time[key][sub_key]/total_times
                print(sub_key, data)
                ax.fill_between(n_cores, data+previous_data, y2=previous_data, label=sub_key, color=cmap_sub(i),step='mid')
                previous_data += data
            ax.set_xscale('log', base=2)
            ax.set_ylim(0,np.max(previous_data))
            ax.set_xlim(np.min(n_cores),np.max(n_cores))
            ax.set_xlabel('N cores')
            ax.set_ylabel(f'{key:s} %time')
            ax.legend(loc='center right', fontsize='small', framealpha=0.7)
            fig.tight_layout()
            key_label = key.replace(' ','_')
            fig.savefig(f'percent_subgroup_{key_label:s}_time{label:s}.png', dpi=300)
