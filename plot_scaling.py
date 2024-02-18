"""
Basic plotting of scaling data using speed metric.  Extracts from text files that have captured standard out from a dedalus script that ends with a successful call to log_stats().

Usage:
     plot_scaling.py <files>...
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import re

from docopt import docopt
# Parse arguments
args = docopt(__doc__)

data = {}
for file in args['<files>']:
    label = file.split('.')[0].split('_')[-1]
    data[label] = {'cores' : [], 'speed' : []}
    with open(file, 'r') as f:
        for line in f:
            if "Speed" in line:
                report = line.split('solvers')[1]
                rr = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", report)
                print(rr)
                data[label]['cores'].append(int(rr[1]))
                data[label]['speed'].append(float(rr[2]))

data = dict(sorted(data.items()))

fig, ax = plt.subplots(figsize=[6,6/1.6])
for label in data:
    ax.scatter(data[label]['cores'], data[label]['speed'], label=label, alpha=0.7)
ax.legend()
ax.set_xlabel('N core')
ax.set_ylabel('speed = mode-stages/core-s')
ax.set_xscale('log')
ax.set_yscale('log')
fig.tight_layout()
fig.savefig('speed.png', dpi=300)

fig, ax = plt.subplots(figsize=[6,6/1.6])
for label in data:
    ax.scatter(data[label]['cores'], np.array(data[label]['speed'])*np.array(data[label]['cores']), label=label, alpha=0.7)
ax.legend()
ax.set_xlabel('N core')
ax.set_ylabel('speed*N core = mode-stages/s')
ax.set_xscale('log')
ax.set_yscale('log')
fig.tight_layout()
fig.savefig('work.png', dpi=300)
