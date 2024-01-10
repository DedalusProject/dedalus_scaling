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
    ncore = int(''.join(filter(str.isdigit, file)))
    data[ncore] = {'cores' : [], 'speed' : []}
    with open(file, 'r') as f:
        for line in f:
            if "Speed" in line:
                report = line.split('solvers')[1]
                rr = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", report)
                print(rr)
                data[ncore]['cores'].append(int(rr[1]))
                data[ncore]['speed'].append(float(rr[2]))

data = dict(sorted(data.items()))

fig, ax = plt.subplots(figsize=[6,6/1.6])
for ncore in data:
    ax.scatter(data[ncore]['cores'], data[ncore]['speed'], label=ncore, alpha=0.7)
ax.legend()
ax.set_xlabel('N core')
ax.set_ylabel('speed')
ax.set_xscale('log')
ax.set_yscale('log')
fig.tight_layout()
fig.savefig('speed.png', dpi=300)
