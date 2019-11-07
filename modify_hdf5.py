"""
Modify aspects of an HDF5 file.

Usage:
    modify_hdf5 <file> [options]

Options:
    --remove_group=<remove_group>      Remove a group from an HDF5 file.
"""

import h5py


if __name__ == "__main__":
    from docopt import docopt
    args = docopt(__doc__)
    filename = args['<file>']

    with h5py.File(filename, 'a') as f:
        if args['--remove_group']:
            del f[args['--remove_group']]
            
