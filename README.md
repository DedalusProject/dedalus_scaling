# README #

Scaling and profiling tools, with associated example problems, for
the [Dedalus](http://dedalus-project.org) pseudospectral
framework.  To run these problems, first install
[Dedalus](http://dedalus-project.org/) (and on
[github](https://github.com/DedalusProject/dedalus)).

## Example problems
Once [Dedalus](http://dedalus-project.org/) is installed and activated, run the triply-periodic shear flow example with:
```bash
mpirun -n 256 python3 shear_flow_3d.py
```

The Rayleigh-Benard convection examples can be run using:
```bash
mpirun -n 256 python3 rayleigh_benard_3d.py
```

## Profiling
If you would like to have detailed profiles of the cython routines, please make sure to set the following envinroment variable:
```bash
export CYTHON_PROFILE=True
```
before installing [Dedalus](http://dedalus-project.org/).

Detailed profiling requires installation of the `gprof2dot` library:
```bash
pip install gprof2dot
```
and the availability of `dot` in the compute environment.

Profiling during runs is controlled in `dedalus.cfg` or at the script level.  When enabled, code profiles are stored by default in the `./profiles` directory.  You can obtain detailed performance data aggregated across cores and on a per-core basis using:
```bash
python3 plot_profiles.py
```
which will produce a variety of graphical outputs, stored in `./profiles` as `.png` files.
