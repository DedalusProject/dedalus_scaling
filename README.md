# README #

Incompressible Navier-Stokes problem with Taylor-Green forcing using
the [Dedalus](http://dedalus-project.org) pseudospectral
framework.  To run these problems, first install
[Dedalus](http://dedalus-project.org/) (and on
[bitbucket](https://bitbucket.org/dedalus-project/dedalus)). 

Once [Dedalus](http://dedalus-project.org/) is installed and activated, do the following:
```
#!bash
mpiexec_mpt -n 256 python3 incompressible_NS_TG.py --mesh=16,16
```
To obtain sparsity plots of the implicit matrices, use the `--verbose`
keyword: 
```
#!bash
python3 incompressible_NS_TG.py --verbose --niter=2 --nx=4 --ny=4 --nz=128
```
Here we've run a reduced horizontal resolution case so that it can be
readily completed by a single core.

For scaling tests on NASA/Pleiades:
```
#!bash
python3 scaling.py run incompressible_NS_TG.py --3D 128 --min-cores=128 --max-cores=512 --MPISGI
```

The following block of code, run on NASA/Pleiades Broadwell nodes (28
cores/node) using an MPI-SGI stack,
```
#!bash
python3 scaling.py run incompressible_NS_TG.py --3D 128 --min-cores=16 --max-cores=2048 --MPISGI --niter=10
python3 scaling.py run incompressible_NS_TG.py --3D 256 --min-cores=64 --max-cores=2048 --MPISGI --niter=10
python3 scaling.py run incompressible_NS_TG.py --3D 512 --min-cores=256 --max-cores=2048 --MPISGI --niter=10

python3 scaling.py plot scaling_data_512x512x512.db scaling_data_256x256x256.db scaling_data_128x128x128.db 
```
recreates this scaling plot:


Ideal scaling is indicated by dashed black lines, extrapolated from
the lowest core count case in all cases.  At each core count, a cloud
of points are shown sampling a variety of processor mesh
configurations.  Generally, the configuration of the processor mesh
has little impact on overall performance.

Contact the exoweather team for more details.

