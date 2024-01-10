ITER=100
for N in 1 4 8 16 32 64 128
do
     mpiexec -n $N python3 rayleigh_benard_2d.py --nz=$1 --aspect=2 --niter=$ITER
     CASE=rayleigh_benard_2d_${1}_${N}
     python3 -m gprof2dot -f pstats --skew 0.5 -n 5 runtime.prof | dot -Tpng -o ${CASE}_runtime_profile.png
     mv runtime.prof ${CASE}_runtime.prof
done
