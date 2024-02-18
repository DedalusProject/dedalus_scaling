ITER=100
for N in 64 128 256 512 1024
do
     mpiexec -n $N python3 shear_flow_3d.py --nz=$1 --niter=$ITER
     CASE=shear_flow_3d_${1}x${1}x${1}_${N}
     mv profiles ${CASE}
done
