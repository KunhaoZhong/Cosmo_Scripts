#! /bin/bash
#SBATCH -N 1 #node count
#SBATCH --ntasks-per-node=5 # core count
#SBATCH -o job_%j.out #job ID
#SBATCH -p debug-28core
export OMP_NUM_THREADS=5
source /gpfs/home/kuzhong/.conda/etc/profile.d/conda.sh
conda activate cocoa 
source /gpfs/home/kuzhong/work/cocoa1/Cocoa/start_cocoa
echo "running lmax=2500" 
mpirun -n 1 --mca btl tcp,self --bind-to core --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run /gpfs/home/kuzhong/work/cocoa1/Cocoa/projects/nonlinear/2500_nonlinear.yaml -f -p /gpfs/home/kuzhong/work/cocoa1/Cocoa/external_modules
mpirun -n 1 --mca btl tcp,self --bind-to core --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run /gpfs/home/kuzhong/work/cocoa1/Cocoa/projects/nonlinear/2500_linear.yaml -f -p /gpfs/home/kuzhong/work/cocoa1/Cocoa/external_modules
echo "running lmax=2400" 
mpirun -n 1 --mca btl tcp,self --bind-to core --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run /gpfs/home/kuzhong/work/cocoa1/Cocoa/projects/nonlinear/2400_nonlinear.yaml -f -p /gpfs/home/kuzhong/work/cocoa1/Cocoa/external_modules
mpirun -n 1 --mca btl tcp,self --bind-to core --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run /gpfs/home/kuzhong/work/cocoa1/Cocoa/projects/nonlinear/2400_linear.yaml -f -p /gpfs/home/kuzhong/work/cocoa1/Cocoa/external_modules
echo "running lmax=2300" 
mpirun -n 1 --mca btl tcp,self --bind-to core --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run /gpfs/home/kuzhong/work/cocoa1/Cocoa/projects/nonlinear/2300_nonlinear.yaml -f -p /gpfs/home/kuzhong/work/cocoa1/Cocoa/external_modules
mpirun -n 1 --mca btl tcp,self --bind-to core --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run /gpfs/home/kuzhong/work/cocoa1/Cocoa/projects/nonlinear/2300_linear.yaml -f -p /gpfs/home/kuzhong/work/cocoa1/Cocoa/external_modules
echo "running lmax=2200" 
mpirun -n 1 --mca btl tcp,self --bind-to core --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run /gpfs/home/kuzhong/work/cocoa1/Cocoa/projects/nonlinear/2200_nonlinear.yaml -f -p /gpfs/home/kuzhong/work/cocoa1/Cocoa/external_modules
mpirun -n 1 --mca btl tcp,self --bind-to core --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run /gpfs/home/kuzhong/work/cocoa1/Cocoa/projects/nonlinear/2200_linear.yaml -f -p /gpfs/home/kuzhong/work/cocoa1/Cocoa/external_modules
echo "running lmax=1800" 
mpirun -n 1 --mca btl tcp,self --bind-to core --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run /gpfs/home/kuzhong/work/cocoa1/Cocoa/projects/nonlinear/1800_nonlinear.yaml -f -p /gpfs/home/kuzhong/work/cocoa1/Cocoa/external_modules
mpirun -n 1 --mca btl tcp,self --bind-to core --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run /gpfs/home/kuzhong/work/cocoa1/Cocoa/projects/nonlinear/1800_linear.yaml -f -p /gpfs/home/kuzhong/work/cocoa1/Cocoa/external_modules