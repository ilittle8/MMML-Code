#! /bin/bash
# Isaiah little's submit script template 
#SBATCH --job-name=MyJobName #Replace with your job name
#SBATCH --parition=Orion 
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=2GB
#SBATCH --nodes =1
#SBATCH --ntasks-per-node=16 #This is requesting 16 CPU cores on one node. Adjust as needed.
#SBATCH --mail-user=ilittle8@charlotte.edu
#SBATCH --mail-type=BEGIN,FAIL,END

# You can use the follow lines to know what "Features" are available on the cluster
# $ sinfo -p Orion -o "%20N %10c %10m %50f"

#Slum supports parallel processing using MPI. If your job can be parallelized, you can use the following command to run your program across multiple CPU cores:
# $ module load openmpi/4.1.0
# $ mpircc myprogram.c #You will want to find the LMMAPS equivalent of this command to compile your code with MPI support.

#When submitting a job to run in parallel, make sure you request more than 1 node and specify the number of tasks per node. 

export WORK_DIR=$SLURM_SUBMIT_DIR
cd $WORK_DIR

# Load modules
module load lammps/02Aug23-mpi

# Define LAMMPS variable
LAMMPS=lmp_urc
INPUT=in.Model.extension

# Run the simulation
echo "Starting the Molecular Dynamics Simulation..."
# $SLURM_NTASKS is automatically calculated by Slurm (nodes * ntasks-per-node)
mpirun -np $SLURM_NTASKS $LAMMPS -in $INPUT

echo "Job finished at : $(date)"