#!/bin/bash                                                      
#SBATCH --partition=Orion                                  # Choose a Slurm partition
#SBATCH --job-name=Cu-Fast-Test                            # Name the job for tracking
#SBATCH --nodes=1                                          # 1 node
#SBATCH --ntasks-per-node=8                               # 32 MPI tasks total
#SBATCH --time=00:30:00                                    # 30-minute limit (test finishes in < 2 mins)
#SBATCH --mem-per-cpu=2GB                                  # Memory allocation per core
#SBATCH --mail-user=ilittle8@charlotte.edu                  # Email notification address
#SBATCH --mail-type=BEGIN,FAIL,END                         # Email triggers

export WORK_DIR=$SLURM_SUBMIT_DIR
cd "$SLURM_SUBMIT_DIR"

# Load LAMMPS MPI module
module load lammps/02Aug23-mpi

echo "Starting Fast Multi-Length NEMD Test..."

# Run fast Python test script
python3 run_cu_mfp_test_fast.py

echo "======================================================"
echo "Finish Time : $(date)"
echo "Job ID/Name : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "Test Run Complete."
echo "======================================================"