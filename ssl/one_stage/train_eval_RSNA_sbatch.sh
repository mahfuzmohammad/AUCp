#!/bin/bash
#SBATCH -N 1            # number of nodes
#SBATCH -c 8          # number of cores, adjust if needed to accommodate 5 parallel processes
#SBATCH --mem=50G       # amount of memory
#SBATCH -G a100:1       # GPU
#SBATCH -t 48:00:00     # time in d-hh:mm:ss
#SBATCH -p general      # partition
#SBATCH -q public       # QOS
#SBATCH -o sbatch_scripts/logs/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e sbatch_scripts/logs/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

# Load required modules for job's environment (uncomment and adjust for your cluster).
# module load mamba/latest
# source activate rafenv

# Run from the directory containing this script so relative imports resolve.
cd "$(dirname "$0")"

num_repeat=1
datasets="brats"
settings="CutPaste FPI FPI-Poisson Shift-Intensity-M"
gpu_id=5

for data in $datasets;do
  for setting in $settings;do
    for((i=0;i<num_repeat;i=i+1));do
      python AUCP_test.py -d "$data" -s "$setting" -g $gpu_id -f "$i";
    done
  done
done
