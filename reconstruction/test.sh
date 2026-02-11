#!/bin/bash
#SBATCH -N 1            # number of nodes
#SBATCH -c 8          # number of cores, adjust if needed to accommodate 5 parallel processes
#SBATCH --mem=30G       # amount of memory
#SBATCH -G a100:1       # GPU
#SBATCH -t 24:00:00     # time in d-hh:mm:ss
#SBATCH -p general      # partition
#SBATCH -q public       # QOS
#SBATCH -o sbatch_scripts/logs/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e sbatch_scripts/logs/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

# Load required modules for job's environment
module load mamba/latest
# Using python, so source activate an appropriate environment
source activate rafenv


num_repeat=1
# datasets="rsna vin brain lag isic c16 brats"
datasets="rsna vin brain lag isic c16 "
# methods="ae ae-l1 ae-ssim ae-perceptual ae-spatial vae constrained-ae memae ceae ganomaly aeu ae-grad vae-rec vae-combi"
methods="ae ae-l1"
gpu=0

printf "Starting testing phase\n"
# for data in $datasets;do
#   for method in $methods;do
#     for((i=0;i<num_repeat;i=i+1));do
#       python train.py -d "$data" -m "$method" -g $gpu -f "$i";
#       python test.py -d "$data" -m "$method" -g $gpu -f "$i" -save;
#     done
#   done

#   # for((i=0;i<num_repeat;i=i+1));do
#   #     python train.py -d "$data" -m dae -g $gpu --input-size 128 -bs 16 -f "$i";
#   #     python test.py -d "$data" -m dae -g $gpu --input-size 128 -f "$i" -save;
#   # done
# done


# for data in $datasets;do
#   for method in $methods;do
#     for((i=0;i<num_repeat;i=i+1));do
#       python test.py -d "$data" -m "$method" -g $gpu -f 0 --input-size 128;
#     done
#   done
# done
