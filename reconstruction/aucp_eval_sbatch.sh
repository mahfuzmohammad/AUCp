#!/bin/bash
#SBATCH -N 1            
#SBATCH -c 16         
#SBATCH -G 1      
#SBATCH -t 7-00:00:00    
#SBATCH -p general      
#SBATCH -q public      
#SBATCH -o logs/slurm.%j.out 
#SBATCH -e logs/slurm.%j.err 

# Load required modules for job's environment
module load mamba/latest
# Using python, so source activate an appropriate environment
source activate aucp

data=$1
method=$2
fold=$3
gpu=$4

echo "Dataset: $data"
echo "Method: $method"
echo "Fold: $fold"
echo "GPU: $gpu"

python test.py -d "$data" -m "$method" -g "$gpu" -f "$fold" -save
