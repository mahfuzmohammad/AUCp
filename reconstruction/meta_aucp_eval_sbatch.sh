#!/bin/bash

# datasets="rsna vin brain lag isic c16 brats"
datasets="isic brats"
methods="memae aeu"
num_repeat=1
gpu=0

mkdir -p logs

for data in $datasets; do
  for method in $methods; do
    for ((i=0; i<num_repeat; i++)); do
      jobid=$(sbatch aucp_eval_sbatch.sh "$data" "$method" "$i" "$gpu" | awk '{print $4}')
      echo "Submitted job for $data $method fold=$i → Slurm ID: $jobid"
    done
  done
done
