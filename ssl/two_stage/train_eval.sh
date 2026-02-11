datasets="rsna vin brain lag isic c16 brats"
# datasets="rsna"
gpu_id=0
methods="normal 3way anatpaste"
# methods="anatpaste"

for data in $datasets;do
  for method in $methods;do
    python run_training.py --variant "$method" --type "$data" --no-pretrained --cuda $gpu_id;
    # python resnetIN_run_training.py --variant "$method" --type "$data" --cuda $gpu_id;
  done
done
