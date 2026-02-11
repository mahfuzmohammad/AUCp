# shellcheck disable=SC1009
# shellcheck disable=SC1061
# shellcheck disable=SC1073

num_repeat=3
#data='rsna'
gpu=5

# shellcheck disable=SC2004
for((i=0;i<$num_repeat;i=i+1));do
python train_fae.py -d isic -g $gpu --loss_fn mse;
python train_fae.py -d brats -g $gpu --loss_fn mse;
done
